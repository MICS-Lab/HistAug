import importlib
import inspect

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.stats import bootstrap

from models.foundation_models import get_foundation_model
from utils.loss_factory import create_loss
from utils.optim_factory import create_optimizer, create_scheduler


class ModelInterface(pl.LightningModule):
    """
    PyTorch Lightning module wrapping a foundation-based adaptation model.

    :param model: Configuration dict for the adaptation model architecture.
    :param loss: Loss configuration for training objective.
    :param optimizer: Optimizer configuration dict.
    :param scheduler: Learning rate scheduler configuration dict.
    :param transforms: Data augmentation and preprocessing parameters.
    """

    def __init__(
        self,
        model,
        loss,
        optimizer,
        scheduler,
        transforms,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Hyperparameter containers
        self.optimizer_params = optimizer
        self.scheduler_params = scheduler
        self.transforms = transforms

        # Instantiate components
        self.load_model()
        self.loss_fn = create_loss(loss)

        # Feature extractor
        self.feature_extractor = get_foundation_model(
            self.hparams.foundation_model,
            device=self.device,
        )
        # Structured containers for metrics
        self.cosine_stats = {
            phase: {"imgaug_sum": 0.0, "id_sum": 0.0, "orig_trans_sum": 0.0, "count": 0}
            for phase in ("train", "val", "test")
        }
        self.losses = {phase: [] for phase in ("train", "val", "test")}
        self._test_imgaug_cos_values = []

    def get_progress_bar_dict(self) -> dict:
        """
        Customize the progress bar to hide version number information.

        :return: Dictionary of progress bar metrics.
        """
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lightning training step.

        :param batch: Tuple containing (original_patch, transformed_patch, aug_params).
        :param batch_idx: Index of the current batch.
        :return: Computed loss tensor for backpropagation.
        """
        return self._shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lightning validation step.

        :param batch: Tuple containing (original_patch, transformed_patch, aug_params).
        :param batch_idx: Index of the current batch.
        :return: Computed loss tensor for validation.
        """
        return self._shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Lightning test step.

        :param batch: Tuple containing (original_patch, transformed_patch, aug_params).
        :param batch_idx: Index of the current batch.
        :return: Computed loss tensor for testing.
        """
        return self._shared_step(batch, batch_idx, phase="test")

    def _shared_step(self, batch: tuple, batch_idx: int, phase: str) -> torch.Tensor:
        """
        Shared logic for train/val/test steps: feature extraction, forward pass, loss and metric computation.

        :param batch: Tuple of (patch_orig, patch_trans, aug_params).
        :param batch_idx: Index of the batch.
        :param phase: One of 'train', 'val', 'test'.
        :return: Combined loss for image-augmentation and identity prediction.
        """
        patch_orig, patch_trans, aug_params = batch
        patch_orig = patch_orig.to(self.device)
        patch_trans = patch_trans.to(self.device)

        with torch.amp.autocast("cuda"):
            feats_orig = self.feature_extractor(patch_orig)
            feats_trans = self.feature_extractor(patch_trans)

            # Predict transformed features
            pred_imgaug = self.model(feats_orig, aug_params["img_aug"])
            loss_imgaug = self.loss_fn(pred_imgaug, feats_trans)

            # Predict identity features
            pred_id = self.model(feats_orig, aug_params["id"])
            loss_id = self.loss_fn(pred_id, feats_orig)

            batch_loss = loss_imgaug + loss_id

            # Cosine similarities per sample
            cos_imgaug = F.cosine_similarity(pred_imgaug, feats_trans)
            cos_id = F.cosine_similarity(pred_id, feats_orig)
            cos_orig_trans = F.cosine_similarity(feats_orig, feats_trans)

        if phase == "test":
            self._test_imgaug_cos_values.append(cos_imgaug.detach().float().cpu())

        # Accumulate sums and counts in dict
        stats = self.cosine_stats[phase]
        stats["imgaug_sum"] += cos_imgaug.sum().item()
        stats["id_sum"] += cos_id.sum().item()
        stats["orig_trans_sum"] += cos_orig_trans.sum().item()
        stats["count"] += cos_imgaug.numel()
        # Store for epoch aggregations
        self.losses[phase].append(batch_loss.detach())
        return batch_loss

    def _epoch_end(self, phase: str) -> None:
        """
        Aggregate losses and cosine similarities at epoch end and log metrics.

        :param phase: One of 'train', 'val', 'test'.
        :return: None.
        """
        # Aggregate loss
        losses = self.losses[phase]
        mean_loss = torch.stack(losses).mean().item()

        # Compute mean similarities
        stats = self.cosine_stats[phase]
        count = stats["count"]
        mean_imgaug_cos = stats["imgaug_sum"] / count
        mean_id_cos = stats["id_sum"] / count
        mean_orig_cos = stats["orig_trans_sum"] / count

        metrics = {
            f"{phase}/epoch_loss": mean_loss,
            f"{phase}/mean_imgaug_cos": mean_imgaug_cos,
            f"{phase}/mean_id_cos": mean_id_cos,
            f"{phase}/mean_origtrans_cos": mean_orig_cos,
        }

        if phase == "test":
            if len(self._test_imgaug_cos_values) > 0:
                vals = torch.cat(
                    self._test_imgaug_cos_values, dim=0
                ).numpy()  # 1D array
                if vals.size > 0:
                    res = bootstrap(
                        (vals,),
                        np.mean,
                        confidence_level=0.95,
                        n_resamples=3000,
                        vectorized=False,
                        random_state=None,
                    )
                    metrics["test/mean_imgaug_cos_ci_low"] = float(
                        res.confidence_interval.low
                    )
                    metrics["test/mean_imgaug_cos_ci_high"] = float(
                        res.confidence_interval.high
                    )

        # Consolidated logging
        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=(phase in ["train", "val"]),
            logger=True,
            sync_dist=True,
        )

        # Reset for next epoch
        self.losses[phase].clear()
        self.cosine_stats[phase] = {
            "imgaug_sum": 0.0,
            "id_sum": 0.0,
            "orig_trans_sum": 0.0,
            "count": 0,
        }

    def on_train_epoch_end(self) -> None:
        """
        Hook called at the end of training epoch to perform epoch-level logging.
        """
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        """
        Hook called at the end of validation epoch to perform epoch-level logging.
        """
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        """
        Hook called at the end of testing epoch to perform epoch-level logging.
        """
        self._epoch_end("test")

    def configure_optimizers(self):
        """
        Set up optimizer and optional scheduler for training.

        :return: Single or tuple of lists: [optimizers], [schedulers] if scheduler exists.
        """
        optimizer = create_optimizer(self.optimizer_params, self.model)
        lr_scheduler = create_scheduler(self.scheduler_params, optimizer)

        if lr_scheduler is None:
            return [optimizer]
        return [optimizer], [lr_scheduler]

    def named_children(self):
        for name, module in super().named_children():
            if name == "loss_fn":
                continue
            yield name, module

    def load_model(self) -> None:
        """
        Dynamically import and instantiate the model class defined in hyperparameters.

        :raises ValueError: If module or class name is invalid.
        """
        name = self.hparams.model.name
        model_class_name = "".join(part.capitalize() for part in name.split("_"))
        try:
            module = importlib.import_module(f"models.{name}")
            Model = getattr(module, model_class_name)
        except (ImportError, AttributeError):
            raise ValueError("Invalid model module or class name")

        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """
        Instantiate a model using matching hyperparameters and additional arguments.

        :param Model: Class of the model to instantiate.
        :param other_args: Additional keyword arguments to override defaults.
        :return: An instance of the given Model.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        args1["transforms"] = self.transforms

        return Model(**args1)

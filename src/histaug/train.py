import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import logging
from collections import OrderedDict
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets import DataInterface
from models import ModelInterface
from pytorch_lightning import Trainer
from utils.utils import load_callbacks, load_loggers, print_run_summary, read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and Test Pipeline with PyTorch Lightning"
    )
    parser.add_argument(
        "--stage",
        choices=["train", "test"],
        default="train",
        help="Execution stage: train or test",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration file"
    )
    return parser.parse_args()


def configure(cfg) -> None:
    """
    Set seeds, logging, callbacks, and normalize and general settings.
    """
    pl.seed_everything(cfg.General.seed)
    loggers = load_loggers(cfg)
    callbacks = load_callbacks(cfg)

    # General defaults
    cfg.General.devices = cfg.General.devices or 1
    cfg.General.accelerator = getattr(cfg.General, "accelerator", "gpu")
    cfg.General.strategy = getattr(cfg.General, "strategy", "auto")

    return loggers, callbacks


def build_datamodule(cfg) -> pl.LightningDataModule:
    """
    Instantiate the LightningDataModule based on configuration.

    :param cfg: Configuration object containing Data and Model parameters.
    :return: Configured DataInterface instance.
    """
    return DataInterface(
        train_batch_size=cfg.Data.train_dataloader.batch_size,
        train_num_workers=cfg.Data.train_dataloader.num_workers,
        test_batch_size=cfg.Data.test_dataloader.batch_size,
        test_num_workers=cfg.Data.test_dataloader.num_workers,
        dataset_name=cfg.Data.dataset_name,
        shuffle_data=cfg.Data.shuffle_data,
        transforms=cfg.Data.Transforms,
        dataset_cfg=cfg.Data,
        general=cfg.General,
        model=cfg.Model,
        foundation_model=cfg.Foundation_model,
    )


def build_model(cfg) -> pl.LightningModule:
    """
    Instantiate the LightningModule for training or testing.

    :param cfg: Configuration object containing Model, Loss, Optimizer, Scheduler, and Data settings.
    :return: Configured ModelInterface instance.
    """
    return ModelInterface(
        general=cfg.General,
        model=cfg.Model,
        loss=cfg.Loss,
        optimizer=cfg.Optimizer,
        scheduler=cfg.Scheduler,
        transforms=cfg.Data.Transforms,
        data=cfg.Data,
        log=cfg.log_path,
        foundation_model=cfg.Foundation_model,
    )


def build_trainer(cfg, loggers, callbacks) -> Trainer:
    """
    Create a PyTorch Lightning Trainer using the configuration.

    :param cfg: Configuration object with keys:
                - load_loggers: list of loggers
                - callbacks: list of callbacks
                - General.epochs, devices, accelerator, strategy, precision, grad_acc
    :return: Configured Trainer instance.
    """

    return Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=cfg.General.epochs,
        devices=cfg.General.devices,
        accelerator=cfg.General.accelerator,
        strategy=cfg.General.strategy,
        precision=cfg.General.precision,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        accumulate_grad_batches=cfg.General.get("grad_acc", 1),
        num_sanity_val_steps=0,
    )


def run_training(
    trainer: Trainer, model: pl.LightningModule, datamodule: pl.LightningDataModule, cfg
) -> None:
    """
    Run the training loop

    :param trainer: Trainer instance.
    :param model: ModelInterface instance.
    :param datamodule: DataInterface instance.
    :param cfg: Configuration object possibly containing resume checkpoint path.
    """
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=cfg.General.get("ckpt_path_resume_training", None),
    )


def run_testing(
    trainer: Trainer, model: pl.LightningModule, datamodule: pl.LightningDataModule, cfg
) -> None:
    """
    Run the testing loop on all checkpoints in the log directory.

    For each .ckpt file (excluding 'temp'), load the model and call Trainer.test.

    :param trainer: Trainer instance.
    :param model: Base ModelInterface class (used to load from checkpoint).
    :param datamodule: DataInterface instance.
    :param cfg: Configuration object with attribute log_path.
    """
    ckpt_dir = Path(cfg.log_path)
    ckpt_paths = [p for p in ckpt_dir.glob("*/*.ckpt") if "temp" not in p.name]
    for ckpt in ckpt_paths:
        logging.info(f"Testing checkpoint: {ckpt}")
        test_model = model.__class__.load_from_checkpoint(
            checkpoint_path=ckpt, cfg=cfg, strict=False, weights_only=False
        )
        test_model.current_model_path = ckpt
        trainer.test(model=test_model, datamodule=datamodule)


def main() -> None:
    """
    Main entrypoint: parse arguments, load config, build components, and execute train or test.
    """
    args = parse_args()
    cfg = read_yaml(args.config)
    cfg.config = str(args.config)
    cfg.General.server = args.stage

    loggers, callbacks = configure(cfg)
    print_run_summary(cfg)
    datamodule = build_datamodule(cfg)
    model = build_model(cfg)
    trainer = build_trainer(cfg, loggers, callbacks)

    if cfg.General.server == "train":
        run_training(trainer, model, datamodule, cfg)
    else:
        run_testing(trainer, model, datamodule, cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Dict

import timm
import torch
from conch.open_clip_custom import create_model_from_pretrained
from torchvision import transforms
from transformers import AutoModel
from utils.constants import NormConstants

_BACKENDS: Dict[str, "FoundationBackend"] = {}


def register(cls):
    _BACKENDS[cls.NAME.upper()] = cls()
    return cls


class FoundationBackend(ABC):
    """
    Abstract base for all foundation model backends.

    Subclasses must implement the _build_model method.
    """

    NAME: str = ""
    IMG_SIZE: int = 224
    MEAN = NormConstants.IMAGENET_MEAN.value
    STD = NormConstants.IMAGENET_STD.value

    @abstractmethod
    def _build_model(
        self, ckpt_path: str | None, device: torch.device
    ) -> torch.nn.Module:
        """
        Create the *raw* model (weights NOT frozen, NOT on device).

        :param ckpt_path: Path to the model checkpoint. If None, load pretrained weights.
        :param device: Device on which the model will be used.
        :return: A torch.nn.Module instance of the model.
        """

    def get_model(self, ckpt_path: str | None, device: torch.device) -> torch.nn.Module:
        """
        Load, freeze, and prepare the foundation model for inference.

        :param ckpt_path: Path to the model checkpoint. If None, use default pretrained weights.
        :param device: Device to move the model to (e.g., 'cpu' or 'cuda').
        :return: A frozen torch.nn.Module in evaluation mode.
        """
        model = self._build_model(ckpt_path).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def get_transform(self) -> transforms.Compose:
        """
        Get the preprocessing transformation for the foundation model.

        :return: A tuple of (transform pipeline, image size) for input preprocessing.
        """
        return (
            transforms.Compose(
                [
                    transforms.Resize(self.IMG_SIZE),
                    transforms.CenterCrop(self.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.MEAN, std=self.STD),
                ]
            ),
            self.IMG_SIZE,
        )


@register
class UNI(FoundationBackend):
    NAME = "UNI"
    IMG_SIZE = 224

    def _build_model(self, ckpt_path: str | None):
        """
        Build the UNI vision transformer model.

        :param ckpt_path: Local checkpoint path. If provided, load from disk.
        :return: Initialized ViT model without classification head.
        """
        if ckpt_path is not None:
            model = timm.create_model(
                "vit_large_patch16_224",
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
            )
            model.load_state_dict(
                torch.load(ckpt_path, map_location="cuda"), strict=True
            )
        else:
            model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )

        return model


@register
class CONCH(FoundationBackend):
    NAME = "CONCH"
    IMG_SIZE = 448
    MEAN = NormConstants.OPENAI_MEAN.value  # different mean
    STD = NormConstants.OPENAI_STD.value  # different std

    def _build_model(self, ckpt_path: str | None):
        """
        Build the CONCH vision model and override forward pass.

        :param ckpt_path: Checkpoint path or None to use hub weights.
        :return: Model with forward bound to encode_image without contrast projection.
        """
        if ckpt_path is not None:
            model, preprocess = create_model_from_pretrained(
                "conch_ViT-B-16", ckpt_path
            )
        else:
            model, _ = create_model_from_pretrained(
                "conch_ViT-B-16", checkpoint_path="hf_hub:MahmoodLab/conch"
            )
        model.forward = partial(
            model.encode_image, proj_contrast=False, normalize=False
        )
        return model

    def get_transform(self) -> transforms.Compose:
        """
        Get the OpenAI-specific preprocessing for CONCH.

        :return: A tuple of (transform pipeline, image size).
        """
        return (
            transforms.Compose(
                [
                    transforms.Resize(
                        self.IMG_SIZE,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(self.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.MEAN, std=self.STD),
                ]
            ),
            self.IMG_SIZE,
        )


@register
class UNI2H(FoundationBackend):
    NAME = "UNI-2H"
    IMG_SIZE = 224

    def _build_model(self, ckpt_path: str | None):
        """
        Build the UNI-2H vision transformer with specified architecture.

        :param ckpt_path: Checkpoint to load weights from disk, ignored for hub.
        :param device: Device context (not used in building).
        :return: A timm ViT model with custom depth, heads, and mlp configuration.
        """

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        if ckpt_path is not None:
            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", pretrained=False, **timm_kwargs
            )
            model.load_state_dict(
                torch.load(ckpt_path, map_location="cuda"), strict=True
            )
        else:
            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
            )
        return model


@register
class CONCH_V1_5(FoundationBackend):
    NAME = "CONCH_V1_5"
    IMG_SIZE = 448

    def _build_model(self, ckpt_path: str | None):
        """
        Build the TITAN backbone from HuggingFace and extract CONCH model.

        :param ckpt_path: Path to the root directory of the TITAN HuggingFace repository (not a single checkpoint file).
        :return: Extracted CONCH model from TITAN.
        """
        if ckpt_path is not None:
            titan = AutoModel.from_pretrained(
                ckpt_path, trust_remote_code=True, local_files_only=True
            )
        else:
            titan = AutoModel.from_pretrained(
                "MahmoodLab/TITAN", trust_remote_code=True
            )

        model, _ = titan.return_conch()

        return model


@register
class H_OPTIMUS_1(FoundationBackend):
    NAME = "H-OPTIMUS-1"
    IMG_SIZE = 224
    MEAN = (0.707223, 0.578729, 0.703617)
    STD = (0.211883, 0.230117, 0.177517)

    def _build_model(self, ckpt_path: str | None):
        """
        Build the H-optimus-1 vision transformer.

        :param ckpt_path: Optional local checkpoint path. If provided, load from disk.
        :return: Initialized model ready for feature extraction.
        """
        if ckpt_path is not None:
            model = timm.create_model(
                "hf-hub:bioptimus/H-optimus-1",
                pretrained=False,
                init_values=1e-5,
                dynamic_img_size=False,
            )
            model.load_state_dict(
                torch.load(ckpt_path, map_location="cuda"), strict=True
            )
        else:
            model = timm.create_model(
                "hf-hub:bioptimus/H-optimus-1",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
            )
        return model

    def get_transform(self) -> transforms.Compose:
        """
        Preprocessing for H-optimus-1.
        """
        return (
            transforms.Compose(
                [
                    transforms.Resize(self.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.MEAN, std=self.STD),
                ]
            ),
            self.IMG_SIZE,
        )


@register
class VIRCHOW2(FoundationBackend):
    NAME = "VIRCHOW2"
    IMG_SIZE = 224  # Virchow2 is 224x224

    def _build_model(self, ckpt_path: str | None):
        """
        Build paige-ai/Virchow2 and override forward to return a 2560-D embedding:
        concat(class_token, mean(patch_tokens)).
        """
        # Need these for proper init as per the reference snippet
        timm_kwargs = dict(mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)

        if ckpt_path is not None:
            model = timm.create_model(
                "hf-hub:paige-ai/Virchow2", pretrained=False, **timm_kwargs
            )
            model.load_state_dict(
                torch.load(ckpt_path, map_location="cuda"), strict=True
            )
        else:
            model = timm.create_model(
                "hf-hub:paige-ai/Virchow2", pretrained=True, **timm_kwargs
            )

        # Keep the original forward that returns token embeddings [B, 261, 1280]
        _orig_forward = model.forward

        def _forward_embedding(x: torch.Tensor) -> torch.Tensor:
            """
            Returns a 2560-D embedding: [cls_token || mean(patch_tokens)].
            Assumes tokens: 0=CLS, 1-4=register, 5: = patches (256 tokens).
            """
            out = _orig_forward(x)  # [B, 261, 1280]
            cls_tok = out[:, 0]  # [B, 1280]
            patch_toks = out[:, 5:]  # [B, 256, 1280]
            pooled = patch_toks.mean(dim=1)  # [B, 1280]
            return torch.cat([cls_tok, pooled], dim=-1)  # [B, 2560]

        # Replace forward with the embedding-producing one
        model.forward = _forward_embedding

        return model

    def get_transform(self):
        return (
            transforms.Compose(
                [
                    transforms.Resize(
                        self.IMG_SIZE,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.CenterCrop(self.IMG_SIZE),
                    transforms.ToTensor(),  # replaces "MaybeToTensor()" for torchvision
                    transforms.Normalize(mean=self.MEAN, std=self.STD),
                ]
            ),
            self.IMG_SIZE,
        )


def _get_backend(name: str) -> FoundationBackend:
    """
    Retrieve a registered foundation backend by name.

    :param name: Identifier of the backend (case-insensitive).
    :return: An instance of FoundationBackend.
    :raises ValueError: If no backend is registered under the given name.
    """

    try:
        return _BACKENDS[name.upper()]
    except KeyError as e:
        raise ValueError(f"Unknown foundation model: {name!r}") from e


def get_foundation_model(params: dict, device: torch.device) -> torch.nn.Module:
    """
    Return a frozen, evaluation-ready feature extractor.

    :param params: Dictionary containing 'name' and optional 'ckpt_path'.
    :param device: Device to which the model should be moved.
    :return: Frozen torch.nn.Module in eval mode.
    """
    backend = _get_backend(params.get("name", ""))
    return backend.get_model(params.get("ckpt_path"), device)


def get_fm_transform(params: dict):
    """
    Return preprocessing transform pipeline for a foundation model.

    :param params: Dictionary containing 'name'.
    :return: A tuple of (transform pipeline, image size).
    """
    backend = _get_backend(params.get("name", ""))
    return backend.get_transform()

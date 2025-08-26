import random
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Union

import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from kornia.morphology import dilation, erosion
from PIL import Image
from torchvision.transforms import functional as TF
from utils.hed import HEDAugmentation
from utils.utils import check_parameters_validity

TransformFactory = Callable[[Mapping[str, Any]], pl.LightningModule]
TRANSFORM_REGISTRY: Dict[str, TransformFactory] = {}


def register_transform(name: str):
    """
    Decorator to register a transform class in the TRANSFORM_REGISTRY.

    :param name: Key under which to register the transform.
    """

    def decorator(factory: TransformFactory) -> TransformFactory:
        TRANSFORM_REGISTRY[name] = factory
        return factory

    return decorator


@register_transform("PatchAugmentation")
class PatchAugmentation(pl.LightningModule):
    """
    LightningModule that samples and applies a sequence of random augmentations to image patches.

    :param transforms_params: Mapping of augmentation names to their configuration values.
                             Supported keys: 'rotation', 'crop', 'h_flip', 'v_flip',
                             'gaussian_blur', 'erosion', 'dilation',
                             'brightness', 'contrast', 'saturation', 'hue', 'gamma', 'hed'.
    :param single_augmentation (bool, optional): If True, apply only one augmentation per call.
    """

    def __init__(self, transforms_params, single_transform_mode=False, **kwargs):
        super(PatchAugmentation, self).__init__()

        self.augmentations_params = transforms_params

        if "hed" in self.augmentations_params.keys():
            self.hed = HEDAugmentation()

        self.aug_param_names = sorted(self.augmentations_params.keys())
        self.single_transform_mode = single_transform_mode

    def sample_aug_params(self) -> OrderedDict:
        """
        Sample random augmentation parameters and their application order.

        :return: OrderedDict mapping each augmentation name to (param_value, position_index).
                 Position index indicates the shuffled sequence in which transforms will apply.
        """
        # 1) make a shuffled copy of the canonical (alphabetical) names
        shuffle_order = self.aug_param_names.copy()
        random.shuffle(shuffle_order)

        aug_params = OrderedDict()
        # 2) sample each param and record its position in the shuffle
        for pos, name in enumerate(shuffle_order):
            cfg = self.augmentations_params[name]
            if name == "rotation":
                p = random.randint(1, 3) if random.random() < cfg else 0
            elif name == "crop":
                p = random.randint(1, 5) if random.random() < cfg else 0
            elif name in ("h_flip", "v_flip", "gaussian_blur", "erosion", "dilation"):
                p = int(random.random() < cfg)
            elif name in (
                "brightness",
                "contrast",
                "saturation",
                "hue",
                "gamma",
                "hed",
            ):
                p = random.uniform(cfg[0], cfg[1])
            else:
                raise ValueError(f"{name} is not a valid augmentation parameter name")
            # store both the sampled value and its shuffle position
            aug_params[name] = (p, pos)

        return aug_params

    def get_identity_aug_params(self) -> OrderedDict:
        """
        Generate augmentation parameters corresponding to the identity transform (all zeros).

        :return: OrderedDict mapping augmentation names to (0, position_index) shuffled.
        """

        aug_param_names = list(self.augmentations_params.keys())
        random.shuffle(aug_param_names)
        id_aug_params = OrderedDict({})

        for pos, aug_param_name in enumerate(aug_param_names):
            id_aug_params[aug_param_name] = (0, pos)
        return id_aug_params

    def apply_transform(self, img, aug_params) -> Union[Image.Image, torch.Tensor]:
        """
        Apply a series of augmentations to an image based on sampled parameters.

        :param img: Input image (PIL Image or Tensor of shape (C, H, W)).
        :param aug_params: OrderedDict from sample_aug_params or get_identity_aug_params.
        :return: Augmented image in the same type as input.
        """
        for aug_param_name, (param, pos) in aug_params.items():
            if aug_param_name == "rotation":
                if param:
                    if isinstance(img, Image.Image):
                        img = TF.rotate(img=img, angle=param * 90)
                    elif isinstance(img, torch.Tensor):
                        img = torch.rot90(img, k=param, dims=[-2, -1])
            elif aug_param_name == "h_flip" and param:
                img = TF.hflip(img)
            elif aug_param_name == "v_flip" and param:
                img = TF.vflip(img)
            elif aug_param_name == "gaussian_blur" and param:
                img = TF.gaussian_blur(img, kernel_size=15)
            elif aug_param_name == "hed" and param:
                img = self.hed(img, param)
            elif aug_param_name == "crop" and param:
                img = TF.five_crop(img, size=128)
                img = img[param - 1]
            elif aug_param_name == "erosion" and param:
                if isinstance(img, Image.Image):
                    img = TF.to_tensor(img).unsqueeze(0).float()
                img = erosion(img, torch.ones(4, 4).to(img.device)).squeeze(0)
                if isinstance(img, torch.Tensor):
                    img = TF.to_pil_image(img)
            elif aug_param_name == "dilation" and param:
                if isinstance(img, Image.Image):
                    img = TF.to_tensor(img).unsqueeze(0).float()
                img = dilation(img, torch.ones(4, 4).to(img.device)).squeeze(0)
                if isinstance(img, torch.Tensor):
                    img = TF.to_pil_image(img)
            elif aug_param_name == "brightness" and param:
                img = TF.adjust_brightness(img, brightness_factor=1 + param)
            elif aug_param_name == "contrast" and param:
                img = TF.adjust_contrast(img, contrast_factor=1 + param)
            elif aug_param_name == "saturation" and param:
                img = TF.adjust_saturation(img, saturation_factor=1 + param)
            elif aug_param_name == "hue" and param:
                img = TF.adjust_hue(img, hue_factor=param)
            elif aug_param_name == "gamma" and param:
                img = TF.adjust_gamma(img, gamma=1 + param)
        return img

    def _sample_nonzero_param(self, name: str, cfg):
        """Return a non-identity parameter for the given transform name."""
        if name == "rotation":
            return random.randint(1, 3)  # 90/180/270
        if name == "crop":
            return random.randint(1, 5)  # one of five-crop indices
        if name in ("h_flip", "v_flip", "gaussian_blur", "erosion", "dilation"):
            return 1  # flip/blur/morph on
        if name in ("brightness", "contrast", "saturation", "hue", "gamma", "hed"):
            low, high = cfg
            val = random.uniform(low, high)
            return val

        raise ValueError(f"{name} is not a valid augmentation parameter name")

    def _build_single_transform_mode_params(self) -> OrderedDict:
        """
        Start from identity for all transforms, then force exactly one transform
        to be non-identity. Preserves the shuffled order via identity params.
        """
        params = self.get_identity_aug_params()
        chosen = random.choice(list(params.keys()))
        cfg = self.augmentations_params[chosen]
        _, pos = params[chosen]
        params[chosen] = (self._sample_nonzero_param(chosen, cfg), pos)
        return params

    def __call__(self, img: Union[Image.Image, torch.Tensor]):
        """
        Generate an augmented and identity version of the input image along with their parameters.

        :param img: Input image (PIL or Tensor).
        :return: Tuple of (augmented_image, params_dict) where params_dict has keys 'img_aug' and 'id'.
        """

        if self.single_transform_mode:
            aug_params = self._build_single_transform_mode_params()
        else:
            aug_params = self.sample_aug_params()

        img_aug = self.apply_transform(img, aug_params)
        id_aug_params = self.get_identity_aug_params()

        return img_aug, {"img_aug": aug_params, "id": id_aug_params}


def create_transform(transforms=None):
    """
    Factory function to instantiate a transform LightningModule from configuration.

    :param transforms: Dictionary containing:
        - transform_class (str): Key in TRANSFORM_REGISTRY.
        - parameters (dict): Parameters to pass to the transform.
        - single_augmentation (bool, optional): If True, apply only one augmentation per call.
    :return: An instance of the selected transform module.

    :raises ValueError: If configuration is invalid or transform_class not registered.
    """
    if transforms is None:
        raise ValueError(
            "Transforms must be provided to train Histaug. Without them, the model will likely learn the identity transformation."
        )

    transform_class = transforms.get("transform_class")
    transforms_parameters = transforms.get("parameters")
    single_augmentation = transforms.get(
        "single_augmentation", False
    )  # new optional key

    if transform_class not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform_class '{transform_class}'. "
            f"Available options: {list(TRANSFORM_REGISTRY)}"
        )

    check_parameters_validity(transforms_parameters)

    # Pass the flag to the transform constructor
    transform = TRANSFORM_REGISTRY[transform_class](
        transforms_parameters, single_transform_mode=single_augmentation
    )

    return transform

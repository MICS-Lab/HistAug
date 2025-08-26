import random
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
import torch.utils.data as data
from models.foundation_models import get_fm_transform
from utils.transform_factory import create_transform


class PatchDataset(data.Dataset):
    """
    PyTorch Dataset for sampling image patches from whole-slide images (WSIs).

    Patches are located via HDF5 coordinate files generated (e.g., by the CLAM toolbox),
    then loaded, augmented, and preprocessed for a given foundation model.

    :param dataset_cfg: Configuration object containing dataset root path, file extension, and patching directories.
    :param state: Dataset split identifier ('train', 'val', 'test').
    :param transforms: Dictionary of augmentation parameters to create the augmentation pipeline.
    :param foundation_model: Dict specifying the foundation model name and any ckpt path, used for preprocessing.
    """

    def __init__(
        self,
        dataset_cfg,
        state="train",
        transforms=None,
        foundation_model=None,
    ):
        if dataset_cfg is None:
            raise ValueError("`dataset_cfg` must be provided")

        if not foundation_model:
            raise ValueError("Foundation model must be provided")
        # Paths and file extensions
        self.dataset_root = Path(dataset_cfg.data_path)
        self.file_extension = str(dataset_cfg.data_path_extension).lstrip(".")

        # Store model and create preprocessing pipeline
        self.foundation_model = foundation_model
        self.transforms = create_transform(transforms)

        # Load mapping from dataset names to patch directories
        self.dataset_folder_map = dataset_cfg.patching
        self.metadata_csv = Path(f"./dataset_csv/{state}.csv")
        self._load_slide_metadata(state)

        # Get image preprocessing transform and patch size
        self.image_preprocessing_pipeline, self.patch_size = get_fm_transform(
            self.foundation_model
        )

    def _load_slide_metadata(self, state: str) -> None:
        """
        Load slide identifiers and dataset sources from metadata CSV for given split.

        :param state: One of 'train', 'val', or 'test'.
        """
        metadata_df = pd.read_csv(str(self.metadata_csv).format(state=state))
        self.slide_ids = metadata_df.slide_id.values
        self.dataset_sources = metadata_df.dataset.values

    def __len__(self) -> int:
        """
        :return: Number of WSIs in this split (number of slide IDs).
        """
        return len(self.slide_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Sample a random patch from the WSI at the given index.

        :param index: Index of the slide in the dataset split.
        :return: Tuple of (original_tensor, augmented_tensor, augmentation_params).
        """
        slide_id = self.slide_ids[index]
        dataset_name = self.dataset_sources[index]

        slide_filepath = self.dataset_root / f"{slide_id}.{self.file_extension}"
        coord_file_path = self._resolve_h5_path(slide_id, dataset_name)

        coordinates, patch_level = self._load_patch_coords(coord_file_path)
        selected_index = random.randrange(len(coordinates))

        return self._load_patch(
            slide_filepath, coordinates[selected_index], patch_level, self.patch_size
        )

    def _resolve_h5_path(self, slide_id: str, dataset_name: str) -> Path:
        """
        Get the HDF5 file path containing patch coordinates for a slide.

        :param slide_id: Identifier of the WSI.
        :param dataset_name: Key to map into dataset_folder_map.
        :return: Path to the .h5 coordinate file.
        :raises KeyError: If dataset_name not in folder map.
        """
        patching_dir = self.dataset_folder_map.get(dataset_name)
        if not patching_dir:
            raise KeyError(f"Unknown dataset key: {dataset_name}")
        return Path(patching_dir) / f"{slide_id}.h5"

    def _load_patch_coords(self, h5_file: Path) -> Tuple[np.ndarray, int, int]:
        """
        Load patch coordinates and patch level from an HDF5 file. The format
        applied here corresponds to patches extracted using the CLAM toolbox
        (https://github.com/mahmoodlab/CLAM/).

        :param h5_file: Path to the HDF5 file storing 'coords' dataset and its attributes.
        :return: (coordinates array of shape [N, 2], patch_level integer).
        """
        with h5py.File(h5_file, "r") as f:
            coordinates = f["coords"][()]
            patch_level = f["coords"].attrs["patch_level"]

        return coordinates, patch_level

    def _load_patch(
        self,
        slide_path: Path,
        coord: np.ndarray,
        level: int,
        size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Extract, augment, and preprocess a patch from the whole-slide image.

        :param slide_path: Filesystem path to the WSI.
        :param coord: (x, y) pixel coordinates at which to extract the patch.
        :param level: Pyramid level at which to extract.
        :param size: Size (width and height) of the square patch.
        :return: Tuple containing:
                 - original_tensor: Preprocessed patch without augmentation.
                 - augmented_tensor: Preprocessed patch after augmentation.
                 - augmentation_params: Dict of applied augmentation parameters.
        """
        # Read patch region
        with openslide.OpenSlide(str(slide_path)) as slide:
            image_region = slide.read_region(tuple(coord), level, (size, size)).convert(
                "RGB"
            )
        # Apply augmentation and capture params
        augmented_image, augmentation_params = self.transforms(image_region)

        # Preprocess for foundation model input
        original_tensor = self.image_preprocessing_pipeline(image_region)
        augmented_tensor = self.image_preprocessing_pipeline(augmented_image)

        return original_tensor, augmented_tensor, augmentation_params

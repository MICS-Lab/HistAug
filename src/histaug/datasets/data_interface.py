import importlib
import inspect
import math

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, Subset


class DataInterface(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for dynamic dataset instantiation and standardized dataloader creation.

    This module loads a dataset class by name, instantiates training, validation, and testing datasets,
    and provides DataLoaders with consistent behavior (shuffling, batching, and worker management).

    :param train_batch_size: Batch size for training and validation loaders.
    :param train_num_workers: Number of worker processes for training and validation.
    :param test_batch_size: Batch size for testing loader (default often 1 for patch-based evaluation).
    :param test_num_workers: Number of worker processes for testing loader.
    :param shuffle_data: Whether to shuffle training data each epoch.
    :param dataset_name: Name of the dataset module under 'datasets' (e.g., 'my_dataset').
    :param kwargs: Additional keyword arguments passed to the dataset constructor (e.g., data paths, transforms).
    """

    def __init__(
        self,
        train_batch_size=64,
        train_num_workers=8,
        test_batch_size=1,
        test_num_workers=1,
        shuffle_data=True,
        dataset_name=None,
        **kwargs,
    ):
        super().__init__()
        # DataLoader parameters
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.shuffle = shuffle_data
        self.kwargs = kwargs
        self.load_data_module()

    def setup(self, stage: str = None) -> None:
        """
        Instantiate datasets for the given stage.

        :param stage: One of 'fit' or 'test'.
                      'fit' loads train and validation splits;
                      'test' loads test split only.
        :raises ValueError: If stage is not 'fit' or 'test'.
        """
        if stage == "fit":
            self.train_dataset = self.instancialize(state="train")
            self.val_dataset = self.instancialize(state="val")
        elif stage == "test":
            self.test_dataset = self.instancialize(state="test")
        else:
            raise ValueError(
                f"Invalid stage provided: {stage}. Must be either train' or 'test'."
            )

    def train_dataloader(self) -> DataLoader:
        """
        Create DataLoader for training dataset.

        :return: DataLoader yielding batches of training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Create DataLoader for validation dataset.

        :return: DataLoader yielding batches of validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        Create DataLoader for testing dataset, ensuring a fixed number of samples.

        This loader concatenates the test dataset to itself multiple times and
        truncates to exactly 10,000 samples, allowing repeated random sampling
        from each WSI until the quota is met.

        :return: DataLoader yielding 10,000 test samples without shuffling.
        """
        # Determine repetition factor to reach 10,000 samples
        L = len(self.test_dataset)
        repeats = math.ceil(10_000 / L)

        # Concatenate and truncate to 10,000 examples
        long_ds = ConcatDataset([self.test_dataset] * repeats)
        truncated = Subset(long_ds, list(range(10_000)))

        # DataLoader configured to return exactly 10,000 random patches.
        # Each call to __getitem__ draws a fresh random patch from each WSI.
        # By concatenating the dataset multiple times, we ensure we sample
        # enough unique patches with different transformation parameters,
        # then truncate to the first 10,000 examples.

        return DataLoader(
            truncated,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.test_num_workers,
            pin_memory=True,
        )

    def load_data_module(self) -> None:
        """
        Dynamically import the dataset class from 'datasets' package based on dataset_name.

        :raises ValueError: If the module or class cannot be found.
        """
        dataset_class_name = "".join(
            [i.capitalize() for i in (self.dataset_name).split("_")]
        )
        try:
            self.data_module = getattr(
                importlib.import_module(f"datasets.{self.dataset_name}"),
                dataset_class_name,
            )
        except:
            raise ValueError("Invalid Dataset File Name or Invalid Class Name!")

    def instancialize(self, **other_args):
        """
        Instantiate the dataset class with proper split state and parameters.

        :param state: Split identifier, typically 'train', 'val', or 'test'.
        :param override_kwargs: Additional arguments to override defaults.
        :return: Instantiated dataset object.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)

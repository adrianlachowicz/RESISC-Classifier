import os
from src.config import TRAIN_SIZE
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder


class RESISCDataset(Dataset):
    """
    The RESISC-45 dataset.

    Arguments:
        train (bool) - Create dataset for training or validation.
        transform (T.Compose) - Transforms to apply on images.
    """

    def __init__(self, train: bool, transform):
        self.dataset_path = os.path.join(
            "..", "..", "data", "external", "resisc", "NWPU-RESISC45"
        )
        self.full_dataset = ImageFolder(root=self.dataset_path, transform=transform)

        self.dataset = self.get_dataset(train=train)
        self.class_to_idx = self.full_dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def get_train_val_sizes(self, train_size: float):
        """
        The function calculates train and validation datasets sizes

        Arguments:
            train_size (float) - A size of training dataset (values between 0 and 1).

        Returns:
            train_size (int) - Num of the images in train dataset.
            val_size (int) - Num of the images in train dataset.
        """
        assert (train_size > 0) and (
            train_size < 1
        ), "Invalid dataset sizes! The size value should be between 0 and 1!"

        full_dataset_size = len(self.full_dataset)
        val_size = round(1 - train_size, 1)

        # Calculate count of samples in every dataset
        train_size = int(train_size * full_dataset_size)
        val_size = int(val_size * full_dataset_size)

        assert (
            train_size + val_size == full_dataset_size
        ), "Invalid calculations of dataset sizes!"

        return train_size, val_size

    def get_dataset(self, train: bool):
        """
        The function splits full dataset into smaller train and validation sets.

        Arguments:
            train (bool) - Return train or validation set.

        Returns:
            dataset (Subset) - The subset.
        """

        train_size, val_size = self.get_train_val_sizes(TRAIN_SIZE)
        train_set, val_set = random_split(self.full_dataset, [train_size, val_size])

        if train is True:
            return train_set
        else:
            return val_set

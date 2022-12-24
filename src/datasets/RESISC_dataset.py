import os
from src.config import TRAIN_SIZE
from torch.utils.data import Dataset
from torch.utils.data import random_split


class RESISCDataset(Dataset):
    """
    The RESISC-45 dataset.

    Arguments:
        train (bool) - Create dataset for training or validation.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

import torch
import torch.nn as nn
import torch.optim as optim
from src.models import ModelLoader
from src.config import (
    FEATURE_EXTRACTOR_CONFIG,
    CLASSIFIER_CONFIG,
    IMG_CONFIG,
    GENERAL_CONFIG,
    DEVICE,
)
from torchsummary import summary


class Model(nn.Module):
    """
    The model.

    Arguments:
        feature_extractor_config (dict) - The feature extractor config.
        classifier_config (dict) - The classifier config.
        img_config (dict) - The image config
        general_config (dict) - The general config.
    """

    def __init__(
        self, feature_extractor_config, classifier_config, img_config, general_config
    ):
        super().__init__()

        model_loader = ModelLoader(
            feature_extractor_config, classifier_config, img_config
        )
        self.model = model_loader.load_model()

        optimizer_name = general_config["optimizer"]
        learning_rate = general_config["learning_rate"]

        self.optimizer = getattr(optim, optimizer_name)(
            self.parameters(), lr=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

import timm
import torch.nn as nn
from torchsummary import summary


class PretrainedFeatureExtractor(nn.Module):
    """
    The pretrained feature extractor module.
    The feature extractors are get from the 'timm' module.

    Arguments:
        model_name (str) - Name of the model.
        pretrained (bool) - Initialize model with pretrained weights or not.
    """

    def __init__(self, model_name: str, pretrained: bool):
        super().__init__()

        self.feature_extractor = timm.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=0
        )

    def forward(self, x):
        return self.feature_extractor(x)

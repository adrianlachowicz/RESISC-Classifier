import torch
from src.models.classifiers import LinearClassifier
from src.models.feature_extractors import PretrainedFeatureExtractor


class ModelLoader:
    """
    The model loader class, for loading model with specific parameters.

    Arguments:
        feature_extractor_config (dict) - The feature extractor config.
        classifier_config (dict) - The classifier config.
        img_config (dict) - The image config
        general_config (dict) - The general config.
    """

    def __init__(
        self, feature_extractor_config, classifier_config, img_config, general_config
    ):
        self.feature_extractor_config = feature_extractor_config
        self.classifier_config = classifier_config
        self.img_config = img_config
        self.general_config = general_config

        self.feature_extractor = self.load_feature_extractor()
        self.classifier = self.load_classifier()

    def load_feature_extractor(self):
        """
        The function loads feature extractor based on user config.

        Returns:
            feature_extractor (nn.Module) - The feature extractor.
        """

        if self.feature_extractor_config["type"] == "pretrained":
            model_name = self.feature_extractor_config["model_name"]
            pretrained = self.feature_extractor_config["pretrained"]

            feature_extractor = PretrainedFeatureExtractor(model_name, pretrained)
            return feature_extractor

    def load_classifier(self):
        """
        The function loads classifier based on user config.

        Returns:
            classifier (nn.Module) - The classifier.
        """

        if self.classifier_config["type"] == "linear":
            input_dim = self.classifier_config["input_dim"]
            hidden_sizes = self.classifier_config["hidden_dims"]
            output_dim = self.classifier_config["output_dim"]
            activation_function_name = self.classifier_config[
                "activation_function_name"
            ]

            # Calculate input dim if user pass None as input_dim
            if input_dim is None:
                channels = self.img_config["channels"]
                width = self.img_config["width"]
                height = self.img_config["height"]

                img = torch.rand(1, channels, height, width)
                feature_extractor = self.feature_extractor
                output = feature_extractor(img)
                input_dim = output.shape[0] * output.shape[1]

            classifier = LinearClassifier(
                input_dim, hidden_sizes, output_dim, activation_function_name
            )
            return classifier

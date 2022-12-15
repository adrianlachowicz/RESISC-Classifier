import torch.cuda

NUM_WORKERS = 1
BATCH_SIZE = 16
TRAIN_SIZE = 0.8
EPOCHS = 10
API_TOKEN = "your_api_key"
TRAINING_NAME = "test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_EXTRACTOR_CONFIG = {
    "type": "pretrained",
    "model_name": "mobilenetv2_100",
    "pretrained": True,
    "freeze": False
}

CLASSIFIER_CONFIG = {
    "type": "linear",
    "input_dim": None,
    "hidden_dims": [],
    "output_dim": 45,
    "activation_function_name": "ReLU",
}

IMG_CONFIG = {"channels": 3, "width": 256, "height": 256}

GENERAL_CONFIG = {"optimizer": "Adam", "learning_rate": 0.001}

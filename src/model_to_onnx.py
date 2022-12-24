import torch
from src.models import Model
from config import (
    FEATURE_EXTRACTOR_CONFIG,
    CLASSIFIER_CONFIG,
    IMG_CONFIG,
    GENERAL_CONFIG,
    DEVICE,
    MODEL_CHECKPOINT_PATH,
)

assert MODEL_CHECKPOINT_PATH is not None, "Model checkpoint path must be defined!"

img_channels = IMG_CONFIG["channels"]
img_height = IMG_CONFIG["height"]
img_width = IMG_CONFIG["width"]

img = torch.rand(1, img_channels, img_height, img_width).to(DEVICE)

model = Model(
    FEATURE_EXTRACTOR_CONFIG, CLASSIFIER_CONFIG, IMG_CONFIG, GENERAL_CONFIG
).to(DEVICE)

print(model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH)))
model.eval()

outputs = model(img)

torch.onnx.export(
    model,
    img,
    f="best_model.onnx",
    export_params=True,
    input_names=["input"],
    output_names=["output"],
)

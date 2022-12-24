import onnxruntime
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.special import softmax
from argparse import ArgumentParser
import json


def parse_args():
    parser = ArgumentParser(description="The script predicts label of passed image.")

    parser.add_argument("--img_path", help="A path to a image.", type=str)
    parser.add_argument("--model_path", help="A path to .onnx model.", type=str, default="./model.onnx")
    parser.add_argument("--labels_path", help="A path to the labels.json file.", type=str, default="labels.json")

    arguments = parser.parse_args()
    return arguments


def load_labels(labels_path: str):
    """
    The function loads labels from file and converts to python dict type.

    Arguments:
        labels_path (str) - A path to JSON labels file.

    Returns:
        data (dict) - A dictionary contains a dictionary, where keys are labels and values numbers.
    """

    file = open(labels_path, "r")
    data = file.read()
    data = json.loads(data)
    return data


def to_numpy(tensor):
    """
    The function converts passed tensor to numpy array.

    Arguments:
        tensor (torch.Tensor) - A image as torch.Tensor.

    Returns:
        output (np.array) - The image as numpy array.
    """
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def predict(image_path: str, onnx_model_path: str):
    """
    The function predicts image label using .onnx model.

    Arguments:
        image_path (str) - Path to an image.
        onnx_model_path (str) - Path to a .onnx model file.

    Returns:
        outputs (torch.tensor) - A tensor, which contains probabilities of every label.
    """

    transform = T.Compose([T.ToTensor()])

    img = Image.open(image_path) # ""
    img = transform(img).unsqueeze(0)

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    outputs = np.array(softmax(ort_outs))
    outputs = torch.from_numpy(outputs)

    return outputs


if __name__ == "__main__":
    args = parse_args()

    img_path = args.img_path
    model_path = args.model_path
    labels_path = args.labels_path

    labels = load_labels(labels_path)

    predictions = predict(img_path, model_path)
    pred_values, pred_indices = torch.topk(predictions, k=5)

    pred_values = pred_values.cpu().detach().numpy()[0] * 100
    pred_indices = pred_indices.cpu().detach().numpy()[0]

    # Convert predicted indices to labels
    predicted_labels = []

    for pred in pred_indices:
        for k, v in labels.items():
            if v == pred:
                predicted_labels.append(k)

    # Print predictions
    print("Predictions:")

    for i in range(5):
        print("\t {}. {}: {}%".format(i+1, predicted_labels[i], round(pred_values[i], 1)))

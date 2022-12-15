import os
import neptune.new as neptune
import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torchvision.transforms as T
from torch.utils.data import DataLoader
from src.models import Model
from src.datasets import RESISCDataset
from src.config import *
from src.utils import get_train_val_sizes
from tqdm import tqdm
from torchsummary import summary

def split_datasets(transform):
    """
    The function splits dataset into train and validation sets.

    Arguments:
        transform (T.Compose) - Transforms to apply on images.

    Returns:
        train_set (Subset) - The train set.
        val_set (Subset) - The validation set.
    """
    dataset_path = os.path.join("..", "data", "external", "resisc", "NWPU-RESISC45")

    full_dataset = ImageFolder(root=dataset_path, transform=transform)
    train_size, val_size = get_train_val_sizes(TRAIN_SIZE, full_dataset)

    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    return train_set, val_set


def define_dataloaders():
    """
    The function creates and returns train and validation dataloader.

    Returns:
        train_dataloader (DataLoader) - The train dataloader.
        val_dataloader (DataLoader) - The validation dataloader.
    """

    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    train_subset, val_subset = split_datasets(transform)

    train_dataset = RESISCDataset(train_subset)
    val_dataset = RESISCDataset(val_subset)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    return train_dataloader, val_dataloader


def load_model():
    """
    The function loads and returns model.

    Returns:
        model (nn.Module) - The model.
    """

    model = Model(
        FEATURE_EXTRACTOR_CONFIG, CLASSIFIER_CONFIG, IMG_CONFIG, GENERAL_CONFIG
    ).to(DEVICE)
    return model


def validate_model(model, val_dataloader):
    """
    The function validates model using passed dataloader.

    Parameters:
        model (nn.Module) - The model.
        val_dataloader (DataLoader) - The dataloader, which will be used for validating model.

    Returns:
         accuracy (float) - Accuracy of a model.
         balanced_accuracy (float) - Balanced accuracy of a model.
         avg_loss (float) - Average validation loss.
    """
    model.eval()
    running_loss = 0.0

    all_outputs = []
    all_targets = []

    p_bar = tqdm(total=len(val_dataloader), position=0, leave=True)

    for i, data in enumerate(val_dataloader):
        inputs, targets = data[0].to(DEVICE), data[1].to(DEVICE)
        outputs = model(inputs)

        loss = model.criterion(outputs, targets)
        running_loss += loss.item()

        outputs = (
            torch.softmax(outputs, dim=1).cpu().detach().numpy().argmax(1).tolist()
        )
        targets = targets.cpu().detach().numpy().tolist()

        all_outputs.extend(outputs)
        all_targets.extend(targets)

        p_bar.update(1)

    accuracy = accuracy_score(all_targets, all_outputs)
    balanced_accuracy = balanced_accuracy_score(all_targets, all_outputs)
    avg_loss = running_loss / len(val_dataloader)

    del p_bar

    return accuracy, balanced_accuracy, avg_loss


if __name__ == "__main__":
    train_loader, val_loader = define_dataloaders()
    model = load_model()

    run = neptune.init_run(
        project="adrianlachowicz/RESISC-Classifier",
        api_token=API_TOKEN,
    )

    run["feature_extractor_config"] = FEATURE_EXTRACTOR_CONFIG
    run["classifier_config"] = CLASSIFIER_CONFIG
    run["img_config"] = IMG_CONFIG
    run["general_config"] = GENERAL_CONFIG
    run["rest_config"] = {
        "epochs": EPOCHS,
        "num_workers": NUM_WORKERS,
        "batch_size": BATCH_SIZE,
        "train_size": TRAIN_SIZE,
        "device": DEVICE,
        "training_name": TRAINING_NAME,
    }

    # Train model
    for epoch in range(1, EPOCHS + 1):
        print("Epoch {}: starting......\n".format(epoch))
        model.train()
        running_loss = 0.0

        print("Training start:")
        p_bar = tqdm(total=len(train_loader), position=0, leave=True)

        for i, data in enumerate(train_loader):
            inputs, targets = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(inputs)

            loss = model.criterion(outputs, targets)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            loss = loss.item()

            running_loss += loss
            run["train/step_loss"].log(loss)

            p_bar.update(1)

        average_epoch_loss = running_loss / len(train_loader)
        run["train/epoch_loss"].log(average_epoch_loss)

        del p_bar

        # Validate model
        print("\nValidation start:")
        val_accuracy, val_balanced_accuracy, val_loss = validate_model(
            model, val_loader
        )

        run["validation/accuracy"].log(val_accuracy)
        run["validation/balanced_accuracy"].log(val_balanced_accuracy)
        run["validation/loss"].log(val_loss)

        # Save model
        os.makedirs(os.path.join("..", "models", TRAINING_NAME), exist_ok=True)
        model_path = os.path.join("..", "models", TRAINING_NAME, "model-{}.pth").format(
            epoch
        )

        torch.save(model.state_dict(), model_path)
        print("Model checkpoint saved as: {}".format(model_path))

        print("Epoch {}: end......\n".format(epoch))

    run.stop()

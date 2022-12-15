def get_train_val_sizes(train_size: float, full_dataset):
    """
    The function calculates train and validation datasets sizes

    Arguments:
        train_size (float) - A size of training dataset (values between 0 and 1).
        full_dataset (ImageFolder) - The dataset.

    Returns:
        train_size (int) - Num of the images in train dataset.
        val_size (int) - Num of the images in train dataset.
    """
    assert (train_size > 0) and (
            train_size < 1
    ), "Invalid dataset sizes! The size value should be between 0 and 1!"

    full_dataset_size = len(full_dataset)
    val_size = round(1 - train_size, 1)

    # Calculate count of samples in every dataset
    train_size = int(train_size * full_dataset_size)
    val_size = int(val_size * full_dataset_size)

    assert (
            train_size + val_size == full_dataset_size
    ), "Invalid calculations of dataset sizes!"

    return train_size, val_size

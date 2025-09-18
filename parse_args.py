import argparse

def _valid_epochs(epochs):
    try:
        epochs = int(epochs)
    except ValueError:
        raise ValueError("Epochs value provided is not interpretable as an integer.")
    if epochs < 0:
        raise ValueError("Epochs value cannot be negative.")
    return epochs

def _valid_lr(lr):
    try:
        lr = float(lr)
    except ValueError:
        raise ValueError("The learning rates value provided is not interpretable as a float")
    if lr < 0:
        raise ValueError("The learning rates value cannot be negative")
    return lr

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs the model will be trained for.",
        action=_valid_epochs,
    )

    parser.add_argument(
        "-l",
        "--lr",
        help="Learning rate used by the optimizer",
        action=_valid_lr,
    )
    
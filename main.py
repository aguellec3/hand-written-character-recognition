import sys
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataset import data_loader, CLASS_NAMES
from classifier import Classifier
from train import training
from evaluate import evaluate
from visualize import visualize
from parse_args import parse_args


def main(args):
    params = parse_args(args)
    folder_root = Path.cwd() / "data"

    # Get and split the data
    train_loader, test_loader, val_loader = data_loader(folder_root, CLASS_NAMES)

    # Initialize the model
    model = Classifier()

    # Train the Model
    epochs = params.epochs
    lr = params.lr
    criterion = nn.CrossEntropyLoss()

    train_acc, val_acc, model = training(
        model,
        epochs,
        optim.Adam(model.parameters(), lr),
        criterion,
        train_loader,
        val_loader,
    )

    test_acc = evaluate(model, test_loader)
    print(f"test acc: {test_acc}")

    visualize(
        train_acc,
        val_acc,
        epochs,
        "Train vs Validation Accuracy",
        "Epochs",
        "Accuracy(%)",
    )


if __name__ == "__main__":
    main(sys.argv[1:])

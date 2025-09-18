import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataset import data_loader
from classifier import Classifier
from train import training
from evaluate import evaluate
from visualize import visualize

# Dataset reference
# https://www.nist.gov/srd/nist-special-database-19



if __name__ == '__main__':
    folder_root = Path.cwd() / "data"

    ''' This is what each class name represents
    0-9: 30-39
    A-I: 41-49
    J-O: 4a-4f
    P-Y: 50-59
    Z: 5a
    a-i: 61-69
    j-o: 6a-6f
    p-y: 70-79
    z: 7a
    '''

    class_names = np.array(['30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '41', '42', '43', '44', '45', '46',
               '47', '48', '49', '4a', '4b', '4c', '4d', '4e', '4f', '50', '51', '52', '53', '54', '55',
               '56', '57', '58', '59', '5a', '61', '62', '63', '64', '65', '66', '67', '68', '69', '6a',
               '6b', '6c', '6d', '6e', '6f', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',
               '7a'])

    # Get and split the data
    train_loader, test_loader, val_loader = data_loader(folder_root, class_names)

    # Initialize the model
    model = Classifier()

    # Train the Model
    epochs = 13 # This is low since I'm working on CPU
    lr = 0.00001
    criterion = nn.CrossEntropyLoss()

    train_acc, val_acc, model = training(model, epochs, optim.Adam(model.parameters(), lr), criterion, train_loader, val_loader)

    test_acc = evaluate(model, test_loader)
    print(f'test acc: {test_acc}')

    visualize(train_acc, val_acc, epochs, "Train vs Validation Accuracy", "Epochs", "Accuracy(%)")
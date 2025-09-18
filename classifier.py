import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # Define Neural Network Architecture
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, 2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2304, 64)
        self.fc2 = nn.Linear(64, 62)
        self.init_weights()

    def init_weights(self):

        torch.manual_seed(442)

        for layer in [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.fc1,
            self.fc2,
        ]:
            l_in = layer.weight.size(1)
            nn.init.normal_(layer.weight, 0.0, 1 / np.sqrt(5 * 5 * l_in))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, input):
        l1 = self.conv1(F.relu(input))
        l2 = self.conv2(F.relu(l1))
        p2 = self.maxpool(l2)
        l3 = self.conv3(F.relu(p2))
        p3 = self.maxpool(l3)
        l4 = self.conv4(F.relu(p3))
        p4 = self.maxpool(l4)
        flat = torch.flatten(p4)
        f1 = self.fc1(flat)
        f2 = self.fc2(f1)

        return f2

import numpy as np
from PIL import Image
import os
import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Dataset reference
# https://www.nist.gov/srd/nist-special-database-19

""" This is what each class name represents
0-9: 30-39
A-I: 41-49
J-O: 4a-4f
P-Y: 50-59
Z: 5a
a-i: 61-69
j-o: 6a-6f
p-y: 70-79
z: 7a
"""

CLASS_NAMES = np.array(
    [
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "4a",
        "4b",
        "4c",
        "4d",
        "4e",
        "4f",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "5a",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "6a",
        "6b",
        "6c",
        "6d",
        "6e",
        "6f",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "7a",
    ]
)


class DatasetImages(Dataset):
    def __init__(self, root_dir, class_names, transform):
        self.class_names = class_names
        self.transform = transform
        self.files = glob.glob(os.path.join(root_dir, "*.png"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = np.asarray(img)
        filename = os.path.basename(self.files[idx])
        label = int(np.where(self.class_names == filename[6:8])[0][0])
        img = self.transform(img)
        return img, label


def data_loader(folder_root, class_names):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_data = DatasetImages(folder_root / "train", class_names, transform)
    test_data = DatasetImages(folder_root / "test", class_names, transform)
    val_data = DatasetImages(folder_root / "val", class_names, transform)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    return train_loader, test_loader, val_loader

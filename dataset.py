import numpy as np
from PIL import Image
import os
import glob
from torch.utils.data import DataLoader, Dataset 
from torchvision import transforms

class DatasetImages(Dataset):
    def __init__(self, root_dir, class_names, transform):
        self.class_names = class_names
        self.transform = transform
        self.files = glob.glob(os.path.join(root_dir, '*.png'))
        print(self.__len__())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = np.asarray(img)
        filename = os.path.basename(self.files[idx])
        label = int(np.where(self.class_names == filename[6:8])[0][0])
        if self.transform:
            img = self.transform(img)
        return img, label



def data_loader(folder_root, class_names):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = DatasetImages(folder_root / "train", class_names, transform)
    test_data = DatasetImages(folder_root / "test", class_names, transform)
    val_data = DatasetImages(folder_root / "val", class_names, transform)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    return train_loader, test_loader, val_loader
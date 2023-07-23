from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import Literal

import os
from PIL import Image



class BrainTumorDataset(Dataset):
    def __init__(self, root, part: Literal["train", "test", "valid"], transforms=None):
        self.root = root
        self.data_dir = f"{root}/{part}"
        self.transforms = transforms
           
        self.images = [
            f"{self.data_dir}/{img}" for img in os.listdir(self.data_dir)
                       if "_mask" not in img and not img.endswith(".csv")
            ]
        self.targets = [
            f"{self.data_dir}/{img.replace('.jpg','')}_mask.png" for img in os.listdir(self.data_dir)
                       if "_mask" not in img and not img.endswith(".csv")
            ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.targets[index]
        print(img)
        print(label)
        img = Image.open(img)
        label = Image.open(label)
        if self.transforms:
            img = self.transforms(img)
            label = self.transforms(label)
            
        return img, label


if __name__ == "__main__":
    dataset = BrainTumorDataset(root="dataset", part="train", transforms=ToTensor())
    
    img, label = dataset[0]
    print(img.size(), label.size())
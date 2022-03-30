import os
import glob
import cv2
import random
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return A.Compose([
        A.Resize(height=14*32, width=18*32),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_test_transforms():
    return A.Compose([
        A.Resize(height=14*32, width=18*32),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def train_test_split(imgs_root: str, split: float = 0.9):
    "Create lists with train/test image paths"
    img_paths = glob.glob(imgs_root + "/*.png")
    random.shuffle(img_paths)
    train_split = int(len(img_paths) * split)
    train = img_paths[:train_split]
    test = img_paths[train_split:]

    return train, test


class Comma10kDataset(Dataset):
    def __init__(self, imgs_root: str, masks_root: str, imgs: list, classes: int, transforms):
        super(Comma10kDataset, self).__init__()
        self.imgs_root = imgs_root
        self.masks_root = masks_root
        self.imgs = imgs
        self.classes = classes
        self.transforms = transforms


    def __getitem__(self, idx):
        img = self.imgs[idx]
        _, filename = os.path.split(img)
        mask = os.path.join(self.masks_root, filename)
        assert os.path.isfile(mask)

        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask, 0).astype("uint8")

        if self.transforms:
            transformed = self.transforms(
                image=image, mask=mask)

            transformed_image = transformed['image']
            transformed_mask = transformed['mask']


        mask = torch.stack([(transformed_mask == v) for v in self.classes], axis=0).float()

        return transformed_image, mask

    def __len__(self):
        return len(self.imgs)




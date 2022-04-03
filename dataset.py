import os
import glob
import cv2
import random
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RGBShift(
            r_shift_limit=20, 
            g_shift_limit=20, 
            b_shift_limit=20, 
            p=0.2,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
            brightness_limit=0.4, 
            contrast_limit=0.4, 
            p=1.0,
            ),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.5),
        
        A.Blur(blur_limit=7, p=0.5),
        A.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.2, 
            p=0.5,
        ),
        A.OneOf([
            A.RandomFog(p=1.0),
            A.RandomRain(p=1.0),
            A.RandomSnow(p=1.0),
            A.RandomSunFlare(p=1.0),
        ], p=0.4),
        A.Resize(
            height=cfg.DATASET.IMG_HEIGHT, 
            width=cfg.DATASET.IMG_WIDTH,
            p=1.0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_test_transforms(cfg):
    return A.Compose([
        A.Resize(
            height=cfg.DATASET.IMG_HEIGHT, 
            width=cfg.DATASET.IMG_WIDTH,
        ),
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




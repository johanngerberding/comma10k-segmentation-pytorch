import os
import glob
import cv2
import random
import numpy as np 
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
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
    

def transforms_img():
    return A.Compose([
        A.OneOf([
            A.RandomRain(p=1.0),
            A.RandomSnow(p=1.0),
            A.RandomSunFlare(p=1.0),
            A.RandomShadow(p=1.0),
            A.RandomFog(p=1.0),
        ], p=0.4),
        A.OneOf([
            A.GaussianBlur(p=1.0),
            A.MotionBlur(p=1.0)
        ], p=0.4),
        A.OneOf([
            A.CLAHE(p=1.0),
            A.ColorJitter(p=1.0)
        ], p=0.4),
        A.OneOf([
            A.RandomContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
        ],p=0.5),
    ])


def train_test_split(imgs_root: str):
    "Create lists with train/test image paths"
    img_paths = glob.glob(imgs_root + "/*.png")
    train = [img for img in img_paths if not img.endswith('9.png')]
    test = [img for img in img_paths if img.endswith('9.png')]
    return train, test


class Comma10kDataset(Dataset):
    def __init__(
        self, 
        imgs_root: str, 
        masks_root: str, 
        imgs: list, 
        cfg,
        transforms,
        train=True,
    ):
        super(Comma10kDataset, self).__init__()
        self.imgs_root = imgs_root
        self.masks_root = masks_root
        self.imgs = imgs
        self.classes = cfg.DATASET.CLASSES
        self.transforms = transforms
        self.train = train


    def __getitem__(self, idx):
        img = self.imgs[idx]
        _, filename = os.path.split(img)
        mask = os.path.join(self.masks_root, filename)
        assert os.path.isfile(mask)

        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask, 0).astype("uint8")
        masks = [(mask == v).astype(int) for v in self.classes]
        
        if self.transforms:
            if self.train:
                image = transforms_img()(image=image)['image']
                
            transformed = self.transforms(
                image=image, masks=masks)

            transformed_image = transformed['image']
            transformed_masks = transformed['masks']

        mask = np.stack(transformed_masks, axis=0)
        mask = torch.tensor(mask).float()

        return transformed_image, mask


    def __len__(self):
        return len(self.imgs)




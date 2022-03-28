import os
import glob
import cv2
import torch
from torch.utils.data import Dataset


class Comma10kDataset(Dataset):
    def __init__(self, imgs_root: str, masks_root: str, classes: int, transforms):
        super(Comma10kDataset, self).__init__()
        self.imgs_root = imgs_root
        self.masks_root = masks_root
        self.classes = classes
        self.imgs = glob.glob(self.imgs_root + "/*.png")
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




import os 
import glob
import cv2
import numpy as np 
import torch 
from torch.utils.data import Dataset 


class Comma10kDataset(Dataset):
    def __init__(self, imgs_root, masks_root, transforms):
        super(Comma10kDataset, self).__init__()
        self.imgs_root = imgs_root
        self.masks_root = masks_root
        self.imgs = glob.glob(self.imgs_root + "/*.png")
        self.transforms = transforms 
        
        
    def __getitem__(self, idx):
        img = self.imgs[idx]
        _, filename = os.path.split(img)
        mask = os.path.join(self.masks_root, filename)
        assert os.path.isfile(mask)
        
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask)
        
        if self.transforms:
            transformed = self.transforms(
                image=image, mask=mask)
        
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        
        transformed_mask = torch.transpose(
            transformed_mask, 0, 2)
        
        return transformed_image, transformed_mask
    
    def __len__(self):
        return len(self.imgs)


    
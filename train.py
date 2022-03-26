import os   
import albumentations as A 
from albumentations.pytorch import ToTensorV2

from dataset import Comma10kDataset
# height 874
# width 1164


def main():
    comma10k_dir = "/home/johann/sonstiges/comma10k-segmenation-pytorch/comma10k"
    imgs_root = os.path.join(comma10k_dir, "imgs")
    masks_root = os.path.join(comma10k_dir, "masks")
    
    transforms = A.Compose([
        A.Resize(256,256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    dataset = Comma10kDataset(
        imgs_root, masks_root, transforms,
    )
    
    
    for img, mask in dataset:
        print(img.size())
        print(mask.size())
        break
    
    
if __name__ == "__main__":
    main()
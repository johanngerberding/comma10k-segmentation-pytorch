import os 
import cv2
import torch 
from model import RegSeg

import albumentations as A 
from albumentations.pytorch import ToTensorV2


def main():
    img_path = ""
    out_path = ""
    checkpoint = ""
    classes = [41, 76, 90, 124, 161]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegSeg(num_classes=len(classes))
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(height=14*32, width=18*32), #height, width
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    transformed = transform(image=img)
    img = transformed['image']

    img = img.unsqueeze(0)
    
    with torch.no_grad():
        pred = model(img)
        
    


if __name__ == "__main__":
    main()
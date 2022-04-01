import os 
import cv2
import torch 
from model import RegSeg
import matplotlib.pyplot as plt 

import albumentations as A 
from albumentations.pytorch import ToTensorV2


def main():
    img_path = "/home/johann/sonstiges/comma10k-segmenation-pytorch/comma10k/imgs/0001_a23b0de0bc12dcba_2018-06-24--00-29-19_17_79.png"
    out_path = "pred-example.png"
    checkpoint = "/home/johann/sonstiges/comma10k-segmenation-pytorch/exps/2022-03-30/final.pth"
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

    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img)
        
    print(pred.size())
    pred = pred.detach().cpu().squeeze().numpy()
    
    plt.figure(figsize=(16,9))
    plt.imshow(pred[0])
    plt.savefig("test-0.jpg")
    
    plt.figure(figsize=(16,9))
    plt.imshow(pred[1])
    plt.savefig("test-1.jpg")
    
    plt.figure(figsize=(16,9))
    plt.imshow(pred[2])
    plt.savefig("test-2.jpg")
    
    plt.figure(figsize=(16,9))
    plt.imshow(pred[3])
    plt.savefig("test-3.jpg")
        
    


if __name__ == "__main__":
    main()
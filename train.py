import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import Comma10kDataset

# original img size
# height 874
# width 1164
from model import RegSeg



def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    for i, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device).long()

        optimizer.zero_grad()

        pred = model(img)

        mask = torch.argmax(mask, axis=1)
        loss = loss_fn(pred, mask)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Iteration {i} - Train Loss: {train_loss / i}")


    train_loss /= i

    return train_loss


def val_epoch(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0

    for i, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device).long()

        with torch.no_grad():
            pred = model(img)

        loss = loss_fn(pred, torch.argmax(mask, axis=1))

        val_loss += loss.item()

    val_loss /= i

    return val_loss



def main():
    comma10k_dir = "/home/johann/sonstiges/comma10k-segmenation-pytorch/comma10k"
    imgs_root = os.path.join(comma10k_dir, "imgs")
    masks_root = os.path.join(comma10k_dir, "masks")
    classes = [41, 76, 90, 124, 161]
    num_epochs = 5
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RegSeg(num_classes=len(classes))
    model.train()
    model.to(device)

    train_transforms = A.Compose([
        A.Resize(height=14*32, width=18*32), #height, width
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    test_transforms = A.Compose([
        A.Resize(height=14*32, width=18*32),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

    dataset = Comma10kDataset(
        imgs_root,
        masks_root,
        classes=[41, 76, 90, 124, 161],
        transforms=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, dataloader, optimizer, loss_fn, device)

        # val_loss = val_epoch(model, dataloader, loss_fn, device)





if __name__ == "__main__":
    main()

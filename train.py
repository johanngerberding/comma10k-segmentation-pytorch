import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import Comma10kDataset
# height 874
# width 1164
from model import RegSeg



def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    train_loss = 0
    for i, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        pred = model(img)

        loss = loss_fn(mask, torch.argmax(pred, axis=1))
        loss.backward()

        optimizer.step()

        train_loss += loss.item()


    train_loss /= i

    return train_loss


@torch.no_grad()
def val_epoch(model, dataloader, loss_fn):
    model.eval()
    val_loss = 0

    for i, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)

        loss = loss_fn(mask, torch.argmax(pred, axis=1))

    return val_loss



def main():
    comma10k_dir = "/home/johann/sonstiges/comma10k-segmenation-pytorch/comma10k"
    imgs_root = os.path.join(comma10k_dir, "imgs")
    masks_root = os.path.join(comma10k_dir, "masks")
    classes = [41, 76, 90, 124, 161]
    num_epochs = 20


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

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, dataloader, optimizer, loss_fn)
        val_loss = val_epoch(model, dataloader, loss_fn)

    for img, mask in dataset:
        print(img.size())
        print(mask.size())

        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        out_mask = model(img)

        print(out_mask.size())
        loss = loss_fn(mask, torch.argmax(out_mask, axis=1))
        print(loss)
        break


if __name__ == "__main__":
    main()

import os
import torch
import shutil
from datetime import date
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import (Comma10kDataset,
                     train_test_split,
                     get_train_transforms,
                     get_test_transforms)

# original img size
# height 874
# width 1164
from model import RegSeg



def train_epoch(model, dataloader, optimizer, loss_fn, device, writer, epoch):
    model.train()
    train_loss = 0
    size = len(dataloader.dataset)
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

        iteration = (epoch * size + (i + 1))
        if iteration % 100 == 0 and i > 0:
            print(f"Iteration {iteration} - Train Loss: {train_loss / i}")
            writer.add_scalar('Loss/train', train_loss/i, iteration)


    train_loss /= i

    return train_loss


def val_epoch(model, dataloader, loss_fn, device, writer, epoch):
    model.eval()
    val_loss = 0
    size = len(dataloader.dataset)
    for i, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device).long()

        with torch.no_grad():
            pred = model(img)

        loss = loss_fn(pred, torch.argmax(mask, axis=1))

        val_loss += loss.item()
        iteration = (epoch * size + (i + 1))
        if iteration % 20 == 0 and i > 0:
            print(f"Iteration {iteration} - Val Loss: {val_loss / i}")
            writer.add_scalar('Loss/val', val_loss/i, iteration)


    val_loss /= i

    return val_loss



def main():
    comma10k_dir = "/home/johann/sonstiges/comma10k-segmenation-pytorch/comma10k"
    imgs_root = os.path.join(comma10k_dir, "imgs")
    masks_root = os.path.join(comma10k_dir, "masks")
    classes = [41, 76, 90, 124, 161]
    num_epochs = 5
    train_batch_size = 32
    val_batch_size = 16

    outdir = f"exps/{date.today().strftime('%Y-%m-%d')}"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RegSeg(num_classes=len(classes))
    model.train()
    model.to(device)

    train_imgs, val_imgs = train_test_split(imgs_root)

    train_dataset = Comma10kDataset(
        imgs_root,
        masks_root,
        train_imgs,
        classes=classes,
        transforms=get_train_transforms(),
    )

    val_dataset = Comma10kDataset(
        imgs_root,
        masks_root,
        val_imgs,
        classes=classes,
        transforms=get_test_transforms(),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    writer = SummaryWriter(outdir)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device, writer, epoch)
        val_loss = val_epoch(model, val_dataloader, loss_fn, device, writer, epoch)
        torch.save(model.state_dict(), os.path.join(outdir, f"{str(epoch + 1).zfill(3)}-epoch.pth"))

    torch.save(model.state_dict(), os.path.join(outdir, "final.pth"))



if __name__ == "__main__":
    main()

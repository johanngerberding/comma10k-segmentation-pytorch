import os
import math
import json
import torch
import torch.nn as nn
import shutil
from datetime import date
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import (Comma10kDataset,
                     train_test_split,
                     get_train_transforms,
                     get_test_transforms)

from model import RegSeg
from config import get_cfg_defaults
from evaluate import IoU


def train_epoch(model, dataloader, optimizer, loss_fn, device, writer, epoch, cfg):
    model.train()
    train_loss = 0
    ious = []

    size = len(dataloader.dataset)

    if epoch > 0:
        iteration = int(math.ceil((epoch * size) / cfg.TRAIN.BATCH_SIZE))
    else:
        iteration = 0

    for i, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device).long()

        optimizer.zero_grad()

        pred = model(img)

        loss = loss_fn(pred, torch.argmax(mask, axis=1))
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        pred = pred.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        iou = IoU(pred, mask)
        ious.append(iou)

        iteration += 1
        if iteration % 20 == 0 and i > 0:
            print(f"Iteration {iteration} - Train Loss: {train_loss / i} - Mean IoU: {sum(ious) / i}")
            writer.add_scalar('Loss/train', train_loss/i, iteration)


    train_loss /= i

    miou = sum(ious) / i

    return train_loss, miou


def val_epoch(model, dataloader, loss_fn, device, writer, epoch, cfg):
    model.eval()
    val_loss = 0
    ious = []

    size = len(dataloader.dataset)

    if epoch > 0:
        iteration = int(math.ceil((epoch * size) / cfg.VAL.BATCH_SIZE))
    else:
        iteration = 0

    for i, (img, mask) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device).long()

        with torch.no_grad():
            pred = model(img)

        loss = loss_fn(pred, torch.argmax(mask, axis=1))

        pred = pred.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        iou = IoU(pred, mask)
        ious.append(iou)

        val_loss += loss.item()
        iteration += 1
        if iteration % 20 == 0 and i > 0:
            print(f"Iteration {iteration} - Val Loss: {val_loss / i} - Mean IoU: {sum(ious) / i}")
            writer.add_scalar('Loss/val', val_loss/i, iteration)


    val_loss /= i

    miou = sum(ious) / i

    return val_loss, miou



def main():
    cfg = get_cfg_defaults()
    cfg.freeze()

    imgs_root = os.path.join(cfg.DATASET.ROOT, "imgs")
    masks_root = os.path.join(cfg.DATASET.ROOT, "masks")

    outdir = f"exps/{date.today().strftime('%Y-%m-%d')}"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=False)

    weights_dir = os.path.join(outdir, "weights")
    os.makedirs(weights_dir, exist_ok=False)

    with open(os.path.join(outdir, "config.yaml"), 'w') as fp:
        fp.write(cfg.dump())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RegSeg(num_classes=len(cfg.DATASET.CLASSES))
    if cfg.SYSTEM.NUM_GPUS > 1:
        model = nn.DataParallel(model)
    model.train()
    model.to(device)

    train_imgs, val_imgs = train_test_split(
        imgs_root, cfg.DATASET.SPLIT)

    with open(os.path.join(outdir, "train.txt"), 'w') as fp:
        for im in train_imgs:
            fp.write(im + "\n")

    with open(os.path.join(outdir, "val.txt"), 'w') as fp:
        for im in val_imgs:
            fp.write(im + "\n")

    train_dataset = Comma10kDataset(
        imgs_root,
        masks_root,
        train_imgs,
        classes=cfg.DATASET.CLASSES,
        transforms=get_train_transforms(cfg),
    )

    val_dataset = Comma10kDataset(
        imgs_root,
        masks_root,
        val_imgs,
        classes=cfg.DATASET.CLASSES,
        transforms=get_test_transforms(cfg),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.TRAIN.BASE_LR,
        momentum=cfg.TRAIN.MOMENTUM,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.1,
        patience=5,
        threshold=0.0001,
        min_lr=0.0000001,
    )

    writer = SummaryWriter(outdir)

    train_stats = {
        "train_loss": [],
        "train_miou": [],
        "val_loss": [],
        "val_miou": [],
    }

    best_val_loss = 10000
    best_epoch = 0

    print(f"Start training for {cfg.TRAIN.NUM_EPOCHS} epochs!")
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            device,
            writer,
            epoch,
            cfg,
        )
        val_loss, miou = val_epoch(
            model,
            val_dataloader,
            loss_fn,
            device,
            writer,
            epoch,
            cfg,
        )

        scheduler.step(val_loss)

        train_stats["train_loss"].append(train_loss)
        train_stats["val_loss"].append(val_loss)
        train_stats["val_miou"].append(miou)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pth"))
            best_val_loss = val_loss
            best_epoch = epoch

    torch.save(model.state_dict(), os.path.join(weights_dir, "final.pth"))

    print(f"Best model from epoch {best_epoch}.")

    with open(os.path.join(outdir, "train_stats.json"), 'w') as fp:
        json.dump(train_stats, fp, indent=4)


if __name__ == "__main__":
    main()

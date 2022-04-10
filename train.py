import os
import math
import glob
import random
import json
import torch
import torch.nn as nn
import shutil
from PIL import ImageColor
from datetime import date
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

from dataset import (Comma10kDataset,
                     train_test_split,
                     get_train_transforms,
                     get_test_transforms)

from models.regseg import RegSeg
from config import get_cfg_defaults
from utils import plot_pred2tgt, plot_stats



def plot_samples(
    model,
    cfg,
    transforms,
    device,
    out_dir,
    imgs_folder,
    masks_folder,
    num_samples=5,
):
    "Plot some prediction samples"
    colors = cfg.DATASET.CHANNEL2COLOR
    rgb_colors = [list(ImageColor.getcolor(color, "RGB"))
                   for color in colors]

    imgs = glob.glob(imgs_folder + "/*.png")
    samples = random.sample(imgs, num_samples)

    for sample in samples:
        _, filename = os.path.split(sample)
        filename = os.path.join(out_dir, filename)
        plot_pred2tgt(
            model,
            sample,
            rgb_colors,
            transforms,
            device,
            masks_folder,
            outname=filename)


def evaluate(preds, tgts, threshold=0.5):
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, tgts, mode='multilabel', threshold=threshold)
    iou_score = smp.metrics.iou_score(
        tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(
        tp, fp, fn, tn, reduction="micro")
    accuracy = smp.metrics.accuracy(
        tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(
        tp, fp, fn, tn, reduction="micro-imagewise")

    return iou_score, f1_score, accuracy, recall



def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
    writer,
    epoch,
    cfg,
    stats,
):
    model.train()
    train_loss = 0
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

        pred = pred.detach().cpu()
        mask = mask.detach().cpu()

        iou_score, f1_score, accuracy, recall = evaluate(pred, mask)

        iteration += 1

        stats["train"]["loss"][iteration] = loss.item()
        stats["train"]["iou_score"][iteration] = iou_score.item()
        stats["train"]["f1_score"][iteration] = f1_score.item()
        stats["train"]["accuracy"][iteration] = accuracy.item()
        stats["train"]["recall"][iteration] = recall.item()

        if iteration % 20 == 0 and i > 0:
            print(f"Iteration {iteration} - Train Loss: {train_loss / i}")
            writer.add_scalar('Loss/train', train_loss/i, iteration)
            last_20_iters = [i for i in range(iteration - 20 + 1, iteration + 1)]
            # Mean Values of metrics from the last 20 iterations
            writer.add_scalar(
                'IoU/train',
                sum([stats["train"]["iou_score"][it]
                     for it in last_20_iters]) / 20,
                iteration,
            )
            writer.add_scalar(
                'F1_score/train',
                sum([stats["train"]["f1_score"][it]
                     for it in last_20_iters]) / 20,
                iteration,
            )
            writer.add_scalar(
                'Accuracy/train',
                sum([stats["train"]["accuracy"][it]
                     for it in last_20_iters]) / 20,
                iteration,
            )
            writer.add_scalar(
                'Recall/train',
                sum([stats["train"]["recall"][it]
                     for it in last_20_iters]) / 20,
                iteration,
            )


    train_loss /= i

    return stats


def val_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    writer,
    epoch,
    cfg,
    stats,
):
    model.eval()
    val_loss = 0

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

        pred = pred.detach().cpu()
        mask = mask.detach().cpu()

        iou_score, f1_score, accuracy, recall = evaluate(pred, mask)

        val_loss += loss.item()
        iteration += 1

        stats["val"]["loss"][iteration] = loss.item()
        stats["val"]["iou_score"][iteration] = iou_score.item()
        stats["val"]["f1_score"][iteration] = f1_score.item()
        stats["val"]["accuracy"][iteration] = accuracy.item()
        stats["val"]["recall"][iteration] = recall.item()


        if iteration % 20 == 0 and i > 0:
            print(f"Iteration {iteration} - Val Loss: {val_loss / i}")
            writer.add_scalar('Loss/val', val_loss/i, iteration)

            last_20_iters = [i for i in range(iteration - 20 + 1, iteration + 1)]
            # Mean Values of metrics from the last 20 iterations
            writer.add_scalar(
                'IoU/val',
                sum([stats["val"]["iou_score"][it]
                     for it in last_20_iters]) / 20,
                iteration,
            )
            writer.add_scalar(
                'F1_score/val',
                sum([stats["val"]["f1_score"][it]
                     for it in last_20_iters]) / 20,
                iteration,
            )
            writer.add_scalar(
                'Accuracy/val',
                sum([stats["val"]["accuracy"][it]
                     for it in last_20_iters]) / 20,
                iteration,
            )
            writer.add_scalar(
                'Recall/val',
                sum([stats["val"]["recall"][it]
                     for it in last_20_iters]) / 20,
                iteration,
            )


    val_loss /= i

    return stats, val_loss


def load_model(cfg, device) -> nn.Module:
    if cfg.MODEL.NAME == "regseg":
        model = RegSeg(num_classes=len(cfg.DATASET.CLASSES))
    elif cfg.MODEL.NAME == "unet":
        model = smp.Unet(
            encoder_name=cfg.MODEL.BACKBONE,
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(cfg.DATASET.CLASSES),
        )
    elif cfg.MODEL.NAME == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=cfg.MODEL.BACKBONE,
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(cfg.DATASET.CLASSES),
        )
    elif cfg.MODEL.NAME == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=cfg.MODEL.BACKBONE,
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(cfg.DATASET.CLASSES),
        )
    else:
        raise NotImplementedError
    
    if cfg.SYSTEM.NUM_GPUS > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    
    return model
    
    

def main():
    cfg = get_cfg_defaults()
    cfg.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    imgs_root = os.path.join(cfg.DATASET.ROOT, "imgs")
    masks_root = os.path.join(cfg.DATASET.ROOT, "masks")

    outdir = f"exps/{date.today().strftime('%Y-%m-%d')}-{cfg.MODEL.NAME}"

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=False)

    weights_dir = os.path.join(outdir, "weights")
    os.makedirs(weights_dir, exist_ok=False)

    test_imgs_dir = os.path.join(outdir, "preds")
    os.makedirs(test_imgs_dir)

    with open(os.path.join(outdir, "config.yaml"), 'w') as fp:
        fp.write(cfg.dump())

    # Load the model
    model = load_model(cfg, device)
    model.train()

    train_imgs, val_imgs = train_test_split(imgs_root)

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
        cfg,
        transforms=get_train_transforms(cfg),
    )

    val_dataset = Comma10kDataset(
        imgs_root,
        masks_root,
        val_imgs,
        cfg,
        transforms=get_test_transforms(cfg),
        train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    #loss_fn = smp.losses.soft_ce.SoftCrossEntropyLoss(smooth_factor=0.15).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.BASE_LR,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        factor=0.1,
        patience=5,
        threshold=0.0001,
        min_lr=0.00000001,
    )

    writer = SummaryWriter(outdir)

    stats = {
        "train": {
            "loss": {},
            "iou_score": {},
            "f1_score": {},
            "accuracy": {},
            "recall": {},
        },
        "val": {
            "loss": {},
            "iou_score": {},
            "f1_score": {},
            "accuracy": {},
            "recall": {},
        },
    }

    best_val_loss = 10000
    best_epoch = 0

    print(f"Start training for {cfg.TRAIN.NUM_EPOCHS} epochs!")
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):

        print(f"=============== Epoch {epoch+1} ===============")
        stats = train_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            device,
            writer,
            epoch,
            cfg,
            stats,
        )
        stats, val_loss = val_epoch(
            model,
            val_dataloader,
            loss_fn,
            device,
            writer,
            epoch,
            cfg,
            stats,
        )

        test_preds_dir = os.path.join(test_imgs_dir, "epoch-{}".format(str(epoch+1).zfill(3)))
        os.makedirs(test_preds_dir)
        plot_samples(model, cfg, get_test_transforms(cfg),
                     device, test_preds_dir, imgs_root, masks_root)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pth"))
            best_val_loss = val_loss
            best_epoch = epoch

    torch.save(model.state_dict(), os.path.join(weights_dir, "final.pth"))

    print(f"Best model from epoch {best_epoch}.")

    with open(os.path.join(outdir, "train_stats.json"), 'w') as fp:
        json.dump(stats, fp, indent=4)
        
    plot_stats(
        stats, 
        os.path.join(outdir, 'train_stats.jpg'), 
        'train',
    ) 
    plot_stats(
        stats, 
        os.path.join(outdir, 'val_stats.jpg'), 
        'val',
    )


if __name__ == "__main__":
    main()

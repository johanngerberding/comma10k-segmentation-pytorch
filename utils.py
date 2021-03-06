import os
import cv2
import json 
import torch
import glob 
import random 
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageColor
import segmentation_models_pytorch as smp


def load_train_test(exp_root):
    train = []
    val = []
    trainfile = os.path.join(exp_root, "train.txt")
    valfile = os.path.join(exp_root, "val.txt")

    with open(trainfile, 'r') as fp:
        train.append(fp.readline())

    with open(valfile, 'r') as fp:
        val.append(fp.readline())

    return train, val


def predict(model, img_path, transforms, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed_img = transforms(image=img)["image"]
    transformed_img = transformed_img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(transformed_img)

    pred = torch.sigmoid(pred)
    pred = pred.detach().cpu().squeeze().numpy()

    return pred


def plot_segmentation(
    model,
    img_path,
    colors,
    transforms,
    device,
    outname,
    figsize=(12,5),
    img_weight=0.5,
):
    pred = predict(model, img_path, transforms, device)
    classMap = np.argmax(pred, axis=0)
    colors = np.array(colors).astype("uint8")
    colored_mask = colors[classMap]

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    colored_mask = cv2.resize(
        colored_mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    output = ((img_weight * image) + ((1. - img_weight) * colored_mask)).astype("uint8")

    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(output)
    plt.savefig(outname)
    plt.close('all')


def plot_pred2tgt(
    model,
    img_path,
    colors,
    transforms,
    device,
    masks_folder,
    outname,
    figsize=(12,5),
    img_weight=0.5,
):
    _, filename = os.path.split(img_path)
    mask_path = os.path.join(masks_folder, filename)

    pred = predict(model, img_path, transforms, device)
    classMap = np.argmax(pred, axis=0)
    
    colors = np.array(colors).astype("uint8")
    colored_mask = colors[classMap]

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tgt_mask = cv2.imread(mask_path)
    tgt_mask = cv2.cvtColor(tgt_mask, cv2.COLOR_BGR2RGB)

    colored_mask = cv2.resize(
        colored_mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    output = ((img_weight * image) + ((1. - img_weight) * colored_mask)).astype("uint8")

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].imshow(output)
    axs[0].set_title('Prediction')
    axs[0].axis('off')
    axs[1].imshow(tgt_mask)
    axs[1].set_title('Groundtruth')
    axs[1].axis('off')
    plt.tight_layout()
    fig.savefig(outname)
    plt.close('all')
    
    
def smoothing(vals: list, window_size: int = 50):
    num_windows = len(vals) // window_size
    nvals = [sum(vals[i*window_size :(i+1) * window_size]) / window_size 
                 for i in range(num_windows)]
    steps = [i*window_size for i in range(num_windows)]
    
    if len(vals) % window_size != 0:
        last = sum(vals[num_windows * window_size:]) / len(vals[num_windows * window_size:])
        nvals.append(last)
        steps.append(len(vals) - 1)
    
    return nvals, steps


def plot_stats(
    file_path: str, 
    outname: str, 
    part: str = 'train',
    figsize: tuple = (25,5), 
    smooth: int = 50
):
    with open(file_path, 'r') as fp:
        stats = json.load(fp)
        
    loss = [v for _, v in stats[part]['loss'].items()]
    ious = [v for _, v in stats[part]['iou_score'].items()]
    f1 = [v for _, v in stats[part]['f1_score'].items()]
    acc = [v for _, v in stats[part]['accuracy'].items()]
    recall = [v for _, v in stats[part]['recall'].items()]

    sm_loss, steps = smoothing(loss, smooth)
    sm_ious, _ = smoothing(ious, smooth)
    sm_f1, _ = smoothing(f1, smooth)
    sm_acc, _ = smoothing(acc, smooth)
    sm_recall, _ = smoothing(recall, smooth)
    
    fig, axs = plt.subplots(1, 5, figsize=figsize)
    axs[0].plot(steps, sm_loss)
    axs[0].set_title(f'{part} Loss')
    axs[0].set_xlabel('Iterations')
    axs[1].plot(steps, sm_ious)
    axs[1].set_title(f'{part} IoU')
    axs[1].set_xlabel('Iterations')
    axs[2].plot(steps, sm_f1)
    axs[2].set_title(f'{part} F1 Score')
    axs[2].set_xlabel('Iterations')
    axs[3].plot(steps, sm_acc)
    axs[3].set_title(f'{part} Accuracy')
    axs[3].set_xlabel('Iterations')
    axs[4].plot(steps, sm_recall)
    axs[4].set_title(f'{part} Recall')
    axs[4].set_xlabel('Iterations')
    plt.tight_layout()
    fig.savefig(outname)


def plot_losses(
    file_path: str, 
    outname: str, 
    figsize: tuple = (12,5), 
    smooth: int = 50
):
    with open(file_path, 'r') as fp:
        stats = json.load(fp)
        
    train_loss = [v for _, v in stats['train']['loss'].items()]
    val_loss = [v for _, v in stats['val']['loss'].items()]
    
    sm_train_loss, train_steps = smoothing(train_loss, smooth)
    sm_val_loss, val_steps = smoothing(val_loss, smooth)
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].plot(train_steps, sm_train_loss)
    axs[0].set_title('Train Loss')
    axs[0].set_xlabel('Iterations')
    
    axs[1].plot(val_steps, sm_val_loss)
    axs[1].set_title('Val Loss')
    axs[1].set_xlabel('Iterations')
    
    plt.tight_layout()
    fig.savefig(outname)



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

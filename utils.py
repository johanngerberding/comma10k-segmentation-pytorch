import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


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

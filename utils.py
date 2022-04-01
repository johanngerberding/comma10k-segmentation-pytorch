import cv2
import torch 
import numpy as np 
import matplotlib.pyplot as plt 


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
    classes,
    transforms, 
    device, 
    figsize=(16,9),
    img_weight=0.5,
):
    pred = predict(model, img_path, transforms, device)
    classMap = np.argmax(pred, axis=0)
    
    # pick some random colors
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes) - 1, 3), dtype="uint8")
    colors = np.vstack([[0, 0, 0], colors]).astype("uint8")
    
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
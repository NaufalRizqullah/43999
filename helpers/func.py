import torch

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from PIL import Image
import io


def load_weights(pth_path, device, model, opt=None, lr_scheduler=None):
    # Load file .pth of dict
    checkpoint = torch.load(pth_path, map_location=torch.device(device))

    # Load model dict
    model.load_state_dict(checkpoint["model"])
    print("[INFO] Model set using Pretrained State Dict")

    # Load Optimizer if provided
    if opt is not None:
        optimizer_state_dict = checkpoint.get("optimizer", None)
        if optimizer_state_dict:
            opt.load_state_dict(optimizer_state_dict)
            print("[INFO] Optimizer set using Pretrained State Dict")
        else:
            print("[WARNING] Optimizer state dict not found in checkpoint. Optimizer not loaded.")

    # Load LR scheduler if provided
    if lr_scheduler is not None:
        lr_scheduler_state_dict = checkpoint.get("lr_scheduler", None)
        if lr_scheduler_state_dict:
            lr_scheduler.load_state_dict(lr_scheduler_state_dict["state_dict"])
            print("[INFO] LR Scheduler set using Pretrained State Dict")
        else:
            print("[WARNING] LR Scheduler state dict not found in checkpoint. LR Scheduler not loaded.")


def load_label(path="./helpers/class_name_inference.json"):
    file = Path(path)

    with open(file, "r") as f:
        data = json.load(f)
        class_names = [data[key] for key in sorted(data.keys(), key=lambda x: int(x))]

    return class_names

def show_one_image_label_score(image, boxes, labels, scores):
    fig, ax = plt.subplots(1)
    
    # Convert the image tensor to numpy array and display
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        box = box.cpu().detach().numpy()
        x1, y1, x2, y2 = box
        # Calculate width and height of the bounding box
        width = x2 - x1
        height = y2 - y1
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add label and score
        ax.text(x1, y1, f'Class {label} - Score: {score:.2f}', color='r', fontsize=10, backgroundcolor='w')

    # Instead of showing, save the modified image to a BytesIO object
    img_byte_arr = io.BytesIO()
    plt.savefig(img_byte_arr, format='png')
    plt.close(fig)
    img_byte_arr.seek(0)
    
    # Convert the BytesIO object to a PIL image
    modified_image = Image.open(img_byte_arr)
    
    return modified_image


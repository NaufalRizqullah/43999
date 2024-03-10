import torch

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms


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
    # Convert PyTorch tensor to a PIL Image object
    img_pil = transforms.ToPILImage()(image)

    draw = ImageDraw.Draw(img_pil)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        # Calculate width and height of the bounding box
        width = x2 - x1
        height = y2 - y1
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

        # Add label and score
        # draw.text((x1, y1), f'Class {label} - Score: {score:.2f}', fill='red')

    return img_pil


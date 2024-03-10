import  gradio as gr
import torch
import os
from torchvision import transforms
from timeit import default_timer as timer

from helpers.model import fasterrcnn_backbone_resnet101
from helpers.func import load_weights, load_label, show_one_image_label_score


# Setup Device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Make and Load Model.
NUM_CLASSES = 196 + 1 # class + background

model = fasterrcnn_backbone_resnet101(
    num_classes=NUM_CLASSES,
    freeze_model=True
)

path_pth = "./models/checkpoint_backbone_resnet101_epoch_206.pth"
load_weights(
    pth_path=path_pth,
    device=DEVICE,
    model=model
)

# Transformations
t = transforms.Compose([
    transforms.Resize((400, 600)),
    transforms.ToTensor(),
])

# Setup class names
class_names = load_label()

# Create prediction functions
def inference(img):
    # Start time
    start_time = timer()

    # transforms the target image and add a batch dimension
    img_transormed = t(img).unsqueeze(0)

    # Inference the image
    model.eval()
    with torch.inference_mode():
        predictions = model(img_transormed)

    # Extract predictions
    prediction = predictions[0]
    boxes = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']

    # Get the class with the highest score for each object
    best_class_index = scores.argmax().item()
    name_label = class_names[labels[best_class_index]]

    best_score = scores[best_class_index].item()

    best_class_indices = [name_label]
    best_scores = [best_score]

    result_image = show_one_image_label_score(img, boxes, best_class_indices, best_scores)


    # Calculate the prediction time
    end_time = timer()
    inference_time = round(end_time - start_time, 5)
    
    # So it will return 3 component, so gradio will prepare output:
    # - image
    # - Label (for Text class)
    # - Number (for time Inferences)
    return result_image, name_label, inference_time


# Make Gradio App.

# Create title, description and Article.
title = "[DEMO] ---"
description = "Demo of ---"
article = "---"

# Create examples list from "examples/" directory
path_example = "./examples/"
example_list = [[path_example + example] for example in os.listdir(path_example)]

# Create Gradio interface
print(f"[INFO] Initialize Instance of Gradio....")
demo = gr.Interface(
    fn=inference,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(),
        gr.Label(num_top_classes=5, label="Class Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
)
print(f"[INFO] Initialize Instance Gradio Done!")

# Launch the app!
print(f"[INFO] Launching the App....")
demo.launch()





"""
----------------------------------------------------
[program desc]
----------------------------------------------------
Author: Salar Haider
ID: 169026899
Email: haid6899@mylaurier.ca
__updated__= "2022-10-01"
----------------------------------------------------
"""

import torch
from torchvision import transforms
from PIL import Image
#import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # conv layer 1 - 1 channel input (grayscale), 10 channels of output, 5x5 filters scanning the image)
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        # some 5x5 filters are turned off in this layer
        self.conv2d_droplayer = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2d_droplayer(self.conv2(x)), 2))        
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = NeuralNet()
model.load_state_dict(torch.load('results/model.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure 1 channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict(input_data):
    # 1) grab the sketch out of the dict (if it’s a dict)
    if isinstance(input_data, dict):
        if "image" in input_data:
            raw = input_data["image"]
        elif "composite" in input_data:
            raw = input_data["composite"]
        else:
            raise ValueError("Dict input missing expected image keys.")
    else:
        raw = input_data

    # 2) now normalize raw into a PIL Image
    if isinstance(raw, np.ndarray):
        img = Image.fromarray(np.uint8(raw)).convert("L")
    elif isinstance(raw, Image.Image):
        img = raw.convert("L")
    else:
        raise TypeError(f"Unsupported input type: {type(raw)}")

    # 3) invert & preprocess as before
    img = ImageOps.invert(img)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()
    return f"Prediction: {pred}"

gr.Interface(
   fn = predict,
   inputs = gr.Sketchpad(image_mode="L",
                     height = 280,
                     width = 280,
                     ),
                     outputs="text",
                     title="MNIST Digit Classifier").launch()
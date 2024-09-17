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
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
#import matplotlib.pyplot as plt
import gradio as gr
import numpy as np

# def get_mnist_image(dataset, index):
#     '''Function to get an image and label from the MNIST dataset.'''
#     image, label = dataset[index]
#     return image, label

# transforms.Compose applies a bunch of transformations onto an image right after eachother
# transform.ToTensor() turns the image into numbers by splitting the image into RGB 0-255, and then flattening that into 0-1
# transforms.Normalize(0.5, 0.5) shifts the pixel values so instead of ranging from 0 to 1, now they range from -1 to 1, the two parameters are to subtract the value by 0.5, and then divide it by 0.5, this centers the pixel values around 0
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5), (0.5)),])

# load model
model = torch.load('./my_mnist_model.pt')
model.eval()

# # Load the test dataset
# test_data = datasets.MNIST(
#     root="/Users/salar/data",
#     train=False,
#     download=True,
#     transform=transform
# )

def predict(image):
    
   #image = Image.open(image_path)
   image = transform(image)
   image = image.view(1, 784)
   with torch.no_grad():
        logps = model(image)
        ps = torch.exp(logps)
        prob = list(ps.numpy()[0])
        predicted_label = prob.index(max(prob))
        
   return predicted_label
    
gr.Interface(fn = predict, inputs = "sketchpad", outputs = "label").launch()

    
# # Get an image from the dataset
# index = 60  # Change this index to get a different image
# image, label = get_mnist_image(test_data, index)
#
# # Make a prediction
# predicted_digit = predict(image)
# print(f'Predicted Digit: {predicted_digit}')
# print(f'True Label: {label}')
#
# # Display the image
# plt.imshow(image.squeeze(), cmap='gray')
# plt.title(f'Predicted Digit: {predicted_digit}, True Label: {label}')
# plt.show()



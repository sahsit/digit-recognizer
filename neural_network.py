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

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


# transforms.Compose applies a bunch of transformations onto an image right after eachother
# transform.ToTensor() turns the image into numbers by splitting the image into RGB 0-255, and then flattening that into 0-1
# transforms.Normalize(0.5, 0.5) shifts the pixel values so instead of ranging from 0 to 1, now they range from -1 to 1 
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))])



# downloading the training set: MNIST is the set
training_data = datasets.MNIST(
    # where to store the pictures
    root = "/Users/salar/handwritten_digit_classifier_dir/data",
    # this is a training data set
    train = True,
    # download the data from the internet if its not at 'root'
    download = True,
    # applying the transformations to the dataset that we did earlier
    transform = transform)

# downloading the test set
testing_data = datasets.MNIST(
    # where to store the pictures
    root = "/Users/salar/handwritten_digit_classifier_dir/data",
    # this is a testing data set
    train = False,
    # download the data from the internet if its not at 'root'
    download = True,
    # applying the transformations to the dataset that we did earlier
    transform = transform)

# now that we've downloaded the data, we need to pass them in "mini batches", reshuffle to avoid over fitting, etc. so we use DataLoader
train_dataloader = DataLoader(
                              # specifying which dataset you're referring to
                              training_data, 
                              # split the set up into batches of 64
                              batch_size = 64, 
                              shuffle = True)

test_dataloader = DataLoader(
                             testing_data, 
                             batch_size = 64, 
                             shuffle = True)

# SEE THE SHAPE OF THE TENSORS BELOW

#for batch_id, (images, labels) in enumerate(test_dataloader):
    #print(batch_id, images.shape, labels.shape)
    # print(image.shape) - torch.Size([64, 1, 28, 28]) 64 images in one batch, 1 channel (grayscale),  28x28 pixels, this is a 4d tensor
    # print(labels.shape) - torch.Size([64]) 64 labels because 64 images in one batch

# SEE THE ACTUAL MNIST IMAGES BELOW

# images, labels = next(iter(test_dataloader))
# plt.imshow(images[0].squeeze(), cmap = "gray")
# plt.title(f"Label: {labels[0]}")
# plt.show()


# 784 = 28*28 for each pixel
# 128 --> 64 hidden layer sizes
# output size = 10 - one for each digit


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # conv layer 1 - 1 chantelnel input (grayscale), 10 channels of output, 5x5 filters scanning the image)
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
    

network = NeuralNet()

# Building a training and testing loop
    # 0. Loop through the data
    # 1. Forward pass to make predictions on data (forward propogation)
    # 2. Calculate the loss
    # 3. Optimizer zero grad
    # 4. Loss backward - moves backward thru network to calc gradient
    # 5. Optimizer step - use optimizer against model's parameters to improve loss


# torch.optim is the syntax to initialize a optimizer
# params = model.parameters() specifies that the optim will update the params of model
# learning rate is how big a step the optim will take each step
optimizer = optim.SGD(network.parameters(), lr = 0.01, momentum = 0.05)


epochs = 25
train_losses = []

def train(epoch):
    network.train()
    for batch_id, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_id % 10 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epoch, batch_id * len(data), len(train_dataloader.dataset),
            100. * batch_id / len(train_dataloader), loss.item()))
        train_losses.append((batch_id*64) + ((epoch-1) * len(train_dataloader.dataset)))
        torch.save(network.state_dict(), 'results/model.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')

test_losses = []

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_dataloader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_dataloader.dataset),
    100. * correct / len(test_dataloader.dataset)))
            
test()
for epoch in range(1, 4):
    train(epoch)
    test()

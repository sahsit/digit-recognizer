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


# transforms.Compose applies a bunch of transformations onto an image right after eachother
# transform.ToTensor() turns the image into numbers by splitting the image into RGB 0-255, and then flattening that into 0-1
# transforms.Normalize(0.5, 0.5) shifts the pixel values so instead of ranging from 0 to 1, now they range from -1 to 1, the two parameters are to subtract the value by 0.5, and then divide it by 0.5, this centers the pixel values around 0
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5), (0.5)),])



# downloading the training set: MNIST is the set
training_data = datasets.MNIST(
    # where to store the pictures
    root = "/Users/salar/data",
    # this is a training data set
    train = True,
    # download the data from the internet if its not at 'root'
    download = True,
    # applying the transformations to the dataset that we did earlier
    transform = transform)

# downloading the test set
testing_data = datasets.MNIST(
    # where to store the pictures
    root = "/Users/salar/data",
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




#print(image.shape) - torch.Size([64, 1, 28, 28]) 64 images in one batch, 28x28 pixels
#print(labels.shape) - 64 labels because 64 images in one batch

"""
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(image[index].numpy().squeeze(), cmap='gray_r')
    
plt.show()
"""

# 784 = 28*28 for each pixel
# 128 --> 64 hidden layer sizes
# output size = 10 - one for each digit

# nn.Sequential creates the neural network as a sequence of layers
model = nn.Sequential(
                    nn.Linear(784, 128),
                    # activation function - ReLU just means that if the output is negative, round it to 0, and if it's positive, then leave it
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10),
                    # Converts the outputs into log-probabilities for each class, which are used for classification.
                    nn.LogSoftmax(dim=1)
                    )


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
optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)
# using negative log-likelihood loss function
criterion = nn.NLLLoss()
# fetches one batch (64) of images with their labels
image, labels = next(iter(train_dataloader))
# this flattens the image from a 28x28 square to a 782-dimensional vector
image = image.view(image.shape[0], -1)
# this passes the 78-dimensional vector to the neural network, and computes the log-probability for each class (0-9)
logps = model(image)
# this uses the NLLLoss to compare the model's log-probs with actual labels and compute the negative log-likelihood
loss = criterion(logps, labels) 

epochs = 25

for epoch in range(epochs):
    running_loss = 0
    for image, labels in train_dataloader:
        # this flattens the image from a 28x28 square to a 782-dimesional vector
        image = image.view(image.shape[0], -1)
        # resets graidents of model param to 0 so it doesn't accumulate, giving us the wrong values
        optimizer.zero_grad()
        # pass the image to the model
        output = model(image)
        # use NLLLoss to get diff. between true labels and log-probabilities and compute the negative log-likelihood
        loss = criterion(output, labels)
        # calculates how much each param contributes to overall loss
        loss.backward()
        #update the model's params with new values for the next epoch
        optimizer.step()
        # update running loss
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(train_dataloader)))

        
      
def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()
    

num_of_images = 60
image, labels = next(iter(test_dataloader))
for index in range(1, num_of_images + 1):
    with torch.no_grad():
        img = image[index].view(1, 784)
        logps = model(img)

    plt.imshow(image[index].numpy().squeeze(), cmap='gray_r')
    
plt.show()
    
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)

correct_count, all_count = 0, 0
for images,labels in train_dataloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

torch.save(model, './my_mnist_model.pt')

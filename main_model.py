# Import necessary libraries.
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import torch
import torchvision
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.optim import Adam
from torchvision.io import read_image
from sklearn.metrics import plot_confusion_matrix
from skorch import NeuralNetClassifier
import sklearn


path = "add the dataset path"
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=torchvision.transforms.Compose([transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                #Resizing all of the images to (227,227)
                transforms.Resize((227,227)),
                #Normalization by using the mean and variance 
                transforms.Normalize((0.3731, 0.3731, 0.3731),(0.2812, 0.2812, 0.2812))]), target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# Read Dataset   
data_set=CustomImageDataset(annotations_file='add the .csv file path', img_dir='add the dataset path')

print('Train data set:', len(data_set))


#build dataloader
train_dataloader = DataLoader(data_set, batch_size=16, shuffle=True)
path = r"C:\Users\malik\Desktop\Medical Image Processing\project\code\dataset_folders\augmented data\train_aug_dataset"

classes = ('benign', 'malignant', 'normal') # 3 classes

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # First layer
            nn.ReLU(inplace=True),  # ReLU activation function

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # Second layer
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.ReLU(inplace=False),  # ReLU activation function
            nn.BatchNorm2d(32),  # Batch Normalization

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # Third layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Dropout(p=0.2),  # Dropout

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # Fourth layer
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
            nn.ReLU(inplace=False),  # ReLU activation function
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.Dropout(p=0.3),  # Dropout

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # Fifth layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.BatchNorm2d(128),  # Batch Normalization

        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(401408, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3)  # number of classes
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)

        return x
    
# Train
network = CNN()

loss_function = nn.CrossEntropyLoss()   
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
num_epochs=3
total_step = len(train_dataloader)
loss_list = []
acc_list = []

for epoch in range(0, num_epochs):
    print(f'Starting epoch {epoch+1}')
    current_loss = 0.0
    total_batches = 0
    avg_loss_epoch = 0
    epoch_accuracy = 0
    epoch_loss = 0

      
    for i, data in enumerate(train_dataloader):
        inputs, targets = data
                               
        # Perform forward pass
        outputs = network(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        loss_list.append(loss.item())

        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Train accuracy
        total = targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        acc_list.append(correct / total)
        
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(correct / total) * 100))
        epoch_accuracy += ((correct / total) * 100)/total_step
        epoch_loss += loss.item()/total_step
    print('Epoch [{}/{}], Loss_Epoch: {:.4f}, Accuracy_Epoch: {:.2f}%'.format(epoch + 1, num_epochs, epoch_loss, epoch_accuracy))
        
print('Training process has finished. Saving trained model.')
torch.save(network.state_dict(), "add a path for save the model\saved_augmented_modelgan.pth")


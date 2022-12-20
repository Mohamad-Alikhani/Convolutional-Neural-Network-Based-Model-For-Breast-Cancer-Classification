# Import necessary libraries.
import pickle
import matplotlib.pyplot as plt
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from skorch import NeuralNetClassifier
import sklearn
import seaborn as sns

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

test_data_size = len(data_set)
print('Test data set:', len(data_set))

batch_size = 16
test_dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True)



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


network = CNN()

#test
print('Starting testing')
network.load_state_dict(torch.load('The saved model path'))
network.eval()
device="cpu"
arr=0

with torch.no_grad():
    correct = 0
    for inputs, targets in test_dataloader:
        y_val = network(inputs)
        predicted = torch.max(y_val,1)[1]

        if len(targets) != batch_size:

            targets = F.pad(input=targets, pad=(0, batch_size-len(targets)), mode='constant', value=0)
            predicted = F.pad(input=predicted, pad=(0, batch_size-len(predicted)), mode='constant', value=0)
            print(targets)

        else:

            print(predicted)
            print(targets)

            print(len(targets))
            correct += (predicted == targets).sum()
            arr += confusion_matrix(targets, predicted)
            print(arr)
print(f'Test accuracy: {correct.item()}/{test_data_size} = {correct.item()*100/(test_data_size):7.3f}%')
accuracy=0
trace=np.trace(arr)
accuracy=trace/test_data_size
print('Accuracy : %.3f %%' % (100.0 * accuracy))
sum_precison=0
sum_recall=0
f1_measure=0
for i in range(1, 4):
    sum_precison+=(arr[i-1][i-1])/(arr[i-1][0]+arr[i-1][1]+arr[i-1][2])
    print(sum_precison)
    sum_recall+=(arr[i-1][i-1])/(arr[0][i-1]+arr[1][i-1]+arr[2][i-1])
precision=sum_precison/3
recall=sum_recall/3
f_measure= 2*(precision*recall)/(precision+recall)
print('Precision : %.3f %%' % (100.0 * precision))
print('Recall : %.3f %%' % (100.0 * recall))
print('F_measure : %.3f %%' % (100.0 * f_measure))
    


# Display the confusion matrix 

class_names = ['benign', 'malignant', 'normal']
df_cm = pd.DataFrame(arr, class_names, class_names)
plt.figure(figsize = (9,6))
sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
plt.xlabel("Target Class")
plt.ylabel("Output Class")
plt.show() 






# Import necessary libraries.

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
import pandas as pd
from torchvision.io import read_image





path = "add the dataset path"


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=torchvision.transforms.Compose([transforms.ToPILImage(),
                                                                                            transforms.Grayscale(num_output_channels=3),
                                                                                            transforms.ToTensor(),
                                                                                            # Resizing all of the images to (227,227)
                                                                                            transforms.Resize((227, 227)),
                                                                                            # Normalization by using the mean and variance
                                                                                            ]),target_transform=None):
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
data_set = CustomImageDataset(annotations_file='add the .csv file path', img_dir='add the dataset path')


# build dataloader
dataloader = DataLoader(data_set, batch_size=16, shuffle=True)

psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
channels_sum = 0
num_batches = 0
channels_squared_sum = 0

for data, _ in dataloader:
    channels_sum += torch.mean(data, dim=[0, 2, 3])  # Computing the mean
    channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
    num_batches += 1

mean = channels_sum / num_batches

std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5  # Computing the standard deviation

# output
print('mean: ' + str(mean))
print('std:  ' + str(std))
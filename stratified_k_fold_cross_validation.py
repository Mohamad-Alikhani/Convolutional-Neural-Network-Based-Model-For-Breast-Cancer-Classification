import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import random
from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
import torchvision
from torchvision import datasets,transforms
import torchvision.transforms as transforms
import pandas as pd
import os
from torchvision.io import read_image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.optim import Adam
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
from sklearn.metrics import plot_confusion_matrix
from skorch import NeuralNetClassifier
import sklearn
from sklearn.datasets import load_digits
import shutil


path = "add the dataset path"
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=torchvision.transforms.Compose([transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                #Resizing all of the images to (227,227)
                transforms.Resize((227,227)),
                #Normalization by using the mean and variance 
                transforms.Normalize(mean=(0.3731, 0.3731, 0.3731),std=(0.2812, 0.2812, 0.2812))]), target_transform=None):
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
'''evaluation part'''   
def evaluation(output, predicted, labels):
    print(classification_report(output, predicted, digits=2, target_names=labels))
    arr = confusion_matrix(output, predicted)
    confusion_plot(arr)
    return output, predicted
'''confusion matrix'''
def confusion_plot(arr):
    class_names = ['benign', 'malignant', 'normal']
    df_cm = pd.DataFrame(arr, class_names, class_names)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.show() 
    accuracy=0
    trace=np.trace(arr)
    accuracy=trace/len(test_set)
    print('Accuracy : %.3f %%' % (100.0 * accuracy))
    sum_precison=0
    sum_recall=0
    f_measure=0
    for i in range(1, 4):
        sum_precison += (arr[i - 1][i - 1]) / (arr[i - 1][0] + arr[i - 1][1] + arr[i - 1][2])
        print(sum_precison)
        sum_recall += (arr[i - 1][i - 1]) / (arr[0][i - 1] + arr[1][i - 1] + arr[2][i - 1])
    precision = sum_precison / 3
    recall = sum_recall / 3
    f_measure = 2 * (precision * recall) / (precision + recall)
    print('Precision : %.3f %%' % (100.0 * precision))
    print('Recall : %.3f %%' % (100.0 * recall))
    print('F_measure : %.3f %%' % (100.0 * f_measure))
    

network = CNN()
criterion = nn.CrossEntropyLoss()   
num_epochs=5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
loss_list = []
acc_list = []
batch_size=16
k=5
foldperf={}

'''training part'''

def train_epoch(network,device,train_loader,loss_fn,optimizer):
    
    train_loss,train_correct=0.0,0
    for images,labels in train_loader:
        optimizer.zero_grad()
        output = network(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        train_loss += loss.item() 
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
        
    return train_loss,train_correct



'''testing part'''        
  
def valid_epoch(network,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    output_list=[]
    predicted_list=[]
    network.eval()
    for images,labels in dataloader:

        
        output = network(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == labels).sum().item()
        output_list += list(labels.numpy())
        predicted_list += list(predictions.numpy())

    return valid_loss,val_correct,  output_list, predicted_list

def makelabel(name):
    os.chdir(rf'add the dataset path\{name}')

    folders = ["benign", "malignant", "normal"]

    files = []
    count = 0;
    for folder in folders:
        for file in os.listdir(folder):
            if (".jpg" in file) or (".png" in file) or (".jpeg" in file):
                files.append([file, count])
        count += 1

    pd.DataFrame(files, columns=['image name', 'label']).to_csv(f'{name}_labels.csv')


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


# Destination path
train_destination = 'train destination path\train'
test_destination = 'test destination path\test'


'''k-fold start'''
for fold in range(1,k+1):
    print('Fold {}'.format(fold))
    try:
        shutil.rmtree('add dataset path\train\benign')
        shutil.rmtree('add dataset path\train\malignant')
        shutil.rmtree('add dataset path\train\normal')
        shutil.rmtree('add dataset path\test\benign')
        shutil.rmtree('add dataset path\test\malignant')
        shutil.rmtree('add dataset path\test\normal')
    except FileNotFoundError:
        print("")

    for i in range(1, k+1):
        if i == fold:
            copytree('add dataset path\{i}',
                     test_destination, symlinks=False, ignore=None)
        else:
            copytree('add dataset path\{i}',
                     train_destination, symlinks=False, ignore=None)

    makelabel('train')
    makelabel('test')
    train_labels = pd.read_csv(
        'add a path for .csv file\train_labels.csv')
    augmented_labels = pd.read_csv(
        'add a path for .csv file\augmented_labels.csv')

    pd.concat([train_labels, augmented_labels]).to_csv(
        'add a path for .csv file\augmented_train_labels.csv',
        index=False)

    train_set = CustomImageDataset(
        annotations_file='add .csv file path\augmented_train_labels.csv',
        img_dir='add the dataset path')
    test_set = CustomImageDataset(
        annotations_file='add .csv file path\test_labels.csv',
        img_dir='add the dataset path')

    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    print('Train data set:', len(train_set))
    print('Test data set:', len(test_set))
    print('Total data set:',len(train_set)+len(test_set))

    network = CNN()
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    train_loss_list_epoch = []
    test_loss_list_epoch = []
    train_acc_list_epoch = []
    test_acc_list_epoch = []
    for epoch in range(num_epochs):
        train_loss, train_correct=train_epoch(network,device,train_loader,criterion,optimizer)
        test_loss, test_correct,  output_list, predicted_list =valid_epoch(network,device,test_loader,criterion)

        train_loss = train_loss / len(train_set)
        train_acc = train_correct / len(train_set) * 100
        test_loss = test_loss / len(test_set)
        test_acc = test_correct / len(test_set) * 100
        
        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1, num_epochs, train_loss, test_loss, train_acc, test_acc))
        train_loss_list_epoch.append(train_loss)
        test_loss_list_epoch.append(test_loss)
        train_acc_list_epoch.append(train_acc)
        test_acc_list_epoch.append(test_acc)


    history['train_loss'].append(mean(train_loss_list_epoch))
    history['test_loss'].append(mean(test_loss_list_epoch))
    history['train_acc'].append(mean(train_acc_list_epoch))
    history['test_acc'].append(mean(test_acc_list_epoch))

    foldperf['fold{}'.format(fold)] = history
    
    evaluation( output_list, predicted_list, classes)




'''k folds average'''
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(1, k+1):
    train_loss_list.append(foldperf[f'fold{i}']['train_loss'][0])
    test_loss_list.append(foldperf[f'fold{i}']['test_loss'][0])
    train_acc_list.append(foldperf[f'fold{i}']['train_acc'][0])
    test_acc_list.append(foldperf[f'fold{i}']['test_acc'][0])


print('Performance of {} fold cross validation'.format(k))
print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f} \n Minimum Training Loss: {:.3f} \t Minimum Test Loss: {:.3f} \t Minimum Training Acc: {:.2f} \t Minimum Test Acc: {:.2f}\n Maximum Training Loss: {:.3f} \t Maximum Test Loss: {:.3f} \t Maximum Training Acc: {:.2f} \t Maximum Test Acc: {:.2f}".format(mean(train_loss_list),mean(test_loss_list),mean(train_acc_list),mean(test_acc_list), min(train_loss_list),min(test_loss_list),min(train_acc_list),min(test_acc_list), max(train_loss_list),max(test_loss_list),max(train_acc_list),max(test_acc_list)))



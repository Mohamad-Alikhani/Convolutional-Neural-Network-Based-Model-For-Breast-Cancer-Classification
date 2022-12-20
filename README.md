**Convolutional-Neural-Network-Based-Models-For-Breast-Cancer-Classification**

**Dataset**

Breast Ultrasound Images (BUSI) containing 780 images in 3 different categories (210 malignant, 437 benign and 133 normal) is used in this project.https://scholar.cu.edu.eg/?q=afahmy/pages/dataset

**Train and Test dataset**

There are two approaches for extending the unbalanced dataset to train the network. First, adding the augmented data to the classes which have less data to equalize amount of data in all three classes. Second, adding constant percent of augmented data to each class but in this method, we have to divide training and testing dataset based on the percent of the data in each class. I chose second approach and divided the dataset to 20% for testing and 80% for training randomly based on the percent of data in each class. For instance, in this case (133/780)% from class “Normal”, (210/780)% from class “Benign” and (437/780)% from class “Malignant” are chosen for both training and testing datasets. 

**CNN network**

I trained different types of CNNs and finally chose two best of them which provided the better accuracy for our datasets. For evaluating the performance of the networks I used stratified K-folds cross validation technique for evaluating them, thus I chose the better network between them. 

**Stratified k-folds cross validation**

Stratified k-folds cross validation technique is used for evaluating the networks. In k-fold cross-validation technique the whole dataset split into k folds and (k-1) folds are used for training and one fold is used for testing. This procedure iterates k times and calculates the loss and accuracy in each iteration. Therefore, calculating the minimum, maximum and mean of accuracy for the model can show the real performance. In unbalanced datasets, stratified k-folds cross validation technique is usually used to gain a proper performance. In this method the dataset is divided to k-folds but based on the percentage of each class

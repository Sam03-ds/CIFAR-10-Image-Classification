## CIFAR-10 Image Classification
This project implements a deep learning model for classifying images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to create a neural network model that can correctly classify these images based on their content.

### Project Overview
This project implements a deep learning model for classifying CIFAR-10 images using PyTorch.

### Directory Structure
model.py — Model architecture.
train.py — Script to train the model.
test.py — Script to evaluate the trained model on the test set.
utils.py — Utility functions.
requirements.txt — Project dependencies.
README.md — Documentation and setup instructions.

### Technologies Used
Python 3.8+
PyTorch and torchvision for deep learning
Matplotlib and Seaborn for visualization
Numpy for data manipulation

### How to Run the Project
1. Install Dependencies
Clone the repository and install required libraries:
  ```shell
  git clone https://github.com/Sam03-ds/CIFAR-10-Image-Classification.git
  cd cifar10-classification
pip install -r requirements.txt
```

2. Train the Model
Train the model using the following command:
```shell
python train.py --epochs 50 --batch_size 128 --lr 0.001
```
3. Evaluate the Model
Evaluate the trained model on the test dataset:
```shell
python test.py --model_path model_weights.pth
```
### Results
The model achieves 85-90% accuracy on the CIFAR-10 test dataset. Performance may vary based on hyperparameters and data augmentation techniques.





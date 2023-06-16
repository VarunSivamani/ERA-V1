# Session - 7 
## In-Depth Coding Practice

<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-v3.7.1-orange)](https://matplotlib.org/stable/index.html)

<br>

# Target

- 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 8000 Parameters
- Do this using your modular code. 

<br>

# Solution

This repository contains 6 `.ipynb` notebooks which has 6 different models trained, `model.py` and `utils.py` file.

<br>

# Model 1 - [Notebook](https://github.com/VarunSivamani/ERA-V1/blob/main/Session%207/S7_Model1.ipynb)


### Target:

- Get the set-up right.
- Set Transforms.
- Set Data Loader.
- Set Basic Working Code.
- Set Basic Training  & Test Loop.

### Results:

- Parameters: 6.3M
- Best Training Accuracy: 99.93
- Best Test Accuracy: 99.26

### Analysis:

- Extremely Heavy Model for such a problem.
- Model is over-fitting, but we are changing our model in the next step.

<br>

# Model 2 - [Notebook](https://github.com/VarunSivamani/ERA-V1/blob/main/Session%207/S7_Model2.ipynb)


### Target:

- Getting the model skeleton right.
- Create a Setup (dataset, data loader, train/test steps and log plots)
- Defining a simple model with Convolution block, GAP layers.

### Results:

- Parameters: 7,272
- Best Training Accuracy: 98.45
- Best Test Accuracy: 98.58

### Analysis:

- Model with 7K parameters is able to reach till 98.58 accuracy in 15 epochs.
- Model is slightly overfitting.

<br>

# Model 3 - [Notebook](https://github.com/VarunSivamani/ERA-V1/blob/main/Session%207/S7_Model3.ipynb)


### Target:

- Add `BatchNormalization` to increase model efficiency.


### Results:

- Parameters: 7,448
- Best Training Accuracy: 99.49
- Best Test Accuracy: 99.47

### Analysis:

- We have started to see over-fitting now as train and test accuracies are diverging. 
- Model is capable of reaching 99.47 accuracy in 15 epochs but can't be pushed further.

<br>

# Model 4 - [Notebook](https://github.com/VarunSivamani/ERA-V1/blob/main/Session%207/S7_Model4.ipynb)


### Target:

- Add Regularization, `Dropout` (10%)

### Results:

- Parameters: 7,448
- Best Training Accuracy: 98.79
- Best Train Accuracy: 98.98

### Analysis:

- Regularization is working but the model is overfitting.
- But with the current capacity, not possible to push it further. 

<br>

# Model 5 - [Notebook](https://github.com/VarunSivamani/ERA-V1/blob/main/Session%207/S7_Model5.ipynb)


### Target: 

- Add Image Augmentation techniques - `RandomRotation` (5-7 degrees should be sufficient). 

### Results:

- Parameters: 7,448
- Best Training Accuracy: 98.60
- Best Test Accuracy: 99.04

### Analysis:

- Need to add an LR Scheduler to stabilise test loss.
- Image augmentation shows some improvement w.r.t. test accuracy.

<br>

# Model 6 - [Notebook](https://github.com/VarunSivamani/ERA-V1/blob/main/Session%207/S7_Model6.ipynb)


### Target:

- Add LR Scheduler `StepLR(step_size=6, gamma=0.1)`

### Results:

- Parameters: 7,416
- Best Training Accuracy: 99.93
- Best Test Accuracy: 99.45

### Analysis:

- Target achieved: crossed 99.4 validation accuracy.
- Increasing model capacity and LR rate scheduler helps meet the accuracy in 15 epochs.

<br>

![Accuracies](../Results/Session%207/Results.png)

<br>

# Receptive Field Calculations

| Layers   | Kernel | R_in | N_in | J_in | Stride | Padding | R_out | N_out | J_out |
|----------|--------|------|------|------|--------|---------|-------|-------|-------|
| input    |        |      |      |      |        |         | 1     | 28    | 1     |
| conv1    | 3      | 1    | 28   | 1    | 1      | 0       | 3     | 26    | 1     |
| conv2    | 3      | 3    | 26   | 1    | 1      | 0       | 5     | 24    | 1     |
| conv3    | 1      | 5    | 24   | 1    | 1      | 0       | 5     | 24    | 1     |
| maxpool  | 2      | 5    | 24   | 1    | 2      | 0       | 6     | 12    | 2     |
| conv4    | 3      | 6    | 12   | 2    | 1      | 0       | 10    | 10    | 2     |
| conv5    | 3      | 10   | 10   | 2    | 1      | 0       | 14    | 8     | 2     |
| conv6    | 3      | 14   | 8    | 2    | 1      | 0       | 18    | 6     | 2     |
| conv7    | 3      | 18   | 6    | 2    | 1      | 0       | 22    | 4     | 2     |
| avgpool  | 4      | 22   | 4    | 2    | 1      | 0       | 28    | 1     | 2     |
| conv8    | 1      | 28   | 1    | 2    | 1      | 0       | 28    | 1     | 2     |




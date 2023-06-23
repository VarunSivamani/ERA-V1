# Session - 8

# Batch Normalization & Regularization

<br>

# Task

1. Change the dataset to `CIFAR10`
2. Make this network:
    - C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
    - Keep the parameter count less than 50000
    - Try and add one layer to another
    - Max Epochs is 20
3. You are making 3 versions of the above code (in each case achieve above 70% accuracy):
- Network with Group Normalization
- Network with Layer Normalization
- Network with Batch Normalization

<br>

# Solution

This repository contains a model trained and validated on `CIFAR10 dataset` using various Normalization techniques `(Group Normalization, Layer Normalization, Batch Normalization)` and some models used for `MINST classification`.

## File Contents

1. `model.py` - This file contains all the models used for CIFAR10 and MNIST experiments (Session 8, Session 7, Session 6 models).

2. `utils.py` - This file contains all the necessary utility functions and methods (detecting device and setting random seed).

3. `backprop.py` - This file contains necessary train and test functions for the model.

4. `dataset.py` - This file contains data loaders (train_transforms and test_transforms).

<br>

# Group Normalization

## Results

- Total params : 47,818   
- Best Training Accuracy : 75.65 
- Best Test Accuracy : 72.90

<br>

![gn-results](../Results/Session%208/GN/gn-results.png)

<br>

## Misclassified Images

![gn-misclassified](../Results/Session%208/GN/gn-misclassified.png)

<br>

# Layer Normalization

## Results

- Total params : 47,818   
- Best Training Accuracy : 81.19  
- Best Test Accuracy : 76.89

<br>

![ln-results](../Results/Session%208/LN/ln-results.png)

<br>

## Misclassified Images

![ln-misclassified](../Results/Session%208/LN/ln-misclassified.png)

<br>

# Batch Normalization

![bn-mathematics](../Results/Session%208/BN/Batch-Normalization-mathematics.png)

## Results

- Total params : 47,818   
- Best Training Accuracy : 84.46    
- Best Test Accuracy : 79.17

<br>

![bn-results](../Results/Session%208/BN/bn-results.png)

<br>

## Misclassified Images

![bn-misclassified](../Results/Session%208/BN/bn-misclassified.png)

<br>

# Receptive Field Calculations

| Layers   | kernel | R_in | N_in | J_in | Stride | Padding | R_out | N_out | J_out |
|----------|--------|------|------|------|--------|---------|-------|-------|-------|
| input    |        |      |      |      |        |         | 1     | 32    | 1     |
| Conv1    | 3      | 1    | 32   | 1    | 1      | 0       | 3     | 30    | 1     |
| Conv2    | 3      | 3    | 30   | 1    | 1      | 0       | 5     | 28    | 1     |
| conv3    | 1      | 5    | 28   | 1    | 1      | 0       | 5     | 28    | 1     |
| maxpool  | 2      | 5    | 28   | 1    | 2      | 0       | 6     | 14    | 2     |
| Conv3a   | 3      | 6    | 14   | 2    | 1      | 1       | 10    | 14    | 2     |
| Conv4    | 3      | 10   | 14   | 2    | 1      | 1       | 14    | 14    | 2     |
| Conv5    | 3      | 14   | 14   | 2    | 1      | 1       | 18    | 14    | 2     |
| conv6    | 1      | 18   | 14   | 2    | 1      | 0       | 18    | 14    | 2     |
| maxpool  | 2      | 18   | 14   | 2    | 2      | 0       | 20    | 7     | 4     |
| Conv7    | 3      | 20   | 7    | 4    | 1      | 1       | 28    | 7     | 4     |
| Conv8    | 3      | 28   | 7    | 4    | 1      | 1       | 36    | 7     | 4     |
| Conv9    | 3      | 36   | 7    | 4    | 1      | 1       | 44    | 7     | 4     |
| GAP      | 7      | 44   | 7    | 4    | 7      | 0       | 68    | 1     | 28    |
| conv10   | 1      | 68   | 1    | 28   | 1      | 0       | 68    | 1     | 28    |

<br>

# Regularization

## L1 Regularization (Lasso Regression)

- L1 regularization adds an L1 penalty equal to the absolute value of the magnitude of coefficients. 
- When our input features have weights closer to zero this leads to a sparse L1 norm. 
- In the Sparse solution, the majority of the input features have zero weights and
very few features have non-zero weights.

<br>

![L1](../Results/Session%208/L1.png)

<br>

## L2 Regularization (Ridge Regularization)

- L2 regularization is similar to L1 regularization. 
- But it adds a squared magnitude of coefficient as a penalty term to the loss function. 
- L2 will not yield sparse models and all coefficients are shrunk by the same factor.

<br>

![L2](../Results/Session%208/L2.png)
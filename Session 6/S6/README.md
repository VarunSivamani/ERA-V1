# Session 6
<br>

# PART - 1 

# BACKPROPAGATION

Backpropagation is a widely used algorithm for training artificial neural networks, particularly in the context of deep learning. It enables neural networks to learn from labeled training data and make predictions or classify new examples.

The goal of backpropagation is to adjust the weights and biases of a neural network's connections to minimize the difference between its predicted output and the desired output. This process involves two key steps: forward propagation and backward propagation.

The process of forward propagation followed by backward propagation is iteratively repeated for multiple training examples until the network converges to a state where the loss is minimized and the predictions are accurate. Backpropagation allows neural networks to automatically learn the appropriate weights and biases through a supervised learning process.

<br>

# Neural Network

![SimpleNeuralNetwork.png](../../Results/Session%206/NeuralNetwork.png)

<br>

# Forward pass

![ForwardPass.png](../../Results/Session%206/ForwardPass.png)

<br>

# Calculating Gradients w.r.t w5

![w5.png](../../Results/Session%206/w5.png)

<br>

# Calculating Gradients in layer 2

![l2.png](../../Results/Session%206/l2.png)

<br>

# Calculating Gradients intermediate step for layer 1

![gl1.png](../../Results/Session%206/gl1.png)

<br>

# Calculating Gradients in layer 1

![l10.png](../../Results/Session%206/l10.png)

<br>

# Calculating Gradients in layer 1

![l11.png](../../Results/Session%206/l11.png)

<br>

# Weight Initialisation

![BackPropagation.png](../../Results/Session%206/Backpropagation.png)

<br>

# Major Steps

Below are the defined major steps in this exercise :     
1. **Initialization** - Weights of the neural network are initialized. (Inputs, Targets, Initial set of weights and Hidden Layer weights)
<br>

2. **Utility functions** - Sigmoid Activation function  : maps the input to a value between 0 and 1.
<br>

3. **Forward propagation** - Given the weights and inputs, this function calculates the predicted output of the network.
<br>

4. **Error Calculation** - Calculate ```0.5 * Squared Error``` between predicted output and target values.
<br>

5. **Gradient functions for each weights of the network** - These functions calculate the gradients of Error with respect to each weights in the network. This determines the direction and size of step we could take in the direction of minima. Two gradient functions are defined one for each layer. ```gradient_layer1``` function updates the weights that connect the input layer to the hidden layer. ```gradient_layer2``` function updates the weights that connect the hidden layer to output layer.
<br>

6. **Updation of weights** - We have incorporated updation of weights for each iteration in a "for loop". Each weight is updated by taking only a fraction of step size. The fraction here is defined using learning rate. Higher the learning rate greater the step we take. As a common practice learning rates are in the range between 0 to 1.
<br>

7. All the above steps are run for different learning rates in a for loop.   
<br>

# Variation of Losses w.r.t Learning Rates (Refer Excel Sheet - 2)

The screenshot shows the different error graphs for learning rates ranging from 0.1-2.0

![Losses.png](../../Results/Session%206/Losses.png)

<br>

# Error graphs

## LR = 0.1
![LR 0.1](../../Results/Session%206/LR-0.1.png)

## LR = 0.2
![LR 0.2](../../Results/Session%206/LR-0.2.png)

## LR = 0.5
![LR 0.5](../../Results/Session%206/LR-0.5.png)

## LR = 0.8
![LR 0.8](../../Results/Session%206/LR-0.8.png)

## LR = 1.0
![LR 1.0](../../Results/Session%206/LR-1.0.png)

## LR = 2.0
![LR 2.0](../../Results/Session%206/LR-2.0.png)

<br>

## Note:
- With higher learning rate, we are reaching global minima for the weights faster. 

<br>

# Part - 2

# Convolutional Neural Network for MNIST

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)


## Description :

The architecture is a deep convolutional neural network (CNN) which achieved outstanding performance on MNIST image classification. The key characteristic is its simplicity and uniformity in design, making it easy to understand and replicate.

The architecture consists of several Convolutional layers with Batch Normalisation, MaxPooling and Dropout Operations followed by a GAP Layer.

The core building block is the repeated use of 3x3 convolutional layers (kernels) stacked on top of each other.

<br>

# Model Architecture

The architecture of the `Net` neural network can be described as follows:

## 1. Convolutional Layers

`self.conv1`:

- Type: 2D Convolutional layer
    - Input channels: 1 (grayscale image)
    - Output channels: 16
    - Kernel size: 3x3
    - Padding: 1 (preserves input spatial dimensions)
- Activation: ReLU (Rectified Linear Unit)
- Normalization: Batch Normalization

`self.conv2`:

- Type: 2D Convolutional layer
    - Input channels: 16
    - Output channels: 16
    - Kernel size: 3x3
    - Padding: 1
- Activation: ReLU
- Normalization: Batch Normalization
- Pooling: Max Pooling with kernel size 2x2 and stride 2x2 (halves spatial dimensions)
- Regularization: Dropout with a rate of 0.25

`self.conv3`:

- Type: 2D Convolutional layer
    - Input channels: 16
    - Output channels: 16
    - Kernel size: 3x3
    - Padding: 1
- Activation: ReLU
- Normalization: Batch Normalization

` self.conv4`:

- Type: 2D Convolutional layer
    - Input channels: 16
    - Output channels: 16
    - Kernel size: 3x3
    - Padding: 1
- Activation: ReLU
- Normalization: Batch Normalization

`self.conv5`:

- Type: 2D Convolutional layer
    - Input channels: 16
    - Output channels: 32
    - Kernel size: 3x3
    - Padding: 1
- Activation: ReLU
- Normalization: Batch Normalization
- Pooling: Max Pooling with kernel size 2x2 and stride 2x2
- Regularization: Dropout with a rate of 0.25

<br>

## 2. Global Average Pooling Layer

`self.gap`:

- Type: 2D Convolutional layer
    - Input channels: 32
    - Output channels: 10
    - Kernel size: 1x1 (reduces spatial dimensions from 7x7 to 1x1)

<br>

## 3. Fully Connected Layers

`self.fc`:
- Type: Fully connected (linear) layer
    - Input size: 90 (flattened output from the previous layer)
    - Output size: 10 (corresponding to the number of classes)

<br>

## 4. Forward Function
The forward method defines the forward pass of the network:
- Input x is passed through the convolutional layers (conv1 to conv5) with ReLU activations and batch normalization.
- The output from conv5 is passed through the global average pooling (gap) layer to reduce spatial dimensions to 1x1.
- The output is then reshaped (view) to have dimensions `(batch_size, -1)`.
- The reshaped output is passed through the fully connected layer (fc) to obtain the final output logits.
- The logits are transformed using a logarithmic softmax function along dimension 1 (which represents the classes) and returned as the final output.

<br>

# Model Summary

```python
================================================================
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
            Conv2d-4           [-1, 16, 28, 28]           2,320
              ReLU-5           [-1, 16, 28, 28]               0
       BatchNorm2d-6           [-1, 16, 28, 28]              32
         MaxPool2d-7           [-1, 16, 14, 14]               0
           Dropout-8           [-1, 16, 14, 14]               0
            Conv2d-9           [-1, 16, 14, 14]           2,320
             ReLU-10           [-1, 16, 14, 14]               0
      BatchNorm2d-11           [-1, 16, 14, 14]              32
        MaxPool2d-12             [-1, 16, 7, 7]               0
          Dropout-13             [-1, 16, 7, 7]               0
           Conv2d-14             [-1, 16, 7, 7]           2,320
             ReLU-15             [-1, 16, 7, 7]               0
      BatchNorm2d-16             [-1, 16, 7, 7]              32
           Conv2d-17             [-1, 32, 7, 7]           4,640
             ReLU-18             [-1, 32, 7, 7]               0
      BatchNorm2d-19             [-1, 32, 7, 7]              64
        MaxPool2d-20             [-1, 32, 3, 3]               0
          Dropout-21             [-1, 32, 3, 3]               0
           Conv2d-22             [-1, 10, 3, 3]             330
           Linear-23                   [-1, 10]             910
================================================================
Total params: 13,192
Trainable params: 13,192
Non-trainable params: 0
================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.76
Params size (MB): 0.05
Estimated Total Size (MB): 0.82
================================================================
```
<br>

# Results

![Validation Accuracy](../../Results/Session%206/Validation_Accuracy.png)

<br>

# Key Achievements

- 99.4% Validation Accuracy
- Less than 20k Parameters
- Less than 20 Epochs

# Session - 5 
## Introduction to Pytorch

<br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-v3.7.1-orange)](https://matplotlib.org/stable/index.html)

<br>

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.

It is made up of a number of grayscale pictures that represent the digits 0 through 9. The collection contains square images that are each 28x28 pixels in size, for a total of 784 pixels per image.

The MNIST database contains 60,000 training images and 10,000 testing images.

<br>

# Folder Contents
<pre>
Session-5   
├───model.py   
├───utils.py   
├───S5.ipynb   
</pre>
<br>

# File Descriptions

`model.py` - contains a MNIST Net model, a convolutional neural network (CNN) architecture for categorising the MNIST handwritten digits dataset.
<br>

`utils.py` - contains utility functions that are used in the project for common tasks such as data preprocessing and loading.
<br>

`S5.ipynb` - contains a Jupyter Notebook that demonstrates how to train and test the MNIST Net model using the provided functions and utility modules.

<br>

# Usage

```
pip install python torch torchvision tqdm matplotlib
```
<br>

# How to Run
1. Install all the prerequisites.
2. Clone this repository.
3. Place all the files in the same folder.
4. Run the S5.ipynb notebook in Jupyter.

<br>

# How To Use

```python
from torchsummary import summary

model1 = Net().to(device)
summary(model1, input_size=(1, 28, 28))
```
<br>

# Model Architecture

```python
class Net(nn.Module): 

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                                     
        x = F.relu(self.conv1(x))            
        x = F.max_pool2d(self.conv2(x), 2)   
        x = F.relu(self.conv3(x))            
        x = F.max_pool2d(self.conv4(x), 2)   
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
```

<br>

# Model Summary
```python
================================================================
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
================================================================
```
<br>

## Note: 
If you are using a Jupyter Notebook or Google Colab, `summary(model, ...)` must be the returned value of the cell.
<br>
If it is not, you should wrap the summary in a print(), e.g. `print(summary(model, ...))`.

<br>

# Results (Accuracy and Loss)

![Results](../Results/Session%205/Results.png)

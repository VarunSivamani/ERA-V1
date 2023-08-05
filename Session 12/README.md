# Session 12

# PyTorch Lightning and AI Application Development

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-red)](https://lightning.ai/docs/pytorch/latest/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)
[![Torch-LR-Finder](https://img.shields.io/badge/TorchLRFinder-v0.2.1-red)](https://pypi.org/project/torch-lr-finder/)

<br>

# Task

1. Move your S10 assignment to Lightning first and then to Spaces such that:
- (You have retrained your model on Lightning)
- You are using Gradio
- Your spaces app has these features:
    - ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
    - ask whether he/she wants to view misclassified images, and how many
    - allow users to upload new images, as well as provide 10 example images
    - ask how many top classes are to be shown (make sure the user cannot enter more than 10)
- Add the full details on what your App is doing to Spaces README 

# Solution

This repository contains a `Custom Resnet18` Pytorch-Lightning model trained and validated on `CIFAR-10` dataset. The scheduler used here is `OneCycleLR`.

<br>

# Applying Albumentations library

```python
def __init__(self):
        self.albumentations_transform = A.Compose([
            A.Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615],always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.PadIfNeeded(40, 40, p=1),
            A.RandomCrop(32, 32, p=1),
            A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=0, p=1),
            A.CenterCrop(32, 32, p=1),
            ToTensorV2()
        ])
```

![Augmented Images](../Results/Session%2012/augmentation_images.png)

<br>

# Model Summary

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
           Dropout-4           [-1, 64, 32, 32]               0
         ConvLayer-5           [-1, 64, 32, 32]               0
      Custom_Layer-6           [-1, 64, 32, 32]               0
            Conv2d-7          [-1, 128, 32, 32]          73,728
         MaxPool2d-8          [-1, 128, 16, 16]               0
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
          Dropout-11          [-1, 128, 16, 16]               0
        ConvLayer-12          [-1, 128, 16, 16]               0
           Conv2d-13          [-1, 128, 16, 16]         147,456
      BatchNorm2d-14          [-1, 128, 16, 16]             256
             ReLU-15          [-1, 128, 16, 16]               0
          Dropout-16          [-1, 128, 16, 16]               0
        ConvLayer-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 128, 16, 16]         147,456
      BatchNorm2d-19          [-1, 128, 16, 16]             256
             ReLU-20          [-1, 128, 16, 16]               0
          Dropout-21          [-1, 128, 16, 16]               0
        ConvLayer-22          [-1, 128, 16, 16]               0
     Custom_Layer-23          [-1, 128, 16, 16]               0
           Conv2d-24          [-1, 256, 16, 16]         294,912
        MaxPool2d-25            [-1, 256, 8, 8]               0
      BatchNorm2d-26            [-1, 256, 8, 8]             512
             ReLU-27            [-1, 256, 8, 8]               0
          Dropout-28            [-1, 256, 8, 8]               0
        ConvLayer-29            [-1, 256, 8, 8]               0
     Custom_Layer-30            [-1, 256, 8, 8]               0
           Conv2d-31            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-32            [-1, 512, 4, 4]               0
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
             ReLU-34            [-1, 512, 4, 4]               0
          Dropout-35            [-1, 512, 4, 4]               0
        ConvLayer-36            [-1, 512, 4, 4]               0
           Conv2d-37            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
             ReLU-39            [-1, 512, 4, 4]               0
          Dropout-40            [-1, 512, 4, 4]               0
        ConvLayer-41            [-1, 512, 4, 4]               0
           Conv2d-42            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-43            [-1, 512, 4, 4]           1,024
             ReLU-44            [-1, 512, 4, 4]               0
          Dropout-45            [-1, 512, 4, 4]               0
        ConvLayer-46            [-1, 512, 4, 4]               0
     Custom_Layer-47            [-1, 512, 4, 4]               0
        MaxPool2d-48            [-1, 512, 1, 1]               0
          Flatten-49                  [-1, 512]               0
           Linear-50                   [-1, 10]           5,130
================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 10.51
Params size (MB): 25.07
Estimated Total Size (MB): 35.59
----------------------------------------------------------------
```

<br>

# Finding Optimal LR

```python
LR suggestion: steepest gradient
Suggested LR: 1.87E-04
```

![lr-finder](../Results/Session%2012/lr_finder.png)

<br>

# Results

Best Training Accuracy : `93.08`            
Best Test Accuracy : `89.75`          

![Results](../Results/Session%2012/results.png)

<br>

# Misclassified Images

<!-- ------------------------------------to do ----------------------------------- -->
![Misclassified Images](../Results/Session%2012/misclassified_images.png)

<br>

# GRADCAM Images

![Gradcam Images](../Results/Session%2012/gradcam_images.png)

<br>

# Training Testing Logs

```python
Validation: 0it [00:00, ?it/s]
Epoch: 1, Valid: Loss: 1.5148, Accuracy: 45.42
Epoch: 1, Train: Loss: 2.0496, Accuracy: 29.11


Validation: 0it [00:00, ?it/s]
Epoch: 2, Valid: Loss: 1.2262, Accuracy: 55.62
Epoch: 2, Train: Loss: 1.4039, Accuracy: 49.47


Validation: 0it [00:00, ?it/s]
Epoch: 3, Valid: Loss: 1.0850, Accuracy: 61.12
Epoch: 3, Train: Loss: 1.1747, Accuracy: 57.94


Validation: 0it [00:00, ?it/s]
Epoch: 4, Valid: Loss: 0.9161, Accuracy: 67.63
Epoch: 4, Train: Loss: 1.0254, Accuracy: 63.59


Validation: 0it [00:00, ?it/s]
Epoch: 5, Valid: Loss: 0.8217, Accuracy: 71.58
Epoch: 5, Train: Loss: 0.8923, Accuracy: 68.30


Validation: 0it [00:00, ?it/s]
Epoch: 6, Valid: Loss: 0.7778, Accuracy: 72.99
Epoch: 6, Train: Loss: 0.7821, Accuracy: 72.70


Validation: 0it [00:00, ?it/s]
Epoch: 7, Valid: Loss: 0.6234, Accuracy: 78.06
Epoch: 7, Train: Loss: 0.6979, Accuracy: 75.66


Validation: 0it [00:00, ?it/s]
Epoch: 8, Valid: Loss: 0.5796, Accuracy: 79.63
Epoch: 8, Train: Loss: 0.6358, Accuracy: 77.97


Validation: 0it [00:00, ?it/s]
Epoch: 9, Valid: Loss: 0.5434, Accuracy: 81.09
Epoch: 9, Train: Loss: 0.5811, Accuracy: 79.73


Validation: 0it [00:00, ?it/s]
Epoch: 10, Valid: Loss: 0.4912, Accuracy: 83.05
Epoch: 10, Train: Loss: 0.5415, Accuracy: 81.14


Validation: 0it [00:00, ?it/s]
Epoch: 11, Valid: Loss: 0.4758, Accuracy: 83.50
Epoch: 11, Train: Loss: 0.5044, Accuracy: 82.61


Validation: 0it [00:00, ?it/s]
Epoch: 12, Valid: Loss: 0.4388, Accuracy: 85.00
Epoch: 12, Train: Loss: 0.4786, Accuracy: 83.41


Validation: 0it [00:00, ?it/s]
Epoch: 13, Valid: Loss: 0.4522, Accuracy: 84.48
Epoch: 13, Train: Loss: 0.4484, Accuracy: 84.54


Validation: 0it [00:00, ?it/s]
Epoch: 14, Valid: Loss: 0.4242, Accuracy: 85.51
Epoch: 14, Train: Loss: 0.4193, Accuracy: 85.70


Validation: 0it [00:00, ?it/s]
Epoch: 15, Valid: Loss: 0.3981, Accuracy: 86.30
Epoch: 15, Train: Loss: 0.3950, Accuracy: 86.40


Validation: 0it [00:00, ?it/s]
Epoch: 16, Valid: Loss: 0.4390, Accuracy: 84.90
Epoch: 16, Train: Loss: 0.3722, Accuracy: 87.31


Validation: 0it [00:00, ?it/s]
Epoch: 17, Valid: Loss: 0.3800, Accuracy: 86.92
Epoch: 17, Train: Loss: 0.3456, Accuracy: 88.12


Validation: 0it [00:00, ?it/s]
Epoch: 18, Valid: Loss: 0.3687, Accuracy: 87.45
Epoch: 18, Train: Loss: 0.3302, Accuracy: 88.68


Validation: 0it [00:00, ?it/s]
Epoch: 19, Valid: Loss: 0.3524, Accuracy: 87.85
Epoch: 19, Train: Loss: 0.3067, Accuracy: 89.64


Validation: 0it [00:00, ?it/s]
Epoch: 20, Valid: Loss: 0.3527, Accuracy: 87.77
Epoch: 20, Train: Loss: 0.2858, Accuracy: 90.34


Validation: 0it [00:00, ?it/s]
Epoch: 21, Valid: Loss: 0.3306, Accuracy: 88.69
Epoch: 21, Train: Loss: 0.2646, Accuracy: 91.16


Validation: 0it [00:00, ?it/s]
Epoch: 22, Valid: Loss: 0.3128, Accuracy: 89.22
Epoch: 22, Train: Loss: 0.2438, Accuracy: 91.94


Validation: 0it [00:00, ?it/s]
Epoch: 23, Valid: Loss: 0.3123, Accuracy: 89.46
Epoch: 23, Train: Loss: 0.2246, Accuracy: 92.66


Validation: 0it [00:00, ?it/s]
Epoch: 24, Valid: Loss: 0.2999, Accuracy: 89.75
Epoch: 24, Train: Loss: 0.2133, Accuracy: 93.08
```

<br>
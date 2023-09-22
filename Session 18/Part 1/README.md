# PART 1 - UNET

# Task

1. First part of your assignment is to train your own UNet from scratch, you can use the dataset and strategy provided. However, you need to train it 4 times:

- MP+Tr+BCE
- MP+Tr+Dice Loss
- StrConv+Tr+BCE
- StrConv+Ups+Dice Loss

and report your results.

<br>

# Model Summary 

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 128, 128]           1,792
       BatchNorm2d-2         [-1, 64, 128, 128]             128
              ReLU-3         [-1, 64, 128, 128]               0
            Conv2d-4         [-1, 64, 128, 128]          36,928
       BatchNorm2d-5         [-1, 64, 128, 128]             128
              ReLU-6         [-1, 64, 128, 128]               0
         MaxPool2d-7           [-1, 64, 64, 64]               0
  ContractingBlock-8  [[-1, 64, 64, 64], [-1, 64, 128, 128]]               0
            Conv2d-9          [-1, 128, 64, 64]          73,856
      BatchNorm2d-10          [-1, 128, 64, 64]             256
             ReLU-11          [-1, 128, 64, 64]               0
           Conv2d-12          [-1, 128, 64, 64]         147,584
      BatchNorm2d-13          [-1, 128, 64, 64]             256
             ReLU-14          [-1, 128, 64, 64]               0
        MaxPool2d-15          [-1, 128, 32, 32]               0
 ContractingBlock-16  [[-1, 128, 32, 32], [-1, 128, 64, 64]]               0
           Conv2d-17          [-1, 256, 32, 32]         295,168
      BatchNorm2d-18          [-1, 256, 32, 32]             512
             ReLU-19          [-1, 256, 32, 32]               0
           Conv2d-20          [-1, 256, 32, 32]         590,080
      BatchNorm2d-21          [-1, 256, 32, 32]             512
             ReLU-22          [-1, 256, 32, 32]               0
        MaxPool2d-23          [-1, 256, 16, 16]               0
 ContractingBlock-24  [[-1, 256, 16, 16], [-1, 256, 32, 32]]               0
           Conv2d-25          [-1, 512, 16, 16]       1,180,160
      BatchNorm2d-26          [-1, 512, 16, 16]           1,024
             ReLU-27          [-1, 512, 16, 16]               0
           Conv2d-28          [-1, 512, 16, 16]       2,359,808
      BatchNorm2d-29          [-1, 512, 16, 16]           1,024
             ReLU-30          [-1, 512, 16, 16]               0
        MaxPool2d-31            [-1, 512, 8, 8]               0
 ContractingBlock-32  [[-1, 512, 8, 8], [-1, 512, 16, 16]]               0
  ConvTranspose2d-33          [-1, 256, 32, 32]         524,544
           Conv2d-34          [-1, 256, 32, 32]       1,179,904
      BatchNorm2d-35          [-1, 256, 32, 32]             512
             ReLU-36          [-1, 256, 32, 32]               0
           Conv2d-37          [-1, 256, 32, 32]         590,080
      BatchNorm2d-38          [-1, 256, 32, 32]             512
             ReLU-39          [-1, 256, 32, 32]               0
   ExpandingBlock-40          [-1, 256, 32, 32]               0
  ConvTranspose2d-41          [-1, 128, 64, 64]         131,200
           Conv2d-42          [-1, 128, 64, 64]         295,040
      BatchNorm2d-43          [-1, 128, 64, 64]             256
             ReLU-44          [-1, 128, 64, 64]               0
           Conv2d-45          [-1, 128, 64, 64]         147,584
      BatchNorm2d-46          [-1, 128, 64, 64]             256
             ReLU-47          [-1, 128, 64, 64]               0
   ExpandingBlock-48          [-1, 128, 64, 64]               0
  ConvTranspose2d-49         [-1, 64, 128, 128]          32,832
           Conv2d-50         [-1, 64, 128, 128]          73,792
      BatchNorm2d-51         [-1, 64, 128, 128]             128
             ReLU-52         [-1, 64, 128, 128]               0
           Conv2d-53         [-1, 64, 128, 128]          36,928
      BatchNorm2d-54         [-1, 64, 128, 128]             128
             ReLU-55         [-1, 64, 128, 128]               0
   ExpandingBlock-56         [-1, 64, 128, 128]               0
           Conv2d-57          [-1, 3, 128, 128]             195
================================================================
Total params: 7,703,107
Trainable params: 7,703,107
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 2785073.88
Params size (MB): 29.39
Estimated Total Size (MB): 2785103.45
----------------------------------------------------------------
```

# Dataset

- The Oxford-IIIT Pet Dataset is a 37 category pet dataset with roughly 200 images for each class created by the Visual Geometry Group at Oxford. - The images have a large variations in scale, pose and lighting. 
- All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation.

## Dataset links

- https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset
- https://www.robots.ox.ac.uk/~vgg/data/pets/


## Visualise dataset

![OxfordIIITPet](../../Results/Session%2018/oxford_data.png)
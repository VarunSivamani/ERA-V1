# Session 11

# CAMs, LRs and Optimizers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)
[![Torch-LR-Finder](https://img.shields.io/badge/TorchLRFinder-v0.2.1-red)](https://pypi.org/project/torch-lr-finder/)

<br>

# Task:

1. You are going to follow the same structure for your Code (as a reference). So Create:
- models folder - this is where you'll add all of your future models. Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class
- main.py - from Google Colab, now onwards, this is the file that you'll import (along with the model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):
    - training and test loops
    - data split between test and train
    - epochs
    - batch size
    - which optimizer to run
    - do we run a scheduler?
- utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:
    1. image transforms,
    2. gradcam,
    3. misclassification code,
    4. tensorboard related stuff
    5. advanced training policies, etc
    6. etc
2. Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
    - pull your Github code to google colab (don't copy-paste code)
    - prove that you are following the above structure
    - that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
3. your colab file must:
    - train resnet18 for 20 epochs on the CIFAR10 dataset
    - show loss curves for test and train datasets
    - show a gallery of 10 misclassified images
    - show gradcam output on 10 misclassified images. 
4. Apply these transforms while training:
    - RandomCrop(32, padding=4)
    - CutOut(16x16)

<br>

# Solution

This repository contains a `Resnet18` Pytorch model trained and validated on `CIFAR-10` dataset. The scheduler used here is `OneCycleLR`.

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

![Augmented Images](../Results/Session%2011/augmentation_images.png)

<br>

# Model Summary

```python
============================================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Param %
============================================================================================================================================
ResNet                                   [32, 3, 32, 32]           [32, 10]                  --                             --
├─Conv2d: 1-1                            [32, 3, 32, 32]           [32, 64, 32, 32]          1,728                       0.02%
├─BatchNorm2d: 1-2                       [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
├─Sequential: 1-3                        [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
│    └─BasicBlock: 2-1                   [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
│    │    └─Conv2d: 3-1                  [32, 64, 32, 32]          [32, 64, 32, 32]          36,864                      0.33%
│    │    └─BatchNorm2d: 3-2             [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
│    │    └─Conv2d: 3-3                  [32, 64, 32, 32]          [32, 64, 32, 32]          36,864                      0.33%
│    │    └─BatchNorm2d: 3-4             [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
│    │    └─Sequential: 3-5              [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
│    └─BasicBlock: 2-2                   [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
│    │    └─Conv2d: 3-6                  [32, 64, 32, 32]          [32, 64, 32, 32]          36,864                      0.33%
│    │    └─BatchNorm2d: 3-7             [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
│    │    └─Conv2d: 3-8                  [32, 64, 32, 32]          [32, 64, 32, 32]          36,864                      0.33%
│    │    └─BatchNorm2d: 3-9             [32, 64, 32, 32]          [32, 64, 32, 32]          128                         0.00%
│    │    └─Sequential: 3-10             [32, 64, 32, 32]          [32, 64, 32, 32]          --                             --
├─Sequential: 1-4                        [32, 64, 32, 32]          [32, 128, 16, 16]         --                             --
│    └─BasicBlock: 2-3                   [32, 64, 32, 32]          [32, 128, 16, 16]         --                             --
│    │    └─Conv2d: 3-11                 [32, 64, 32, 32]          [32, 128, 16, 16]         73,728                      0.66%
│    │    └─BatchNorm2d: 3-12            [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    │    └─Conv2d: 3-13                 [32, 128, 16, 16]         [32, 128, 16, 16]         147,456                     1.32%
│    │    └─BatchNorm2d: 3-14            [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    │    └─Sequential: 3-15             [32, 64, 32, 32]          [32, 128, 16, 16]         --                             --
│    │    │    └─Conv2d: 4-1             [32, 64, 32, 32]          [32, 128, 16, 16]         8,192                       0.07%
│    │    │    └─BatchNorm2d: 4-2        [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    └─BasicBlock: 2-4                   [32, 128, 16, 16]         [32, 128, 16, 16]         --                             --
│    │    └─Conv2d: 3-16                 [32, 128, 16, 16]         [32, 128, 16, 16]         147,456                     1.32%
│    │    └─BatchNorm2d: 3-17            [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    │    └─Conv2d: 3-18                 [32, 128, 16, 16]         [32, 128, 16, 16]         147,456                     1.32%
│    │    └─BatchNorm2d: 3-19            [32, 128, 16, 16]         [32, 128, 16, 16]         256                         0.00%
│    │    └─Sequential: 3-20             [32, 128, 16, 16]         [32, 128, 16, 16]         --                             --
├─Sequential: 1-5                        [32, 128, 16, 16]         [32, 256, 8, 8]           --                             --
│    └─BasicBlock: 2-5                   [32, 128, 16, 16]         [32, 256, 8, 8]           --                             --
│    │    └─Conv2d: 3-21                 [32, 128, 16, 16]         [32, 256, 8, 8]           294,912                     2.64%
│    │    └─BatchNorm2d: 3-22            [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    │    └─Conv2d: 3-23                 [32, 256, 8, 8]           [32, 256, 8, 8]           589,824                     5.28%
│    │    └─BatchNorm2d: 3-24            [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    │    └─Sequential: 3-25             [32, 128, 16, 16]         [32, 256, 8, 8]           --                             --
│    │    │    └─Conv2d: 4-3             [32, 128, 16, 16]         [32, 256, 8, 8]           32,768                      0.29%
│    │    │    └─BatchNorm2d: 4-4        [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    └─BasicBlock: 2-6                   [32, 256, 8, 8]           [32, 256, 8, 8]           --                             --
│    │    └─Conv2d: 3-26                 [32, 256, 8, 8]           [32, 256, 8, 8]           589,824                     5.28%
│    │    └─BatchNorm2d: 3-27            [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    │    └─Conv2d: 3-28                 [32, 256, 8, 8]           [32, 256, 8, 8]           589,824                     5.28%
│    │    └─BatchNorm2d: 3-29            [32, 256, 8, 8]           [32, 256, 8, 8]           512                         0.00%
│    │    └─Sequential: 3-30             [32, 256, 8, 8]           [32, 256, 8, 8]           --                             --
├─Sequential: 1-6                        [32, 256, 8, 8]           [32, 512, 4, 4]           --                             --
│    └─BasicBlock: 2-7                   [32, 256, 8, 8]           [32, 512, 4, 4]           --                             --
│    │    └─Conv2d: 3-31                 [32, 256, 8, 8]           [32, 512, 4, 4]           1,179,648                  10.56%
│    │    └─BatchNorm2d: 3-32            [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    │    └─Conv2d: 3-33                 [32, 512, 4, 4]           [32, 512, 4, 4]           2,359,296                  21.11%
│    │    └─BatchNorm2d: 3-34            [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    │    └─Sequential: 3-35             [32, 256, 8, 8]           [32, 512, 4, 4]           --                             --
│    │    │    └─Conv2d: 4-5             [32, 256, 8, 8]           [32, 512, 4, 4]           131,072                     1.17%
│    │    │    └─BatchNorm2d: 4-6        [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    └─BasicBlock: 2-8                   [32, 512, 4, 4]           [32, 512, 4, 4]           --                             --
│    │    └─Conv2d: 3-36                 [32, 512, 4, 4]           [32, 512, 4, 4]           2,359,296                  21.11%
│    │    └─BatchNorm2d: 3-37            [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    │    └─Conv2d: 3-38                 [32, 512, 4, 4]           [32, 512, 4, 4]           2,359,296                  21.11%
│    │    └─BatchNorm2d: 3-39            [32, 512, 4, 4]           [32, 512, 4, 4]           1,024                       0.01%
│    │    └─Sequential: 3-40             [32, 512, 4, 4]           [32, 512, 4, 4]           --                             --
├─Linear: 1-7                            [32, 512]                 [32, 10]                  5,130                       0.05%
============================================================================================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
Total mult-adds (G): 17.77
============================================================================================================================================
Input size (MB): 0.39
Forward/backward pass size (MB): 314.58
Params size (MB): 44.70
Estimated Total Size (MB): 359.66
============================================================================================================================================
```

<br>

# Finding Optimal LR

```python
LR suggestion: steepest gradient
Suggested LR: 5.09E-03
```

![lr-finder](../Results/Session%2011/lr_finder.png)

<br>

# Results

Best Training Accuracy : `93.38`        
Best Test Accuracy : `92.22`      

![Train Results](../Results/Session%2011/Train_Results.png)
![Test Results](../Results/Session%2011/Test_Results.png)

<br>

# Misclassified Images

![Misclassified Images](../Results/Session%2011/misclassified_images.png)

<br>

# GRADCAM Images

![Gradcam Images](../Results/Session%2011/gradcam_images.png)

<br>

# Accuracy of each class

```python
Accuracy for class: plane is 93.8 %
Accuracy for class: car   is 96.7 %
Accuracy for class: bird  is 88.9 %
Accuracy for class: cat   is 81.5 %
Accuracy for class: deer  is 93.7 %
Accuracy for class: dog   is 86.6 %
Accuracy for class: frog  is 95.1 %
Accuracy for class: horse is 94.3 %
Accuracy for class: ship  is 96.1 %
Accuracy for class: truck is 95.5 %
```

<br>

# Training Testing Logs

```python
Epoch: 1
Train: Average Loss: 1.61, Accuracy: 41.19 LR: 0.001059064382880664: 100%|██████████| 1563/1563 [00:59<00:00, 26.10it/s]
Test set: Average loss: 1.21, Accuracy: 5569/10000 (55.69%)

Epoch: 2
Train: Average Loss: 1.14, Accuracy: 59.35 LR: 0.002067832790864593: 100%|██████████| 1563/1563 [00:59<00:00, 26.18it/s]
Test set: Average loss: 1.02, Accuracy: 6447/10000 (64.47%)

Epoch: 3
Train: Average Loss: 0.92, Accuracy: 67.62 LR: 0.0030766011988485227: 100%|██████████| 1563/1563 [00:59<00:00, 26.06it/s]
Test set: Average loss: 0.73, Accuracy: 7502/10000 (75.02%)

Epoch: 4
Train: Average Loss: 0.78, Accuracy: 72.93 LR: 0.004085369606832451: 100%|██████████| 1563/1563 [01:00<00:00, 25.73it/s]
Test set: Average loss: 0.77, Accuracy: 7430/10000 (74.30%)

Epoch: 5
Train: Average Loss: 0.70, Accuracy: 75.86 LR: 0.0050941380148163806: 100%|██████████| 1563/1563 [01:00<00:00, 25.71it/s]
Test set: Average loss: 0.55, Accuracy: 8123/10000 (81.23%)

Epoch: 6
Train: Average Loss: 0.62, Accuracy: 78.72 LR: 0.00475456277474872: 100%|██████████| 1563/1563 [01:02<00:00, 25.14it/s]
Test set: Average loss: 0.54, Accuracy: 8233/10000 (82.33%)

Epoch: 7
Train: Average Loss: 0.56, Accuracy: 80.68 LR: 0.00441498753468106: 100%|██████████| 1563/1563 [01:01<00:00, 25.27it/s]
Test set: Average loss: 0.45, Accuracy: 8433/10000 (84.33%)

Epoch: 8
Train: Average Loss: 0.51, Accuracy: 82.25 LR: 0.004075412294613401: 100%|██████████| 1563/1563 [01:00<00:00, 25.65it/s]
Test set: Average loss: 0.43, Accuracy: 8490/10000 (84.90%)

Epoch: 9
Train: Average Loss: 0.47, Accuracy: 83.81 LR: 0.003735837054545741: 100%|██████████| 1563/1563 [01:00<00:00, 25.72it/s]
Test set: Average loss: 0.43, Accuracy: 8558/10000 (85.58%)

Epoch: 10
Train: Average Loss: 0.43, Accuracy: 85.02 LR: 0.003396261814478081: 100%|██████████| 1563/1563 [01:00<00:00, 25.81it/s]
Test set: Average loss: 0.37, Accuracy: 8751/10000 (87.51%)

Epoch: 11
Train: Average Loss: 0.41, Accuracy: 85.78 LR: 0.003056686574410421: 100%|██████████| 1563/1563 [01:00<00:00, 26.03it/s]
Test set: Average loss: 0.35, Accuracy: 8772/10000 (87.72%)

Epoch: 12
Train: Average Loss: 0.38, Accuracy: 86.65 LR: 0.0027171113343427613: 100%|██████████| 1563/1563 [00:59<00:00, 26.29it/s]
Test set: Average loss: 0.33, Accuracy: 8873/10000 (88.73%)

Epoch: 13
Train: Average Loss: 0.36, Accuracy: 87.69 LR: 0.002377536094275101: 100%|██████████| 1563/1563 [00:59<00:00, 26.24it/s]
Test set: Average loss: 0.31, Accuracy: 8906/10000 (89.06%)

Epoch: 14
Train: Average Loss: 0.33, Accuracy: 88.53 LR: 0.0020379608542074414: 100%|██████████| 1563/1563 [00:59<00:00, 26.14it/s]
Test set: Average loss: 0.36, Accuracy: 8782/10000 (87.82%)

Epoch: 15
Train: Average Loss: 0.31, Accuracy: 89.08 LR: 0.0016983856141397817: 100%|██████████| 1563/1563 [01:00<00:00, 25.99it/s]
Test set: Average loss: 0.29, Accuracy: 9004/10000 (90.04%)

Epoch: 16
Train: Average Loss: 0.29, Accuracy: 89.84 LR: 0.0013588103740721216: 100%|██████████| 1563/1563 [00:59<00:00, 26.08it/s]
Test set: Average loss: 0.28, Accuracy: 9049/10000 (90.49%)

Epoch: 17
Train: Average Loss: 0.27, Accuracy: 90.70 LR: 0.0010192351340044615: 100%|██████████| 1563/1563 [01:00<00:00, 25.95it/s]
Test set: Average loss: 0.27, Accuracy: 9119/10000 (91.19%)

Epoch: 18
Train: Average Loss: 0.24, Accuracy: 91.47 LR: 0.0006796598939368013: 100%|██████████| 1563/1563 [01:01<00:00, 25.56it/s]
Test set: Average loss: 0.25, Accuracy: 9194/10000 (91.94%)

Epoch: 19
Train: Average Loss: 0.22, Accuracy: 92.39 LR: 0.00034008465386914204: 100%|██████████| 1563/1563 [01:01<00:00, 25.32it/s]
Test set: Average loss: 0.24, Accuracy: 9200/10000 (92.00%)

Epoch: 20
Train: Average Loss: 0.19, Accuracy: 93.38 LR: 5.09413801481895e-07: 100%|██████████| 1563/1563 [01:01<00:00, 25.33it/s]
Test set: Average loss: 0.23, Accuracy: 9222/10000 (92.22%)
```


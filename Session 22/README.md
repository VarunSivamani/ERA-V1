# Session 22

# Training a Transformer from Scratch!

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Transformers](https://img.shields.io/badge/transformers-v4.34.0-lightgreen)](https://huggingface.co/docs/transformers/index)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-red)](https://lightning.ai/docs/pytorch/latest/)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)

<br>

# Task

All the code is made available. You'd need to add data to the data folder.
1. Set up the complete code on Amazon Sagemaker and train the model.
2. Train the model such that it reaches 3.49 (basically less than 3.5) on the loss.

<br>

# Solution

This repository contains a Jupyter notebook that consists of GPT model trained on AWS SageMaker.

<br>

# Pre-requisites

- huggingface_hub
- tokenizers
- sentencepiece
- lightning
- jsonargparse

<br>

# Dataset

## RedPajama-1T-Sample dataset

- [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/blob/main/README.md) is a clean-room, fully open-source implementation of the LLaMa dataset. 
- This HuggingFace repo contains a 1B-token sample of the RedPajama dataset. 

<br>

## Dataset Structure

```python
{
    "text": ...,
    "meta": {"url": "...", "timestamp": "...", "source": "...", "language": "...", ...}
}
```

<br>

## Data proportions

```python
# Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
data_config = [
    ("arxiv", 2.5),
    ("book", 4.5),
    ("c4", 15.0),
    ("cc", 67.0),
    ("github", 4.5),
    ("stackexchange", 2.0),
    ("wikipedia", 4.5),
]
```

<br>




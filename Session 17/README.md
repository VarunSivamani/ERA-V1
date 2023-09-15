# Session 17 

# BERT, GPT & ViT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![Transformers 4.33.1](https://img.shields.io/badge/transformers-v4.33.1-red)](https://huggingface.co/docs/transformers/index)

<br>

# Task

1. Re-write the code in such a way where there is **one transformer.py file** that you can use to train all 3 models.

<br>

# Solution

This repository contains `3 ipynb files` used for training `BERT, GPT and ViT models`.

<br>

# Transformer

![Transformer-Encoder-Decoder](../Results/Session%2017/transformer-encoder-decoder.png)

<br>

# BERT

- **Bidirectional Encoder Representations** from Transformers, is a State-Of-The-Art Natural Language Processing (NLP) model that was introduced by **Google AI researchers** in **2018**.   
- BERT is basically just the encoder part of the Transformer.
- In BERT we used the MASK token to predict the missing work.    

![BERT](../Results/Session%2017/bert.png)

<br>

# GPT

- **Generative Pre-trained Transformer** is a family of State-Of-The-Art Natural Language Processing (NLP) models developed by **OpenAI**.    
- GPT is just the decoder part of the network. GPT will output one token at a time.   
- In GPT, we will MASK "all" future words. So our attention would be "masked" attention.   
- GPT models are based on the Transformer architecture and are known for their ability to generate coherent and contextually relevant text.   

![GPT](../Results/Session%2017/gpt.png)

<br>

# ViT

- The **Vision Transformer** is a model for image classification that employs a Transformer-like architecture over patches of the image.     
- An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder.    
- In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.   

<br>

Overall architecture can be described easily in five simple steps:

1. Split an input image into patches
2. Get linear embeddings (representations) from each patch referred to as Patch Embeddings
3. Add positional embeddings and a [CLS] token to each of the Patch Embeddings
there is more to the CLS token that we would cover today. Would request you to consider the definition of CLS token as shared in the last class as wrong, as we need to further decode it.
4. Pass through a Transformer Encoder and get the output values for each of the [CLS] tokens.
5. Pass the representations of [CLS] tokens through an MLP Head to get final class predictions.

![ViT](../Results/Session%2017/vit.png)

<br>

## ViT Model Summary

```python
============================================================================================================================================
Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
============================================================================================================================================
Transformer (Transformer)                                    [32, 3, 224, 224]    [32, 3]              152,064              True
├─PatchEmbedding (patch_embedding)                           [32, 3, 224, 224]    [32, 196, 768]       --                   True
│    └─Conv2d (patcher)                                      [32, 3, 224, 224]    [32, 768, 14, 14]    590,592              True
│    └─Flatten (flatten)                                     [32, 768, 14, 14]    [32, 768, 196]       --                   --
├─Dropout (embedding_dropout)                                [32, 197, 768]       [32, 197, 768]       --                   --
├─Sequential (transformer_encoder)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    └─TransformerEncoderBlock (0)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (1)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (2)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (3)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (4)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (5)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (6)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (7)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (8)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (9)                           [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (10)                          [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
│    └─TransformerEncoderBlock (11)                          [32, 197, 768]       [32, 197, 768]       --                   True
│    │    └─MultiHeadAttention (msa_block)                   [32, 197, 768]       [32, 197, 768]       2,363,904            True
│    │    └─MLPBlock (mlp_block)                             [32, 197, 768]       [32, 197, 768]       4,723,968            True
├─Sequential (classifier)                                    [32, 768]            [32, 3]              --                   True
│    └─LayerNorm (0)                                         [32, 768]            [32, 768]            1,536                True
│    └─Linear (1)                                            [32, 768]            [32, 3]              2,307                True
============================================================================================================================================
Total params: 85,800,963
Trainable params: 85,800,963
Non-trainable params: 0
Total mult-adds (G): 5.52
============================================================================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 3292.20
Params size (MB): 229.20
Estimated Total Size (MB): 3540.67
============================================================================================================================================
```

<br>

# Training Logs

## BERT

```python
it: 9900  | loss 4.2  | Δw: 11.062
it: 9910  | loss 4.27  | Δw: 11.246
it: 9920  | loss 4.16  | Δw: 11.302
it: 9930  | loss 4.23  | Δw: 11.333
it: 9940  | loss 4.24  | Δw: 10.71
it: 9950  | loss 4.28  | Δw: 11.05
it: 9960  | loss 4.33  | Δw: 11.705
it: 9970  | loss 4.29  | Δw: 11.032
it: 9980  | loss 4.28  | Δw: 11.107
it: 9990  | loss 4.1  | Δw: 11.15
```

<br>

## GPT

```python
step          0 | train loss 10.7297 | val loss 10.7062
step        500 | train loss 0.4823 | val loss 8.2294
step       1000 | train loss 0.1586 | val loss 9.8948
step       1500 | train loss 0.1447 | val loss 10.0364
step       2000 | train loss 0.1243 | val loss 10.3339
step       2500 | train loss 0.1183 | val loss 10.7762
step       3000 | train loss 0.1150 | val loss 11.0317
step       3500 | train loss 0.1170 | val loss 10.9939
step       4000 | train loss 0.1065 | val loss 11.4041
step       4500 | train loss 0.1090 | val loss 11.0132
step       4999 | train loss 0.1033 | val loss 11.3228
```

<br>

## ViT

```python
Epoch: 1 | train_loss: 4.0213 | train_acc: 0.2695 | test_loss: 1.8297 | test_acc: 0.5417
Epoch: 2 | train_loss: 1.6439 | train_acc: 0.3906 | test_loss: 2.1778 | test_acc: 0.2604
Epoch: 3 | train_loss: 1.2724 | train_acc: 0.3242 | test_loss: 1.0046 | test_acc: 0.5417
Epoch: 4 | train_loss: 1.1596 | train_acc: 0.3047 | test_loss: 1.0133 | test_acc: 0.5417
Epoch: 5 | train_loss: 1.1104 | train_acc: 0.4297 | test_loss: 1.4873 | test_acc: 0.2604
Epoch: 6 | train_loss: 1.2359 | train_acc: 0.3086 | test_loss: 1.0748 | test_acc: 0.5417
Epoch: 7 | train_loss: 1.1101 | train_acc: 0.3008 | test_loss: 1.3549 | test_acc: 0.1979
Epoch: 8 | train_loss: 1.1826 | train_acc: 0.3867 | test_loss: 1.0088 | test_acc: 0.5417
Epoch: 9 | train_loss: 1.1993 | train_acc: 0.2773 | test_loss: 1.3577 | test_acc: 0.1979
Epoch: 10 | train_loss: 1.2045 | train_acc: 0.2930 | test_loss: 1.1185 | test_acc: 0.2604
```

<br>

# Results

## ViT

![ViT results](../Results/Session%2017/vit_graph.png)
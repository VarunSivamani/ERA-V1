# Session 15

# Dawn of Transformers - Part II Part A & Part B

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-red)](https://lightning.ai/docs/pytorch/latest/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)

<br>

# Task

- Rewrite the whole code covered in the class in Pytorch-Lightning.
- Train the model for 10 epochs.
- Achieve a loss of less than 4.

<br>

# Objective 

- Understand the internal structure of Transformers.

<br>

# Solution

This repository contains a `Pytorch-Lightning Transformer model` trained and validated using `Opus Dataset for 10 epochs`.

<br>

# Dataset

The dataset used for training and validation is [`opus_books`](https://huggingface.co/datasets/opus_books).

### Dataset Summary

- OPUS-100 is English-centric, meaning that all training pairs include English on either the source or target side. The corpus covers 100 languages (including English). - Selected the languages based on the volume of parallel data available in OPUS.

### Languages
- OPUS-100 contains approximately 55M sentence pairs. Of the 99 language pairs, 44 have 1M sentence pairs of training data, 73 have at least 100k, and 95 have at least 10k.

<br>

# Model Summary 

```python
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | Encoder            | 12.6 M
1 | decoder          | Decoder            | 18.9 M
2 | projection_layer | ProjectionLayer    | 11.5 M
3 | src_embed        | InputEmbeddings    | 8.0 M 
4 | tgt_embed        | InputEmbeddings    | 11.5 M
5 | src_pos          | PositionalEncoding | 0     
6 | tgt_pos          | PositionalEncoding | 0     
7 | loss_fn          | CrossEntropyLoss   | 0     
--------------------------------------------------------
62.5 M    Trainable params
0         Non-trainable params
62.5 M    Total params
250.151   Total estimated model params size (MB)
```

<br>

# Metrics Used

## CharErrorRate

- Character Error Rate (CER) is a metric of the performance of an automatic speech recognition (ASR) system.
- This value indicates the percentage of characters that were incorrectly predicted. 
- The lower the value, the better the performance of the ASR system with a CharErrorRate of 0 being a perfect score.

<br>

## WordErrorRate

- Word error rate (WordErrorRate) is a common metric of the performance of an automatic speech recognition.
- This value indicates the percentage of words that were incorrectly predicted. 
- The lower the value, the better the performance of the ASR system with a WER of 0 being a perfect score.

<br>

## BLEUSCORE

- BLEU (Bilingual Evaluation Understudy) is a commonly used metric for evaluating the quality of machine-generated text, such as machine translation or text generation tasks.
- It measures the similarity between a machine-generated text and one or more reference texts (human-generated or gold-standard translations). 
- The BLEU score is a number between 0 and 1, where a higher score indicates better similarity between the generated text and the reference text(s).

<br>

# Architecture

![architecture](../Results/Session%2015/Transformer%20Architecture.png)

<br>

# Architecture Understanding
[Reference](https://machinelearningmastery.com/the-transformer-model/)

The Transformer model runs as follows : 

1. Each word forming an input sequence is transformed into a d<sub>model</sub>-dimensional embedding vector. 
2. Each embedding vector representing an input word is augmented by summing it (element-wise) to a positional encoding vector of the same d<sub>model</sub> length, hence introducing positional information into the input. 
3. The augmented embedding vectors are fed into the encoder block consisting of the two sublayers. Since the encoder attends to all words in the input sequence, irrespective if they precede or succeed the word under consideration, then the `Transformer encoder is bidirectional`. 
4. The decoder receives as input its own predicted output word at time-step, `t-1`.
5. The input to the decoder is also augmented by positional encoding in the same manner done on the encoder side. 
6. The augmented decoder input is fed into the three sublayers comprising the decoder block. Masking is applied in the first sublayer in order to stop the decoder from attending to the succeeding words. At the second sublayer, the decoder also receives the output of the encoder, which now allows the decoder to attend to all the words in the input sequence.
7. The output of the decoder finally passes through a fully connected layer, followed by a `softmax` layer, to generate a prediction for the next word of the output sequence. 

<br>

![architecture-understanding](../Results/Session%2015/Architecture.png)

<br>

# Self Attention

- The `query` is the representation for the word we want to calculate self-attention for.
- The `key` is a representation of each word in the sequence and is used to match against the query of the word for which we currently want to calculate self-attention.
- The `value` is the actual representation of each word in a sequence, the representation we really care about. 

<br>

![self-attention](../Results/Session%2015/Attention.png)

<br>

# Inference

![inference](../Results/Session%2015/Inference.png)

<br>

# Training - Validation Logs

The training - valiation logs for the model can be found [here](Training_Validation_Logs.md)








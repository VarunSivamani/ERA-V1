# Session 16

# Transformer Architectures & Speeding them up!

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch-Lightning](https://img.shields.io/badge/pytorch_lightning-v2.0.6-red)](https://lightning.ai/docs/pytorch/latest/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)

<br>

# Task

1. Pick the "en-fr" dataset from opus_books
2. Remove all English sentences with more than 150 "tokens"
3. Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10
4. Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8
5. Enjoy! 

<br>

# Solution

This repository contains a `Pytorch-Lightning Transformer model` trained and validated using `Opus Dataset for 25 epochs`.

<br>

# Whats New ?

### 1. Changed target-language 
- The target-language was changed from `Italian` to `French`, keeping the source-language as `English`.

### 2. Preprocessed the dataset
- Filtered all sentences whose length were `greater than 150`.
- Also filtered sentences where, `len(fench_sentences) > len(english_sentence) + 10`.

### 3. Implemented OCP
- OneCyclePolicy was implemented to allow the model to be trained on higher learning rates and converge faster.

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["lr"],
            epochs=self.trainer.max_epochs,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=10,
            three_phase=True,
            anneal_strategy='linear',
            verbose=False
        )
```
<br>

### 4. Parameter sharing
- helps to reduce the number of parameters in the model.
- shares encoders and decoders in the architecture.

### 5. Sequence length changed
- sequence_length of the model was changed to `160` due to pre-processing of sentences.

### 6. Dynamic Padding
- Dynamic Padding helps to implement padding of sentences in a batch dynamically according to batches and increases training time effectively.

<br>

# Dataset

The dataset used for training and validation is `en-fr` of [opus_books](https://huggingface.co/datasets/opus_books/viewer/en-fr/train). 

- **Source Language** - `en`   
- **Target Language** - `fr`

### Dataset Summary

- OPUS-100 is English-centric, meaning that all training pairs include English on either the source or target side. The corpus covers 100 languages (including English). - Selected the languages based on the volume of parallel data available in OPUS.

### Languages
- OPUS-100 contains approximately 55M sentence pairs. Of the 99 language pairs, 44 have 1M sentence pairs of training data, 73 have at least 100k, and 95 have at least 10k.

<br>

# Model Summary 

```python
  | Name            | Type             | Params
-----------------------------------------------------
0 | net             | Transformer      | 68.1 M
1 | loss_fn         | CrossEntropyLoss | 0     
2 | char_error_rate | CharErrorRate    | 0     
3 | word_error_rate | WordErrorRate    | 0     
4 | bleu_score      | BLEUScore        | 0     
-----------------------------------------------------
68.1 M    Trainable params
0         Non-trainable params
68.1 M    Total params
272.582   Total estimated model params size (MB)
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

![architecture](../Results/Session%2016/Transformer%20Architecture.png)

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

![architecture-understanding](../Results/Session%2016/Architecture.png)

<br>

# Self Attention

- The `query` is the representation for the word we want to calculate self-attention for.
- The `key` is a representation of each word in the sequence and is used to match against the query of the word for which we currently want to calculate self-attention.
- The `value` is the actual representation of each word in a sequence, the representation we really care about. 

<br>

![self-attention](../Results/Session%2016/Attention.png)

<br>

# Inference

![inference](../Results/Session%2016/Inference.png)

<br>

# Training Logs

The training logs for the model can be found [here](Training_Logs.md)


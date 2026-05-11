# Brain Tumor Classification using ResNet + Transformer

## Overview
This project implements a binary image classification model for brain tumor classification (benign vs malignant). The architecture combines a pretrained ResNet50 convolutional backbone with a Transformer encoder to capture both local and global image features.

---

## Problem Definition
The objective is to classify histopathology images into two categories:
- Benign
- Malignant

This task is relevant for medical image analysis and early cancer detection research.

---

## Model Architecture
The model consists of the following components:

- ResNet50 backbone for feature extraction
- Feature tokenization layer to convert CNN feature maps into sequences
- Learnable positional encoding
- Transformer encoder for global dependency modeling
- Classification head using a fully connected layer

---

## Key Design Choices
- Transfer learning using pretrained ResNet50
- Hybrid CNN-Transformer architecture
- CLS token for sequence-level classification
- Layer normalization before classification head

---

## Training Pipeline
The training process includes:

- Optimizer: AdamW
- Loss function: CrossEntropyLoss
- Learning rate scheduler: CosineAnnealingLR
- Mixed precision training (FP16 optional)
- Gradient accumulation
- Gradient clipping
- Early stopping (optional)
- Model checkpointing

---

## Evaluation Metrics
The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

---

## Project Structure

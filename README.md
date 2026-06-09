# Distorted Visual Sequence Pattern Recognition using Deep Learning

## Overview

This project presents a deep learning solution for distorted visual sequence pattern recognition. The objective is to reconstruct ordered character sequences from grayscale images affected by challenging distortions such as noise, blur, overlapping symbols, shape deformation, occlusion, and irregular spacing.

The proposed solution uses a Convolutional Recurrent Neural Network (CRNN) architecture consisting of a residual CNN backbone for visual feature extraction and stacked Bidirectional LSTM layers for sequence modeling. Connectionist Temporal Classification (CTC) loss enables end-to-end training without explicit character segmentation, while beam search decoding and Test-Time Augmentation (TTA) improve inference performance.

The model is trained entirely from scratch using the provided dataset without any pretrained weights.

---

## Problem Statement

Given a distorted grayscale image containing an ordered sequence of characters, the task is to predict the correct character sequence.

Challenges present in the dataset include:

* Background noise
* Blur and visual artifacts
* Overlapping symbols
* Shape deformation
* Occlusion and random patches
* Irregular spacing and alignment

The evaluation metric used is **Character Error Rate (CER)**.

---

## Model Architecture

The proposed architecture follows a CRNN design:

* Residual CNN backbone for visual feature extraction
* Two stacked Bidirectional LSTM layers for sequence modeling
* Linear classification head
* CTC Loss for alignment-free training
* Beam Search decoding (width = 10)
* Test-Time Augmentation (5 inference passes)

### Architecture Flow

Input (B×1×48×160)

→ Stem: Conv 64, MaxPool → 24×80

→ Stage 1: ResBlock 128, stride=2 → 12×40

→ Stage 2: ResBlock ×2 (256) → 6×40

→ Stage 3: ResBlock ×2 (512) → 3×40

→ AdaptiveAvgPool → 1×40

→ BiLSTM ×2 (hidden=256) → 40×512

→ Linear → 40×NUM_CLASSES

→ Beam Search Decoding

→ Predicted Character Sequence

**Note:** The CNN backbone produces 40 sequence timesteps for a target sequence length of 6 characters, providing sufficient alignment flexibility for CTC decoding.

---

## Training Configuration

| Component               | Configuration          |
| ----------------------- | ---------------------- |
| Optimizer               | AdamW                  |
| Scheduler               | OneCycleLR             |
| Loss Function           | CTC Loss               |
| Sequence Decoder        | Beam Search (width=10) |
| Test-Time Augmentation  | 5 passes               |
| Early Stopping Patience | 12 epochs              |

---

## Results

### Validation Performance

| Metric                       | Value               |
| ---------------------------- | ------------------- |
| Best Validation CER (Greedy) | 0.0005              |
| Validation CER (Beam Search) | 0.0010              |
| Validation Exact Accuracy    | 99.55%              |
| Best Checkpoint              | Epoch 28            |
| Training Epochs              | 40 (Early Stopping) |

### Test Inference

| Metric                  | Value   |
| ----------------------- | ------- |
| Test Samples            | 5000    |
| Inference Time          | 214.8 s |
| Average Time per Sample | 43.0 ms |

---

## Key Techniques

* Residual CNN feature extraction
* Bidirectional LSTM sequence modeling
* Connectionist Temporal Classification (CTC)
* Beam Search decoding
* Test-Time Augmentation (TTA)
* Data augmentation using blur, brightness variation, and random erasing

---



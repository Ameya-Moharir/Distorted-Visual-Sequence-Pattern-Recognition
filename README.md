# Distorted Visual Sequence Pattern Recognition

CRNN (ResNet CNN + BiLSTM + CTC) solution for distorted CAPTCHA recognition.

## Results
- Val CER: 0.0004 (greedy) / 0.0018 (beam search)
- Val Exact Accuracy: 99.40%

## Architecture
Custom ResNet-style CNN backbone + 2-layer BiLSTM + CTC loss.
No pretrained weights. Trained from scratch on 20,000 images.

## Requirements
pip install torch torchvision pillow pandas matplotlib tqdm

## Usage
Run all cells in notebook_YourName_EnrollNo.ipynb top to bottom.
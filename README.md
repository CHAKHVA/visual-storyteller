# Visual Storyteller: Image Captioning with CNN-Transformer

An end-to-end deep learning system that generates natural language descriptions for images using a CNN encoder and Transformer decoder architecture.

## Overview

This project implements an image captioning model that combines:

- **Encoder**: Pretrained ResNet-101 CNN for extracting spatial visual features
- **Decoder**: 6-layer Transformer decoder with multi-head cross-attention for caption generation
- **Training Strategy**: Two-stage training with progressive encoder fine-tuning

The model is trained on the Flickr8k dataset and achieves competitive BLEU scores while generating coherent and descriptive captions.

## Features

- **Config-driven architecture**: All hyperparameters managed via YAML configuration
- **Flexible inference**: Supports both greedy decoding and beam search
- **Attention visualization**: Visualize what the model focuses on while generating captions
- **Robust training**: Includes checkpointing, early stopping, and resume capability
- **Comprehensive evaluation**: BLEU-1/2/3/4 metrics with detailed analysis
- **Interactive notebooks**: Jupyter notebooks for exploration and demonstration

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/CHAKHVA/visual-storyteller.git
cd visual-storyteller
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Download NLTK data (required for BLEU evaluation):

```python
python -c "import nltk; nltk.download('punkt')"
```

## Training

### Quick Start

Train with default configuration:

```bash
python -m src.train
```

### Custom Configuration

Modify `configs/config.yaml` to adjust hyperparameters, then run:

```bash
python -m src.train --config configs/config.yaml
```

### Resume Training

Resume from a checkpoint (e.g., after interruption):

```bash
python -m src.train --resume checkpoints/emergency_checkpoint.pt
```

## Inference

For interactive inference and visualization:

```bash
jupyter notebook notebooks/inference.ipynb
```

## Evaluation

Evaluate model performance on the test set:

```bash
python -m src.evaluate
```

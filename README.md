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

## Architecture

```
Input Image (224x224)
    ↓
ResNet-101 Encoder (pretrained on ImageNet)
    ↓
Spatial Features (7x7x2048) → Projection → (49, 512)
    ↓
Transformer Decoder (6 layers, 8 heads)
    ↓
Caption Generation (autoregressive with teacher forcing)
```

**Key Components:**

- **Embed Size**: 512
- **Attention Heads**: 8
- **Decoder Layers**: 6
- **Vocabulary**: ~2,500 tokens (frequency threshold ≥ 5)
- **Max Caption Length**: 50 tokens

## Installation

### Requirements

- Python 3.13+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/CHAKHVA/visual-storyteller.git
cd visual-storyteller
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for BLEU evaluation):

```python
python -c "import nltk; nltk.download('punkt')"
```

## Dataset Setup

This project uses the Flickr8k dataset.

### Directory Structure

Place your dataset in the following structure:

```
visual-storyteller/
├── caption_data/
│   ├── images/              # 8,000 JPEG images
│   └── captions.txt         # CSV file with image,caption columns
```

### Caption File Format

The `captions.txt` file should be a CSV with columns: `image,caption`

```csv
image,caption
1000268201_693b08cb0e.jpg,a child in a pink dress is climbing up a set of stairs in an entry way
1000268201_693b08cb0e.jpg,a girl going into a wooden building
...
```

Each image should have 5 captions (as per Flickr8k convention).

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

### Training Process

The training script will:

1. Build vocabulary from the dataset (or load existing)
2. Create train/val/test splits (75%/12.5%/12.5%)
3. Train decoder-only for first 5 epochs
4. Unfreeze encoder layer4 and fine-tune with differential learning rates
5. Apply early stopping (patience=5) based on validation loss
6. Save best model to `checkpoints/best_model.pt`

### Training Features

- **Gradient Clipping**: Max norm of 5.0 for stability
- **Label Smoothing**: 0.1 for regularization
- **Learning Rate Scheduling**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Differential Learning Rates**: Encoder LR = Decoder LR × 0.1
- **Emergency Checkpointing**: Press Ctrl+C to save progress

### Expected Training Time

- **CPU**: ~20-30 hours for 30 epochs
- **GPU (e.g., RTX 3080)**: ~3-5 hours for 30 epochs

## Inference

### Using Python Script

```python
from src.inference import load_model, generate_caption

# Load trained model
model, vocab, config = load_model("checkpoints/best_model.pt")

# Generate caption for an image
caption = generate_caption(
    image_path="path/to/image.jpg",
    model=model,
    vocab=vocab,
    device="cuda",
    method="beam",      # or "greedy"
    beam_width=5
)

print(f"Generated caption: {caption}")
```

### Using Jupyter Notebook

For interactive inference and visualization:

```bash
jupyter notebook notebooks/inference.ipynb
```

The inference notebook includes:

- Single image captioning demo
- Batch processing examples
- Attention visualization
- Success/failure analysis
- Custom image upload

## Evaluation

Evaluate model performance on the test set:

```bash
python -m src.evaluate
```

This will output:

- BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
- Comparison between greedy and beam search decoding
- Best and worst examples with reference captions

### Expected Performance

Typical BLEU-4 scores on Flickr8k:

- **Greedy Decoding**: 0.18-0.22
- **Beam Search (width=5)**: 0.20-0.24

## Project Structure

```
visual-storyteller/
├── src/
│   ├── __init__.py
│   ├── utils.py              # Config, device, seed utilities
│   ├── vocab.py              # Vocabulary class
│   ├── dataset.py            # Dataset, transforms, dataloaders
│   ├── model.py              # CNN encoder + Transformer decoder
│   ├── train.py              # Training loop and orchestration
│   ├── inference.py          # Caption generation and beam search
│   └── evaluate.py           # BLEU metrics and analysis
├── configs/
│   └── config.yaml           # Hyperparameters configuration
├── notebooks/
│   ├── data_and_training.ipynb    # Data exploration and training
│   └── inference.ipynb            # Inference demos and visualization
├── checkpoints/              # Saved models and vocabulary
│   ├── best_model.pt
│   └── vocab.pkl
├── caption_data/             # Dataset (not included in repo)
│   ├── images/
│   └── captions.txt
├── requirements.txt
└── README.md
```

## Configuration

Key hyperparameters in `configs/config.yaml`:

### Data Configuration

```yaml
data:
  batch_size: 32
  freq_threshold: 5 # Min word frequency for vocabulary
  max_caption_length: 50
  num_workers: 4
  train_split: 0.75
  val_split: 0.125
```

### Model Configuration

```yaml
model:
  encoder_backbone: "resnet101"
  embed_size: 512
  num_heads: 8
  num_layers: 6
  dropout: 0.1
```

### Training Configuration

```yaml
training:
  epochs: 30
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  grad_clip: 5.0
  label_smoothing: 0.1
  unfreeze_encoder_epoch: 5 # When to start fine-tuning encoder
  encoder_lr_factor: 0.1 # Encoder LR multiplier
  early_stopping_patience: 5
```

### Inference Configuration

```yaml
inference:
  max_length: 50
  beam_width: 5
  temperature: 1.0
  length_penalty: 0.7 # For beam search normalization
```

## Notebooks

### 1. Data and Training (`notebooks/data_and_training.ipynb`)

Comprehensive notebook covering:

- Dataset statistics and exploration
- Caption length distribution analysis
- Word frequency visualization
- Vocabulary coverage analysis
- Sample images with captions
- Model architecture summary
- Training execution
- Parameter counts and layer breakdown

### 2. Inference (`notebooks/inference.ipynb`)

Interactive demos including:

- Model loading and setup
- Single image captioning
- Batch inference examples
- Quantitative evaluation (BLEU scores)
- Success and failure case analysis
- Attention visualization
- Custom image upload and testing

## Advanced Features

### Attention Visualization

Visualize cross-attention weights to see which image regions the model focuses on:

```python
from src.inference import visualize_attention

visualize_attention(
    image_path="path/to/image.jpg",
    model=model,
    vocab=vocab,
    device="cuda"
)
```

The visualization shows:

- Original image
- Attention heatmap for each generated word
- Final caption with attention overlay

### Batch Processing

Generate captions for multiple images efficiently:

```python
from src.inference import generate_caption_batch

image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
captions = generate_caption_batch(
    image_paths=image_paths,
    model=model,
    vocab=vocab,
    device="cuda",
    batch_size=8
)

for path, caption in zip(image_paths, captions):
    print(f"{path}: {caption}")
```

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM) Error**

- Reduce `batch_size` in config.yaml
- Use gradient accumulation
- Use a smaller model (reduce `embed_size` or `num_layers`)

**2. Vocabulary Not Found**

- Delete `checkpoints/vocab.pkl` to rebuild from scratch
- Ensure `captions.txt` is in correct format

**3. Low BLEU Scores**

- Train for more epochs
- Increase vocabulary size (lower `freq_threshold`)
- Use beam search instead of greedy decoding
- Fine-tune encoder earlier (reduce `unfreeze_encoder_epoch`)

**4. Training Interrupted**

- Use the emergency checkpoint: `--resume checkpoints/emergency_checkpoint.pt`
- Ctrl+C during training automatically saves progress

**5. CUDA Out of Memory**

```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

## Model Checkpoints

Checkpoints contain:

- Model state dict (encoder + decoder weights)
- Optimizer state dict
- Vocabulary object
- Configuration dictionary
- Training epoch number
- Validation loss

Load a checkpoint manually:

```python
import torch
checkpoint = torch.load("checkpoints/best_model.pt", weights_only=False)
print(f"Epoch: {checkpoint['epoch']}")
print(f"Loss: {checkpoint['loss']:.4f}")
print(f"Vocab size: {len(checkpoint['vocab'])}")
```

## Technical Details

### Vocabulary Construction

- Special tokens: `<PAD>` (0), `<SOS>` (1), `<EOS>` (2), `<UNK>` (3)
- Frequency threshold: Words appearing < 5 times → `<UNK>`
- Typical vocabulary size: ~2,500 tokens
- Coverage: ~95% of words in dataset

### Data Augmentation

**Training:**

- Random resized crop (224x224)
- Random horizontal flip
- ImageNet normalization

**Validation/Test:**

- Resize to 256x256
- Center crop to 224x224
- ImageNet normalization

### Loss Function

CrossEntropyLoss with:

- `ignore_index=0` (ignore padding tokens)
- `label_smoothing=0.1` (prevent overconfidence)

### Beam Search

Length-normalized beam search:

```
score = log_prob / (length ** 0.7)
```

This prevents bias toward shorter sequences.

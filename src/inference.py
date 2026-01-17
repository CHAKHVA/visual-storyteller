"""Inference utilities for generating captions from images."""

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image

from src.dataset import get_transforms
from src.model import ImageCaptioningModel
from src.vocab import Vocabulary


def load_model(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[ImageCaptioningModel, Vocabulary, dict]:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load model onto.

    Returns:
        Tuple of (model, vocabulary, config).
    """
    # Create a dummy model to load checkpoint into
    # We'll get the actual architecture from config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract vocab and config
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]

    # Create model from config
    model = ImageCaptioningModel.create_from_config(config, vocab_size=len(vocab))

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set to eval mode
    model.eval()

    # Move to device
    model = model.to(device)

    print(f"Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    print(f"  Vocab size: {len(vocab)}")

    return model, vocab, config


def preprocess_image(image_path: str, transform) -> torch.Tensor:
    """
    Load and preprocess an image.

    Args:
        image_path: Path to the image file.
        transform: Torchvision transform to apply.

    Returns:
        Preprocessed image tensor with batch dimension (1, C, H, W).
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Apply transform
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def greedy_decode(
    model: ImageCaptioningModel,
    image_tensor: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = 50,
    device: str = "cpu",
) -> list[int]:
    """
    Generate caption using greedy decoding.

    Args:
        model: Trained image captioning model.
        image_tensor: Preprocessed image tensor (1, C, H, W).
        vocab: Vocabulary object.
        max_len: Maximum caption length.
        device: Device to run inference on.

    Returns:
        List of token indices (excluding <SOS>).
    """
    model.eval()

    # Move image to device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Get encoder features (once, reused for all decoding steps)
        encoder_out = model.encoder(image_tensor)  # (1, 49, embed_size)

        # Initialize sequence with <SOS> token
        sequence = [vocab.stoi["<SOS>"]]

        for _ in range(max_len):
            # Convert sequence to tensor (1, seq_len)
            current_seq = torch.tensor([sequence], dtype=torch.long, device=device)

            # Get decoder output
            # Pass through decoder (need to provide encoder output and current sequence)
            outputs = model.decoder(
                encoder_out, current_seq
            )  # (1, seq_len, vocab_size)

            # Get prediction for the last position
            last_output = outputs[:, -1, :]  # (1, vocab_size)

            # Get argmax (greedy selection)
            predicted_token = last_output.argmax(dim=-1).item()

            # Check if <EOS> token
            if predicted_token == vocab.stoi["<EOS>"]:
                break

            # Append to sequence
            sequence.append(predicted_token)

    # Return sequence excluding <SOS>
    return sequence[1:]


def generate_caption(
    image_path: str,
    model: ImageCaptioningModel,
    vocab: Vocabulary,
    device: str = "cpu",
    max_len: int = 50,
) -> str:
    """
    Generate caption for an image.

    This is the main interface for inference.

    Args:
        image_path: Path to the image file.
        model: Trained image captioning model.
        vocab: Vocabulary object.
        device: Device to run inference on.
        max_len: Maximum caption length.

    Returns:
        Generated caption as a string.
    """
    # Get transform for evaluation
    transform = get_transforms(train=False)

    # Preprocess image
    image_tensor = preprocess_image(image_path, transform)

    # Generate caption using greedy decoding
    token_ids = greedy_decode(model, image_tensor, vocab, max_len, device)

    # Convert tokens to words
    caption = vocab.denumericalize(token_ids)

    return caption


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("Testing Inference Functions")
    print("=" * 70)

    # Check if checkpoint exists
    checkpoint_path = "checkpoints/best_model.pt"

    if not Path(checkpoint_path).exists():
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Please train a model first using: python -m src.train")
        sys.exit(0)

    # Load model
    print(f"\n[1/3] Loading model from checkpoint...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    model, vocab, config = load_model(checkpoint_path, device=device)

    print("\n[2/3] Testing with a sample image...")

    # Get a sample image from the dataset
    image_dir = Path(config["data"]["image_dir"])

    if not image_dir.exists():
        print(f"\nImage directory not found: {image_dir}")
        print("Cannot test without images.")
        sys.exit(0)

    # Get first image from the directory
    image_files = list(image_dir.glob("*.jpg"))

    if not image_files:
        image_files = list(image_dir.glob("*.png"))

    if not image_files:
        print(f"\nNo images found in {image_dir}")
        sys.exit(0)

    test_image_path = str(image_files[0])
    print(f"  Testing with image: {test_image_path}")

    # Generate caption
    print("\n[3/3] Generating caption...")
    caption = generate_caption(
        image_path=test_image_path,
        model=model,
        vocab=vocab,
        device=device,
        max_len=config["data"]["max_caption_length"],
    )

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Image: {Path(test_image_path).name}")
    print(f"Caption: {caption}")
    print("=" * 70)

    # Test with a few more images
    print("\n\nTesting with more images...")
    print("-" * 70)

    for i, image_path in enumerate(image_files[1:6], 1):  # Test 5 more images
        caption = generate_caption(
            image_path=str(image_path),
            model=model,
            vocab=vocab,
            device=device,
            max_len=config["data"]["max_caption_length"],
        )
        print(f"{i}. {image_path.name}: {caption}")

    print("\n" + "=" * 70)
    print("All inference tests passed!")
    print("=" * 70)

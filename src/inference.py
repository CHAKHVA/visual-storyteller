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


def beam_search(
    model: ImageCaptioningModel,
    image_tensor: torch.Tensor,
    vocab: Vocabulary,
    beam_width: int = 5,
    max_len: int = 50,
    device: str = "cpu",
) -> list[int]:
    """
    Generate caption using beam search decoding.

    Beam search explores multiple candidate sequences simultaneously,
    keeping the top beam_width sequences at each step.

    Args:
        model: Trained image captioning model.
        image_tensor: Preprocessed image tensor (1, C, H, W).
        vocab: Vocabulary object.
        beam_width: Number of beams to keep at each step.
        max_len: Maximum caption length.
        device: Device to run inference on.

    Returns:
        List of token indices for best sequence (excluding <SOS>).
    """
    model.eval()

    # Move image to device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Get encoder features (once, reused for all beams)
        encoder_out = model.encoder(image_tensor)  # (1, 49, embed_size)

        # Initialize: start with <SOS> token
        # Each beam is: (sequence, score)
        active_beams = [([vocab.stoi["<SOS>"]], 0.0)]
        completed_sequences = []

        for step in range(max_len):
            candidates = []

            for sequence, score in active_beams:
                # If sequence ends with <EOS>, move to completed
                if sequence[-1] == vocab.stoi["<EOS>"]:
                    completed_sequences.append((sequence, score))
                    continue

                # Convert sequence to tensor (1, seq_len)
                current_seq = torch.tensor([sequence], dtype=torch.long, device=device)

                # Expand encoder_out to match beam size
                # For single image, encoder_out is (1, 49, embed_size)
                outputs = model.decoder(
                    encoder_out, current_seq
                )  # (1, seq_len, vocab_size)

                # Get log probabilities for last position
                last_output = outputs[:, -1, :]  # (1, vocab_size)
                log_probs = torch.log_softmax(last_output, dim=-1)  # (1, vocab_size)

                # Get top beam_width candidates
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

                # Create new candidate sequences
                for log_prob, token_id in zip(top_log_probs[0], top_indices[0]):
                    new_sequence = sequence + [token_id.item()]
                    # Cumulative score (sum of log probabilities)
                    new_score = score + log_prob.item()
                    candidates.append((new_sequence, new_score))

            # If no candidates (all completed), break
            if not candidates:
                break

            # Sort candidates by normalized score (to avoid bias toward short sequences)
            # Length normalization: score / (length ** alpha), alpha=0.7 is common
            candidates.sort(
                key=lambda x: x[1] / (len(x[0]) ** 0.7),
                reverse=True,
            )

            # Keep top beam_width sequences
            active_beams = candidates[:beam_width]

            # Check if all beams have completed
            if all(seq[-1] == vocab.stoi["<EOS>"] for seq, _ in active_beams):
                completed_sequences.extend(active_beams)
                break

        # Combine completed and active beams
        all_sequences = completed_sequences + active_beams

        # Sort by normalized score
        all_sequences.sort(
            key=lambda x: x[1] / (len(x[0]) ** 0.7),
            reverse=True,
        )

        # Get best sequence
        best_sequence = all_sequences[0][0]

        # Return sequence excluding <SOS> (and <EOS> if present)
        result = best_sequence[1:]
        if result and result[-1] == vocab.stoi["<EOS>"]:
            result = result[:-1]

        return result


def generate_caption(
    image_path: str,
    model: ImageCaptioningModel,
    vocab: Vocabulary,
    device: str = "cpu",
    max_len: int = 50,
    method: str = "greedy",
    beam_width: int = 5,
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
        method: Decoding method - "greedy" or "beam".
        beam_width: Beam width for beam search (only used if method="beam").

    Returns:
        Generated caption as a string.
    """
    # Get transform for evaluation
    transform = get_transforms(train=False)

    # Preprocess image
    image_tensor = preprocess_image(image_path, transform)

    # Generate caption using specified method
    if method == "greedy":
        token_ids = greedy_decode(model, image_tensor, vocab, max_len, device)
    elif method == "beam":
        token_ids = beam_search(model, image_tensor, vocab, beam_width, max_len, device)
    else:
        raise ValueError(f"Unknown decoding method: {method}. Use 'greedy' or 'beam'.")

    # Convert tokens to words
    caption = vocab.denumericalize(token_ids)

    return caption


def generate_caption_batch(
    image_paths: list[str],
    model: ImageCaptioningModel,
    vocab: Vocabulary,
    device: str = "cpu",
    max_len: int = 50,
    method: str = "greedy",
    beam_width: int = 5,
) -> list[str]:
    """
    Generate captions for multiple images.

    Processes images one by one (true batch processing would require
    more complex beam search implementation with batched decoding).

    Args:
        image_paths: List of paths to image files.
        model: Trained image captioning model.
        vocab: Vocabulary object.
        device: Device to run inference on.
        max_len: Maximum caption length.
        method: Decoding method - "greedy" or "beam".
        beam_width: Beam width for beam search.

    Returns:
        List of generated captions.
    """
    captions = []

    for image_path in image_paths:
        caption = generate_caption(
            image_path=image_path,
            model=model,
            vocab=vocab,
            device=device,
            max_len=max_len,
            method=method,
            beam_width=beam_width,
        )
        captions.append(caption)

    return captions


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

    # Generate caption with greedy decoding
    print("\n[3/5] Generating caption with greedy decoding...")
    caption_greedy = generate_caption(
        image_path=test_image_path,
        model=model,
        vocab=vocab,
        device=device,
        max_len=config["data"]["max_caption_length"],
        method="greedy",
    )

    # Generate caption with beam search
    print("\n[4/5] Generating caption with beam search...")
    beam_width = config["inference"]["beam_width"]
    caption_beam = generate_caption(
        image_path=test_image_path,
        model=model,
        vocab=vocab,
        device=device,
        max_len=config["data"]["max_caption_length"],
        method="beam",
        beam_width=beam_width,
    )

    print("\n" + "=" * 70)
    print("Results - Single Image")
    print("=" * 70)
    print(f"Image: {Path(test_image_path).name}")
    print(f"\nGreedy Decoding:")
    print(f"  {caption_greedy}")
    print(f"\nBeam Search (width={beam_width}):")
    print(f"  {caption_beam}")
    print("=" * 70)

    # Test batch processing
    print("\n[5/5] Testing batch processing...")
    test_images = [str(p) for p in image_files[:3]]

    print(f"  Processing {len(test_images)} images with greedy decoding...")
    captions_greedy = generate_caption_batch(
        image_paths=test_images,
        model=model,
        vocab=vocab,
        device=device,
        max_len=config["data"]["max_caption_length"],
        method="greedy",
    )

    print(f"  Processing {len(test_images)} images with beam search...")
    captions_beam = generate_caption_batch(
        image_paths=test_images,
        model=model,
        vocab=vocab,
        device=device,
        max_len=config["data"]["max_caption_length"],
        method="beam",
        beam_width=beam_width,
    )

    # Compare greedy vs beam search
    print("\n" + "=" * 70)
    print("Greedy vs Beam Search Comparison")
    print("=" * 70)

    for i, (img_path, cap_greedy, cap_beam) in enumerate(
        zip(test_images, captions_greedy, captions_beam), 1
    ):
        img_name = Path(img_path).name
        print(f"\n{i}. {img_name}")
        print(f"   Greedy: {cap_greedy}")
        print(f"   Beam:   {cap_beam}")

        # Check if they're different
        if cap_greedy != cap_beam:
            print("   → Different captions generated")
        else:
            print("   → Same caption")

    print("\n" + "=" * 70)
    print("All inference tests passed!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - Greedy decoding: Fast, deterministic")
    print(
        f"  - Beam search (width={beam_width}): Explores more options, often better quality"
    )
    print("=" * 70)

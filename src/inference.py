"""Inference utilities for generating captions from images."""

import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.dataset import get_transforms
from src.model import ImageCaptioningModel
from src.vocab import Vocabulary


def load_model(
    checkpoint_path: str, device: str = "cpu"
) -> Tuple[ImageCaptioningModel, Vocabulary, dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab = checkpoint["vocab"]
    config = checkpoint["config"]

    model = ImageCaptioningModel.create_from_config(config, vocab_size=len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)

    print(f"Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    print(f"  Vocab size: {len(vocab)}")

    return model, vocab, config


def preprocess_image(image_path: str, transform) -> torch.Tensor:
    """Load and preprocess an image."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)


def greedy_decode(
    model: ImageCaptioningModel,
    image_tensor: torch.Tensor,
    vocab: Vocabulary,
    max_len: int = 50,
    device: str = "cpu",
) -> list[int]:
    """Generate caption using greedy decoding."""
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)
        sequence = [vocab.stoi["<SOS>"]]

        for _ in range(max_len):
            current_seq = torch.tensor([sequence], dtype=torch.long, device=device)
            outputs = model.decoder(encoder_out, current_seq)
            predicted_token = outputs[:, -1, :].argmax(dim=-1).item()

            if predicted_token == vocab.stoi["<EOS>"]:
                break
            sequence.append(predicted_token)

    return sequence[1:]


def beam_search(
    model: ImageCaptioningModel,
    image_tensor: torch.Tensor,
    vocab: Vocabulary,
    beam_width: int = 5,
    max_len: int = 50,
    device: str = "cpu",
) -> list[int]:
    """Generate caption using beam search decoding."""
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)
        active_beams = [([vocab.stoi["<SOS>"]], 0.0)]
        completed_sequences = []

        for step in range(max_len):
            candidates = []

            for sequence, score in active_beams:
                if sequence[-1] == vocab.stoi["<EOS>"]:
                    completed_sequences.append((sequence, score))
                    continue

                current_seq = torch.tensor([sequence], dtype=torch.long, device=device)
                outputs = model.decoder(encoder_out, current_seq)
                last_output = outputs[:, -1, :]
                log_probs = torch.log_softmax(last_output, dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

                for log_prob, token_id in zip(top_log_probs[0], top_indices[0]):
                    new_sequence = sequence + [token_id.item()]
                    new_score = score + log_prob.item()
                    candidates.append((new_sequence, new_score))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)
            active_beams = candidates[:beam_width]

            if all(seq[-1] == vocab.stoi["<EOS>"] for seq, _ in active_beams):
                completed_sequences.extend(active_beams)
                break

        all_sequences = completed_sequences + active_beams
        all_sequences.sort(key=lambda x: x[1] / (len(x[0]) ** 0.7), reverse=True)

        best_sequence = all_sequences[0][0]
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
    """Generate caption for an image."""
    transform = get_transforms(train=False)
    image_tensor = preprocess_image(image_path, transform)

    if method == "greedy":
        token_ids = greedy_decode(model, image_tensor, vocab, max_len, device)
    elif method == "beam":
        token_ids = beam_search(model, image_tensor, vocab, beam_width, max_len, device)
    else:
        raise ValueError(f"Unknown decoding method: {method}. Use 'greedy' or 'beam'.")

    return vocab.denumericalize(token_ids)


def generate_caption_batch(
    image_paths: list[str],
    model: ImageCaptioningModel,
    vocab: Vocabulary,
    device: str = "cpu",
    max_len: int = 50,
    method: str = "greedy",
    beam_width: int = 5,
) -> list[str]:
    """Generate captions for multiple images."""
    return [
        generate_caption(
            image_path, model, vocab, device, max_len, method, beam_width
        )
        for image_path in image_paths
    ]


def visualize_attention(
    image_path: str,
    model: ImageCaptioningModel,
    vocab: Vocabulary,
    device: str = "cpu",
    max_len: int = 50,
) -> plt.Figure:
    """Visualize attention weights for each generated word."""
    from scipy.ndimage import zoom

    model.eval()
    transform = get_transforms(train=False)
    image_tensor = preprocess_image(image_path, transform).to(device)
    original_img = Image.open(image_path).convert("RGB")
    img_array = np.array(original_img)

    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)
        sequence = [vocab.stoi["<SOS>"]]
        attention_maps = []

        for _ in range(max_len):
            current_seq = torch.tensor([sequence], dtype=torch.long, device=device)
            outputs, attention_weights = model.decoder.forward_with_attention(
                encoder_out, current_seq
            )
            predicted_token = outputs[:, -1, :].argmax(dim=-1).item()
            attn = attention_weights[0, -1, :].cpu().numpy()
            attention_maps.append(attn)

            if predicted_token == vocab.stoi["<EOS>"]:
                break
            sequence.append(predicted_token)

    generated_words = [vocab.itos[idx] for idx in sequence[1:]]
    num_words = len(generated_words)
    cols = min(5, num_words)
    rows = math.ceil((num_words + 1) / cols)

    fig = plt.figure(figsize=(cols * 3, rows * 3))
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(img_array)
    ax.axis("off")
    caption = " ".join(generated_words)
    ax.set_title(f"Generated Caption:\n{caption}", fontsize=10, weight="bold")

    for idx, (word, attn) in enumerate(zip(generated_words, attention_maps)):
        ax = plt.subplot(rows, cols, idx + 2)
        attn_map = attn.reshape(7, 7)
        img_h, img_w = img_array.shape[:2]
        attn_upscaled = zoom(attn_map, (img_h / 7, img_w / 7), order=1)
        attn_upscaled = (attn_upscaled - attn_upscaled.min()) / (
            attn_upscaled.max() - attn_upscaled.min() + 1e-8
        )
        ax.imshow(img_array)
        ax.imshow(attn_upscaled, cmap="jet", alpha=0.6, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f'"{word}"', fontsize=12, weight="bold")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    checkpoint_path = "checkpoints/best_model.pt"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, config = load_model(checkpoint_path, device=device)

    image_dir = Path(config["data"]["image_dir"])
    image_files = list(image_dir.glob("*.jpg")) or list(image_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {image_dir}")
        exit(0)

    test_image = str(image_files[0])
    caption_greedy = generate_caption(test_image, model, vocab, device, method="greedy")
    caption_beam = generate_caption(
        test_image, model, vocab, device, method="beam", beam_width=5
    )

    print(f"Image: {Path(test_image).name}")
    print(f"Greedy: {caption_greedy}")
    print(f"Beam:   {caption_beam}")

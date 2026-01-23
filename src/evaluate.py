"""Evaluation utilities for image captioning model."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.inference import generate_caption, load_model


def calculate_bleu(
    references: list[list[list[str]]], hypotheses: list[list[str]]
) -> dict[str, float]:
    """Calculate BLEU-1 through BLEU-4 scores."""
    smoothing = SmoothingFunction().method1

    return {
        "BLEU-1": corpus_bleu(
            references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothing
        ),
        "BLEU-2": corpus_bleu(
            references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing
        ),
        "BLEU-3": corpus_bleu(
            references,
            hypotheses,
            weights=(1 / 3, 1 / 3, 1 / 3, 0),
            smoothing_function=smoothing,
        ),
        "BLEU-4": corpus_bleu(
            references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        ),
    }


def evaluate_model(
    model: Any,
    test_loader: DataLoader,
    vocab: Any,
    device: str,
    method: str = "greedy",
    beam_width: int = 5,
) -> dict[str, float]:
    """Evaluate model on test set using BLEU metrics."""
    model.eval()
    image_to_refs = defaultdict(list)

    print(f"Evaluating model on test set using {method} decoding...")
    print("Collecting reference captions...")
    for images, captions, image_names in tqdm(test_loader, desc="References"):
        captions = captions.transpose(0, 1)
        for i, img_name in enumerate(image_names):
            caption_tokens = captions[i].tolist()
            caption_text = vocab.denumericalize(caption_tokens)
            image_to_refs[img_name].append(caption_text.split())

    unique_images = list(image_to_refs.keys())
    print(f"Found {len(unique_images)} unique images with references")

    all_references, all_hypotheses = [], []
    print("Generating captions...")
    image_dir = Path(test_loader.dataset.image_dir)

    for img_name in tqdm(unique_images, desc="Generation"):
        try:
            generated_caption = generate_caption(
                str(image_dir / img_name), model, vocab, device, method=method, beam_width=beam_width
            )
            all_hypotheses.append(generated_caption.split())
            all_references.append(image_to_refs[img_name])
        except Exception as e:
            print(f"\nWarning: Failed to generate caption for {img_name}: {e}")

    print(f"\nEvaluated {len(all_hypotheses)} images")
    print("Calculating BLEU scores...")
    return calculate_bleu(all_references, all_hypotheses)


def analyze_captions(
    model: Any,
    test_loader: DataLoader,
    vocab: Any,
    device: str,
    num_samples: int = 20,
    method: str = "greedy",
    beam_width: int = 5,
) -> list[dict]:
    """Analyze generated captions with BLEU-4 scores."""
    model.eval()
    image_to_refs = defaultdict(list)
    image_to_refs_text = defaultdict(list)

    print("Collecting reference captions...")
    for images, captions, image_names in tqdm(test_loader, desc="References"):
        captions = captions.transpose(0, 1)
        for i, img_name in enumerate(image_names):
            caption_text = vocab.denumericalize(captions[i].tolist())
            image_to_refs[img_name].append(caption_text.split())
            image_to_refs_text[img_name].append(caption_text)

    unique_images = list(image_to_refs.keys())[:num_samples]
    print(f"\nAnalyzing {len(unique_images)} images...")
    image_dir = Path(test_loader.dataset.image_dir)
    smoothing = SmoothingFunction().method1
    results = []

    for img_name in tqdm(unique_images, desc="Analysis"):
        try:
            generated_caption = generate_caption(
                str(image_dir / img_name), model, vocab, device, method=method, beam_width=beam_width
            )
            bleu_4 = corpus_bleu(
                [image_to_refs[img_name]],
                [generated_caption.split()],
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
            results.append({
                "image_path": img_name,
                "generated_caption": generated_caption,
                "reference_captions": image_to_refs_text[img_name],
                "bleu_4": bleu_4,
            })
        except Exception as e:
            print(f"\nWarning: Failed to analyze {img_name}: {e}")

    results.sort(key=lambda x: x["bleu_4"], reverse=True)
    return results


if __name__ == "__main__":
    from src.dataset import get_dataloaders
    from src.utils import get_device

    checkpoint_path = "checkpoints/best_model.pt"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        exit(0)

    device = get_device()
    model, vocab, config = load_model(checkpoint_path, device=str(device))
    train_loader, val_loader, test_loader = get_dataloaders(config, vocab)

    metrics_greedy = evaluate_model(model, test_loader, vocab, str(device), method="greedy")
    metrics_beam = evaluate_model(
        model, test_loader, vocab, str(device), method="beam", beam_width=config["inference"]["beam_width"]
    )

    print("Greedy:", {k: f"{v:.4f}" for k, v in metrics_greedy.items()})
    print("Beam:  ", {k: f"{v:.4f}" for k, v in metrics_beam.items()})

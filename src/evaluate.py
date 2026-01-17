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
    """
    Calculate BLEU scores (BLEU-1 through BLEU-4).

    Args:
        references: List of reference sets. For each hypothesis, there are
            multiple references. Each reference is a list of tokens.
            Shape: [num_samples, num_refs_per_sample, num_tokens]
        hypotheses: List of generated captions. Each hypothesis is a list of tokens.
            Shape: [num_samples, num_tokens]

    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.
    """
    # Use smoothing to handle cases where n-gram precision is zero
    smoothing = SmoothingFunction().method1

    # Calculate BLEU scores with different n-gram weights
    bleu_scores = {}

    # BLEU-1: unigrams only
    bleu_1 = corpus_bleu(
        references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothing
    )

    # BLEU-2: up to bigrams
    bleu_2 = corpus_bleu(
        references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing
    )

    # BLEU-3: up to trigrams
    bleu_3 = corpus_bleu(
        references,
        hypotheses,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smoothing,
    )

    # BLEU-4: up to 4-grams (standard BLEU)
    bleu_4 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    bleu_scores["BLEU-1"] = bleu_1
    bleu_scores["BLEU-2"] = bleu_2
    bleu_scores["BLEU-3"] = bleu_3
    bleu_scores["BLEU-4"] = bleu_4

    return bleu_scores


def evaluate_model(
    model: Any,
    test_loader: DataLoader,
    vocab: Any,
    device: str,
    method: str = "greedy",
    beam_width: int = 5,
) -> dict[str, float]:
    """
    Evaluate model on test set using BLEU metrics.

    Args:
        model: Trained image captioning model.
        test_loader: DataLoader for test set.
        vocab: Vocabulary object.
        device: Device to run inference on.
        method: Decoding method - "greedy" or "beam".
        beam_width: Beam width for beam search.

    Returns:
        Dictionary containing BLEU scores.
    """
    model.eval()

    all_references = []  # List of reference sets
    all_hypotheses = []  # List of generated captions

    print(f"Evaluating model on test set using {method} decoding...")

    # Need to track which captions belong to which image
    # Since dataset returns image_names, we can use that
    image_to_refs = defaultdict(list)

    # First pass: collect all references from the dataset
    # The dataset has 5 captions per image
    print("Collecting reference captions...")
    for images, captions, image_names in tqdm(test_loader, desc="References"):
        # captions shape: (seq_len, batch)
        # Transpose to (batch, seq_len)
        captions = captions.transpose(0, 1)

        for i, img_name in enumerate(image_names):
            caption_tokens = captions[i].tolist()
            # Denumericalize and tokenize
            caption_text = vocab.denumericalize(caption_tokens)
            tokens = caption_text.split()
            image_to_refs[img_name].append(tokens)

    # Get unique images (each appears once in test set for generation)
    unique_images = list(image_to_refs.keys())

    print(f"Found {len(unique_images)} unique images with references")

    # Second pass: generate captions
    print("Generating captions...")
    image_dir = Path(test_loader.dataset.image_dir)

    for img_name in tqdm(unique_images, desc="Generation"):
        img_path = str(image_dir / img_name)

        # Generate caption
        try:
            generated_caption = generate_caption(
                image_path=img_path,
                model=model,
                vocab=vocab,
                device=device,
                method=method,
                beam_width=beam_width,
            )

            # Tokenize generated caption
            hypothesis = generated_caption.split()

            # Get references for this image (should be 5)
            references = image_to_refs[img_name]

            all_hypotheses.append(hypothesis)
            all_references.append(references)
        except Exception as e:
            print(f"\nWarning: Failed to generate caption for {img_name}: {e}")
            continue

    print(f"\nEvaluated {len(all_hypotheses)} images")

    # Calculate BLEU scores
    print("Calculating BLEU scores...")
    bleu_scores = calculate_bleu(all_references, all_hypotheses)

    return bleu_scores


def analyze_captions(
    model: Any,
    test_loader: DataLoader,
    vocab: Any,
    device: str,
    num_samples: int = 20,
    method: str = "greedy",
    beam_width: int = 5,
) -> list[dict]:
    """
    Analyze generated captions with detailed metrics.

    Args:
        model: Trained image captioning model.
        test_loader: DataLoader for test set.
        vocab: Vocabulary object.
        device: Device to run inference on.
        num_samples: Number of samples to analyze.
        method: Decoding method - "greedy" or "beam".
        beam_width: Beam width for beam search.

    Returns:
        List of dictionaries containing analysis for each sample.
    """
    model.eval()

    results = []

    # Collect references first
    image_to_refs = defaultdict(list)
    image_to_refs_text = defaultdict(list)

    print("Collecting reference captions...")
    for images, captions, image_names in tqdm(test_loader, desc="References"):
        captions = captions.transpose(0, 1)

        for i, img_name in enumerate(image_names):
            caption_tokens = captions[i].tolist()
            caption_text = vocab.denumericalize(caption_tokens)
            tokens = caption_text.split()

            image_to_refs[img_name].append(tokens)
            image_to_refs_text[img_name].append(caption_text)

    # Get unique images
    unique_images = list(image_to_refs.keys())[:num_samples]

    print(f"\nAnalyzing {len(unique_images)} images...")
    image_dir = Path(test_loader.dataset.image_dir)

    smoothing = SmoothingFunction().method1

    for img_name in tqdm(unique_images, desc="Analysis"):
        img_path = str(image_dir / img_name)

        # Generate caption
        try:
            generated_caption = generate_caption(
                image_path=img_path,
                model=model,
                vocab=vocab,
                device=device,
                method=method,
                beam_width=beam_width,
            )

            # Tokenize
            hypothesis = generated_caption.split()
            references = image_to_refs[img_name]

            # Calculate individual BLEU-4 score
            bleu_4 = corpus_bleu(
                [references],
                [hypothesis],
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )

            result = {
                "image_path": img_name,
                "generated_caption": generated_caption,
                "reference_captions": image_to_refs_text[img_name],
                "bleu_4": bleu_4,
            }

            results.append(result)
        except Exception as e:
            print(f"\nWarning: Failed to analyze {img_name}: {e}")
            continue

    # Sort by BLEU-4 score (descending)
    results.sort(key=lambda x: x["bleu_4"], reverse=True)

    return results


if __name__ == "__main__":
    import sys

    from src.dataset import get_dataloaders
    from src.utils import get_device, load_config

    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)

    # Check if checkpoint exists
    checkpoint_path = "checkpoints/best_model.pt"

    if not Path(checkpoint_path).exists():
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Please train a model first using: python -m src.train")
        sys.exit(0)

    # Load model
    print("\n[1/4] Loading model...")
    device = get_device()
    print(f"  Device: {device}")

    model, vocab, config = load_model(checkpoint_path, device=str(device))

    # Load test data
    print("\n[2/4] Loading test data...")
    train_loader, val_loader, test_loader = get_dataloaders(config, vocab)
    print(f"  Test set: {len(test_loader)} batches")

    # Evaluate with greedy decoding
    print("\n[3/4] Evaluating with greedy decoding...")
    metrics_greedy = evaluate_model(
        model=model,
        test_loader=test_loader,
        vocab=vocab,
        device=str(device),
        method="greedy",
    )

    print("\n" + "=" * 80)
    print("Greedy Decoding Results")
    print("=" * 80)
    for metric, score in metrics_greedy.items():
        print(f"{metric}: {score:.4f}")

    # Evaluate with beam search
    print("\n[4/4] Evaluating with beam search...")
    beam_width = config["inference"]["beam_width"]
    metrics_beam = evaluate_model(
        model=model,
        test_loader=test_loader,
        vocab=vocab,
        device=str(device),
        method="beam",
        beam_width=beam_width,
    )

    print("\n" + "=" * 80)
    print(f"Beam Search Results (width={beam_width})")
    print("=" * 80)
    for metric, score in metrics_beam.items():
        print(f"{metric}: {score:.4f}")

    # Detailed analysis
    print("\n" + "=" * 80)
    print("Detailed Analysis (20 samples)")
    print("=" * 80)

    results = analyze_captions(
        model=model,
        test_loader=test_loader,
        vocab=vocab,
        device=str(device),
        num_samples=20,
        method="beam",
        beam_width=beam_width,
    )

    # Show best 3 examples
    print("\n" + "=" * 80)
    print("Best 3 Examples (Highest BLEU-4)")
    print("=" * 80)

    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. {result['image_path']} (BLEU-4: {result['bleu_4']:.4f})")
        print(f"   Generated: {result['generated_caption']}")
        print(f"   References:")
        for j, ref in enumerate(result["reference_captions"], 1):
            print(f"     {j}. {ref}")

    # Show worst 3 examples
    print("\n" + "=" * 80)
    print("Worst 3 Examples (Lowest BLEU-4)")
    print("=" * 80)

    for i, result in enumerate(results[-3:], 1):
        print(f"\n{i}. {result['image_path']} (BLEU-4: {result['bleu_4']:.4f})")
        print(f"   Generated: {result['generated_caption']}")
        print(f"   References:")
        for j, ref in enumerate(result["reference_captions"], 1):
            print(f"     {j}. {ref}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nGreedy vs Beam Search:")
    print(f"  BLEU-1: {metrics_greedy['BLEU-1']:.4f} vs {metrics_beam['BLEU-1']:.4f}")
    print(f"  BLEU-2: {metrics_greedy['BLEU-2']:.4f} vs {metrics_beam['BLEU-2']:.4f}")
    print(f"  BLEU-3: {metrics_greedy['BLEU-3']:.4f} vs {metrics_beam['BLEU-3']:.4f}")
    print(f"  BLEU-4: {metrics_greedy['BLEU-4']:.4f} vs {metrics_beam['BLEU-4']:.4f}")

    improvement = (
        (metrics_beam["BLEU-4"] - metrics_greedy["BLEU-4"])
        / metrics_greedy["BLEU-4"]
        * 100
    )
    print(f"\nBeam search improvement: {improvement:+.2f}%")
    print("=" * 80)

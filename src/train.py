"""Training utilities for image captioning model."""

import argparse
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_vocab_from_dataloader, get_dataloaders
from src.model import ImageCaptioningModel
from src.utils import get_device, load_config, parse_captions_file, set_seed
from src.vocab import Vocabulary


def train_step(
    model: nn.Module,
    images: torch.Tensor,
    captions: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    """
    Perform one training step.

    Args:
        model: Image captioning model.
        images: Input images of shape (batch, 3, 224, 224).
        captions: Caption tokens of shape (seq_len, batch) from collate_fn.
        criterion: Loss function.
        optimizer: Optimizer.
        grad_clip: Maximum gradient norm for clipping.

    Returns:
        Loss value for this step.
    """
    model.train()

    # Get device from model
    device = next(model.parameters()).device

    # Move data to device
    images = images.to(device)
    captions = captions.to(device)

    # Transpose captions from (seq_len, batch) to (batch, seq_len)
    captions = captions.transpose(0, 1)

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    # model expects captions and outputs (batch, seq_len-1, vocab_size)
    outputs = model(images, captions)

    # Targets are all tokens except <SOS> (captions[:, 1:])
    targets = captions[:, 1:]

    # Reshape for CrossEntropyLoss
    # outputs: (batch, seq_len-1, vocab_size) -> (batch * (seq_len-1), vocab_size)
    # targets: (batch, seq_len-1) -> (batch * (seq_len-1))
    batch_size, seq_len, vocab_size = outputs.shape
    outputs = outputs.reshape(-1, vocab_size)
    targets = targets.reshape(-1)

    # Compute loss
    loss = criterion(outputs, targets)

    # Backward pass
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    optimizer.step()

    return loss.item()


def validate_step(
    model: nn.Module,
    images: torch.Tensor,
    captions: torch.Tensor,
    criterion: nn.Module,
) -> float:
    """
    Perform one validation step.

    Args:
        model: Image captioning model.
        images: Input images of shape (batch, 3, 224, 224).
        captions: Caption tokens of shape (seq_len, batch) from collate_fn.
        criterion: Loss function.

    Returns:
        Loss value for this step.
    """
    model.eval()

    # Get device from model
    device = next(model.parameters()).device

    # Move data to device
    images = images.to(device)
    captions = captions.to(device)

    # Transpose captions from (seq_len, batch) to (batch, seq_len)
    captions = captions.transpose(0, 1)

    # Forward pass without gradients
    with torch.no_grad():
        # Forward pass
        outputs = model(images, captions)

        # Targets are all tokens except <SOS>
        targets = captions[:, 1:]

        # Reshape for CrossEntropyLoss
        batch_size, seq_len, vocab_size = outputs.shape
        outputs = outputs.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        # Compute loss
        loss = criterion(outputs, targets)

    return loss.item()


def get_criterion(label_smoothing: float, pad_idx: int = 0) -> nn.Module:
    """
    Create loss function for training.

    Args:
        label_smoothing: Label smoothing value for regularization.
        pad_idx: Padding token index to ignore in loss computation.

    Returns:
        CrossEntropyLoss with label smoothing and padding ignore.
    """
    return nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=label_smoothing,
    )


def get_optimizer(
    model: nn.Module,
    config: dict,
    fine_tune: bool = False,
) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates for encoder/decoder.

    When fine_tune is False, only decoder parameters are trained.
    When fine_tune is True, encoder gets a lower learning rate.

    Args:
        model: Image captioning model with encoder and decoder.
        config: Configuration dictionary with training parameters.
        fine_tune: If True, train encoder with lower LR. If False, only train decoder.

    Returns:
        AdamW optimizer configured for the training stage.
    """
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    if fine_tune:
        # Use different learning rates for encoder and decoder
        encoder_lr = lr * config["training"]["encoder_lr_factor"]

        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.decoder.parameters(), "lr": lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        # Only train decoder parameters
        optimizer = torch.optim.AdamW(
            model.decoder.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    return optimizer


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> float:
    """
    Train model for one epoch.

    Args:
        model: Image captioning model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        grad_clip: Maximum gradient norm for clipping.
        device: Device to run training on.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    model.to(device)

    total_loss = 0.0
    num_batches = len(loader)

    # Create progress bar
    pbar = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (images, captions, _) in enumerate(pbar):
        # Perform training step
        loss = train_step(model, images, captions, criterion, optimizer, grad_clip)

        total_loss += loss

        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss:.4f}"})

    # Calculate average loss
    avg_loss = total_loss / num_batches

    return avg_loss


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Validate model for one epoch.

    Args:
        model: Image captioning model.
        loader: Validation data loader.
        criterion: Loss function.
        device: Device to run validation on.

    Returns:
        Average loss for the epoch.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    num_batches = len(loader)

    # Optional progress bar for validation
    pbar = tqdm(loader, desc="Validation", leave=False)

    for images, captions, _ in pbar:
        # Perform validation step
        loss = validate_step(model, images, captions, criterion)

        total_loss += loss

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss:.4f}"})

    # Calculate average loss
    avg_loss = total_loss / num_batches

    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab: Any,
    config: dict,
    epoch: int,
    loss: float,
    filepath_str: str,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Image captioning model.
        optimizer: Optimizer.
        vocab: Vocabulary object.
        config: Configuration dictionary.
        epoch: Current epoch number.
        loss: Current loss value.
        filepath: Path where checkpoint should be saved.
    """
    filepath = Path(filepath_str)

    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "config": config,
        "epoch": epoch,
        "loss": loss,
    }

    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath_str: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    """
    Load model checkpoint.

    Args:
        filepath: Path to the checkpoint file.
        model: Model to load state dict into.
        optimizer: Optional optimizer to load state dict into.
        device: Device to map checkpoint to.

    Returns:
        Full checkpoint dictionary containing vocab, config, epoch, loss, etc.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    filepath = Path(filepath_str)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Load checkpoint
    # Note: weights_only=False is required to load custom objects like Vocabulary
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load model state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Optionally load optimizer state dict
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")

    return checkpoint


def train(config_path: str) -> str:
    """
    Main training orchestration function.

    Args:
        config_path: Path to configuration file.

    Returns:
        Path to the best model checkpoint.
    """
    print("=" * 80)
    print("Image Captioning Training")
    print("=" * 80)

    # ========== Setup ==========
    print("\n[1/5] Loading configuration and setting up environment...")
    config = load_config(config_path)

    # Set random seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()
    print(f"  Device: {device}")

    # ========== Vocabulary ==========
    print("\n[2/5] Setting up vocabulary...")
    vocab_path = Path("checkpoints/vocab.pkl")

    if vocab_path.exists():
        print(f"  Loading existing vocabulary from {vocab_path}")
        vocab = Vocabulary.load(str(vocab_path))
    else:
        print(f"  Building vocabulary from dataset...")
        captions_df = parse_captions_file(config["data"]["captions_file"])
        vocab = build_vocab_from_dataloader(
            captions_df, config["data"]["freq_threshold"]
        )
        # Save vocabulary
        vocab.save(str(vocab_path))
        print(f"  Vocabulary saved to {vocab_path}")

    print(f"  Vocabulary size: {len(vocab)}")

    # ========== Data Loaders ==========
    print("\n[3/5] Creating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config, vocab)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # ========== Model Setup ==========
    print("\n[4/5] Setting up model...")
    model = ImageCaptioningModel.create_from_config(config, vocab_size=len(vocab))
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = get_criterion(
        label_smoothing=config["training"]["label_smoothing"],
        pad_idx=0,
    )

    # Optimizer (initially decoder only)
    optimizer = get_optimizer(model, config, fine_tune=False)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
    )

    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"  Initial learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")

    # ========== Training Loop ==========
    print("\n[5/5] Starting training...")
    print("=" * 80)

    num_epochs = config["training"]["epochs"]
    unfreeze_encoder_epoch = config["training"]["unfreeze_encoder_epoch"]
    early_stopping_patience = config["training"]["early_stopping_patience"]
    grad_clip = config["training"]["grad_clip"]

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = Path("checkpoints/best_model.pt")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 80)

        # Unfreeze encoder at specified epoch
        if epoch == unfreeze_encoder_epoch:
            print(f"\n  >>> Unfreezing encoder at epoch {epoch}")
            model.fine_tune_encoder(enable=True)

            # Recreate optimizer with differential learning rates
            optimizer = get_optimizer(model, config, fine_tune=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=3,
                factor=0.5,
            )
            print(
                f"  >>> Optimizer recreated with encoder LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            print(f"  >>> Decoder LR: {optimizer.param_groups[1]['lr']:.2e}")

        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, grad_clip, device
        )

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        if len(optimizer.param_groups) > 1:
            decoder_lr = optimizer.param_groups[1]["lr"]
            lr_str = f"encoder={current_lr:.2e}, decoder={decoder_lr:.2e}"
        else:
            lr_str = f"{current_lr:.2e}"

        # Print epoch summary
        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {lr_str}")

        # Check for improvement
        if val_loss < best_val_loss:
            print(f"  ✓ Val loss improved ({best_val_loss:.4f} → {val_loss:.4f})")
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                vocab=vocab,
                config=config,
                epoch=epoch,
                loss=val_loss,
                filepath_str=str(best_model_path),
            )
        else:
            patience_counter += 1
            print(
                f"  ✗ No improvement (patience: {patience_counter}/{early_stopping_patience})"
            )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n  Early stopping triggered after {epoch} epochs")
                break

    # ========== End ==========
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print("=" * 80)

    return str(best_model_path)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train image captioning model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Run training
    best_model_path = train(args.config)

"""Training utilities for image captioning model."""

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Training Utilities")
    print("=" * 70)

    # Create dummy model (simplified)
    class DummyModel(nn.Module):
        def __init__(self, vocab_size=1000):
            super().__init__()
            self.encoder = nn.Linear(224 * 224 * 3, 512)
            self.decoder = nn.Linear(512, vocab_size)
            self.vocab_size = vocab_size

        def forward(self, images, captions):
            batch_size = images.shape[0]
            seq_len = captions.shape[1] - 1  # Exclude last token

            # Dummy forward pass
            features = self.encoder(images.flatten(1))
            outputs = self.decoder(features).unsqueeze(1).expand(-1, seq_len, -1)
            return outputs

    # Create dummy data
    batch_size = 4
    seq_len = 10
    vocab_size = 1000

    model = DummyModel(vocab_size=vocab_size)
    images = torch.randn(batch_size, 3, 224, 224)

    # Captions in collate_fn format: (seq_len, batch)
    captions = torch.randint(1, vocab_size, (seq_len, batch_size))

    print(f"\nDummy model created")
    print(f"  Images shape: {images.shape}")
    print(f"  Captions shape (collate_fn format): {captions.shape}")

    # Test get_criterion
    criterion = get_criterion(label_smoothing=0.1, pad_idx=0)
    print(f"\nCriterion created: {type(criterion).__name__}")
    print(f"  Label smoothing: 0.1")
    print(f"  Ignore index: 0 (PAD)")

    # Test get_optimizer
    dummy_config = {
        "training": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "encoder_lr_factor": 0.1,
        }
    }

    optimizer = get_optimizer(model, dummy_config, fine_tune=False)
    print(f"\nOptimizer created (decoder only): {type(optimizer).__name__}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Weight decay: {optimizer.param_groups[0]['weight_decay']}")

    optimizer_ft = get_optimizer(model, dummy_config, fine_tune=True)
    print(f"\nOptimizer created (fine-tune): {type(optimizer_ft).__name__}")
    print(f"  Encoder LR: {optimizer_ft.param_groups[0]['lr']}")
    print(f"  Decoder LR: {optimizer_ft.param_groups[1]['lr']}")

    # Test train_step
    print("\n" + "=" * 70)
    print("Testing train_step")
    print("=" * 70)

    loss = train_step(model, images, captions, criterion, optimizer, grad_clip=5.0)
    print(f"\nTrain step completed")
    print(f"  Loss: {loss:.4f}")
    print(f"  Loss type: {type(loss)}")
    assert isinstance(loss, float), "Loss should be a float"
    print("✓ train_step returns float loss!")

    # Test validate_step
    print("\n" + "=" * 70)
    print("Testing validate_step")
    print("=" * 70)

    val_loss = validate_step(model, images, captions, criterion)
    print(f"\nValidation step completed")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Loss type: {type(val_loss)}")
    assert isinstance(val_loss, float), "Loss should be a float"
    print("✓ validate_step returns float loss!")

    # Verify gradient clipping works
    print("\n" + "=" * 70)
    print("Testing gradient clipping")
    print("=" * 70)

    # Run a step with very small clipping
    optimizer.zero_grad()
    loss = train_step(model, images, captions, criterion, optimizer, grad_clip=0.001)
    print(f"Train step with grad_clip=0.001 completed")
    print(f"  Loss: {loss:.4f}")
    print("✓ Gradient clipping works!")

    # Test train_epoch and validate_epoch
    print("\n" + "=" * 70)
    print("Testing train_epoch and validate_epoch")
    print("=" * 70)

    # Create a dummy dataloader
    from torch.utils.data import TensorDataset

    # Create multiple batches of data
    num_samples = 16
    dataset_images = torch.randn(num_samples, 3, 224, 224)
    dataset_captions = torch.randint(1, vocab_size, (seq_len, num_samples))

    # Create dataset with image, caption pairs
    # Note: we need to transpose captions for the dataset
    dataset = TensorDataset(
        dataset_images,
        dataset_captions.transpose(0, 1),  # Store as (batch, seq_len)
    )

    # Custom collate to match our format
    def dummy_collate(batch):
        images = torch.stack([item[0] for item in batch])
        captions = torch.stack([item[1] for item in batch])
        # Transpose to (seq_len, batch) to match collate_fn
        captions = captions.transpose(0, 1)
        image_names = [f"img_{i}.jpg" for i in range(len(batch))]
        return images, captions, image_names

    dummy_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dummy_collate)

    device = torch.device("cpu")

    # Test train_epoch
    avg_train_loss = train_epoch(
        model, dummy_loader, criterion, optimizer, grad_clip=5.0, device=device
    )
    print(f"\nTrain epoch completed")
    print(f"  Average loss: {avg_train_loss:.4f}")
    assert isinstance(avg_train_loss, float), "Average loss should be float"
    print("✓ train_epoch works!")

    # Test validate_epoch
    avg_val_loss = validate_epoch(model, dummy_loader, criterion, device=device)
    print(f"\nValidation epoch completed")
    print(f"  Average loss: {avg_val_loss:.4f}")
    assert isinstance(avg_val_loss, float), "Average loss should be float"
    print("✓ validate_epoch works!")

    # Test save_checkpoint and load_checkpoint
    print("\n" + "=" * 70)
    print("Testing save_checkpoint and load_checkpoint")
    print("=" * 70)

    # Create a dummy vocabulary object
    class DummyVocab:
        def __init__(self):
            self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
            self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}

    dummy_vocab = DummyVocab()

    # Save checkpoint
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            vocab=dummy_vocab,
            config=dummy_config,
            epoch=5,
            loss=avg_train_loss,
            filepath_str=str(checkpoint_path),
        )

        assert checkpoint_path.exists(), "Checkpoint file should exist"
        print("✓ Checkpoint saved successfully!")

        # Create a new model to load into
        new_model = DummyModel(vocab_size=vocab_size)
        new_optimizer = get_optimizer(new_model, dummy_config, fine_tune=False)

        # Load checkpoint
        loaded_checkpoint = load_checkpoint(
            filepath_str=str(checkpoint_path),
            model=new_model,
            optimizer=new_optimizer,
            device="cpu",
        )

        print(f"\nCheckpoint loaded successfully!")
        print(f"  Loaded epoch: {loaded_checkpoint['epoch']}")
        print(f"  Loaded loss: {loaded_checkpoint['loss']:.4f}")
        print(f"  Vocab in checkpoint: {type(loaded_checkpoint['vocab']).__name__}")

        # Verify the checkpoint contains expected keys
        assert "model_state_dict" in loaded_checkpoint
        assert "optimizer_state_dict" in loaded_checkpoint
        assert "vocab" in loaded_checkpoint
        assert "config" in loaded_checkpoint
        assert loaded_checkpoint["epoch"] == 5
        assert isinstance(loaded_checkpoint["vocab"], DummyVocab)

        print("✓ Checkpoint loaded successfully with all components!")

        # Test loading without optimizer
        another_model = DummyModel(vocab_size=vocab_size)
        loaded_checkpoint_no_opt = load_checkpoint(
            filepath_str=str(checkpoint_path),
            model=another_model,
            optimizer=None,
            device="cpu",
        )
        print("✓ Checkpoint loaded successfully without optimizer!")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

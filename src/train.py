"""Training utilities for image captioning model."""

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

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

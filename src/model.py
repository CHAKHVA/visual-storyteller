"""Model components for image captioning: CNN encoder and Transformer decoder."""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights


class EncoderCNN(nn.Module):
    """CNN Encoder that extracts spatial features from images."""

    def __init__(self, embed_size: int, backbone: str = "resnet101"):
        """
        Initialize the CNN encoder.

        Args:
            embed_size: Dimension of the output embedding.
            backbone: Name of the backbone architecture (default: resnet101).
        """
        super(EncoderCNN, self).__init__()

        self.embed_size = embed_size

        # Load pretrained ResNet101
        if backbone == "resnet101":
            resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the final avgpool and fc layers
        # Keep: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.backbone = nn.Sequential(*modules)

        # Freeze all parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Adaptive average pooling to produce (7, 7) spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Project from ResNet output dimension (2048) to embed_size
        self.projection = nn.Linear(2048, embed_size)

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from images.

        Args:
            images: Input images of shape (batch, 3, 224, 224).

        Returns:
            Encoded features of shape (batch, 49, embed_size).
            49 spatial locations from 7x7 feature map.
        """
        # Pass through backbone: (batch, 3, 224, 224) -> (batch, 2048, H, W)
        features = self.backbone(images)

        # Apply adaptive pooling: (batch, 2048, H, W) -> (batch, 2048, 7, 7)
        features = self.adaptive_pool(features)

        # Flatten spatial dimensions: (batch, 2048, 7, 7) -> (batch, 2048, 49)
        batch_size = features.size(0)
        features = features.view(batch_size, 2048, -1)

        # Permute to: (batch, 49, 2048)
        features = features.permute(0, 2, 1)

        # Project to embed_size: (batch, 49, 2048) -> (batch, 49, embed_size)
        features = self.projection(features)

        # Apply layer normalization
        features = self.layer_norm(features)

        return features

    def fine_tune(self, enable: bool = True) -> None:
        """
        Enable or disable fine-tuning of the encoder.

        When enabled, only layer4 (the last ResNet block) is unfrozen
        to allow gradual fine-tuning.

        Args:
            enable: If True, unfreeze layer4. If False, freeze all parameters.
        """
        if enable:
            # Unfreeze only layer4 (last ResNet block)
            # The backbone is a Sequential of: conv1, bn1, relu, maxpool, layer1-4
            # layer4 is at index 7 (0: conv1, 1: bn1, 2: relu, 3: maxpool, 4: layer1, 5: layer2, 6: layer3, 7: layer4)
            for i, child in enumerate(self.backbone.children()):
                if i == 7:  # layer4
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False


if __name__ == "__main__":
    print("=" * 70)
    print("Testing EncoderCNN")
    print("=" * 70)

    # Create encoder with embed_size=512
    embed_size = 512
    encoder = EncoderCNN(embed_size=embed_size)

    print(f"\nEncoder created with embed_size={embed_size}")
    print(f"Backbone: {type(encoder.backbone).__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"\nParameter count:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")

    # Create dummy input: (batch=2, channels=3, height=224, width=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = encoder(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (2, 49, 512)")

    # Verify output shape
    expected_shape = (2, 49, embed_size)
    assert output.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    )

    print(f"\n✓ Output shape is correct!")

    # Test fine_tune method
    print("\n" + "=" * 70)
    print("Testing fine_tune() method")
    print("=" * 70)

    # Initially all should be frozen
    trainable_before = sum(
        p.numel() for p in encoder.backbone.parameters() if p.requires_grad
    )
    print(f"\nBefore fine_tune(True):")
    print(f"  Trainable backbone params: {trainable_before:,}")

    # Enable fine-tuning
    encoder.fine_tune(enable=True)
    trainable_after = sum(
        p.numel() for p in encoder.backbone.parameters() if p.requires_grad
    )
    print(f"\nAfter fine_tune(True):")
    print(f"  Trainable backbone params: {trainable_after:,}")

    # Disable fine-tuning
    encoder.fine_tune(enable=False)
    trainable_disabled = sum(
        p.numel() for p in encoder.backbone.parameters() if p.requires_grad
    )
    print(f"\nAfter fine_tune(False):")
    print(f"  Trainable backbone params: {trainable_disabled:,}")

    assert trainable_before == 0, "Initially, all backbone params should be frozen"
    assert trainable_after > 0, "After fine_tune(True), some params should be trainable"
    assert trainable_disabled == 0, (
        "After fine_tune(False), all params should be frozen again"
    )

    print("\n✓ fine_tune() method works correctly!")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

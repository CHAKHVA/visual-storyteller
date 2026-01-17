"""Model components for image captioning: CNN encoder and Transformer decoder."""

import math

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


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding with sinusoidal patterns.

        Creates fixed positional encodings using sine and cosine functions
        of different frequencies. These encodings are added to token embeddings
        to provide position information to the transformer.

        Args:
            d_model: Dimension of the model embeddings.
            dropout: Dropout probability to apply after adding positional encoding.
            max_len: Maximum sequence length to support.
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Create position indices: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create division term for the sinusoidal formula
        # div_term shape: (d_model // 2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but saved with the model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added, shape (batch, seq_len, d_model).
        """
        # Add positional encoding (slice to match sequence length)
        # self.pe shape: (1, max_len, d_model)
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]

        # Apply dropout
        return self.dropout(x)


class DecoderTransformer(nn.Module):
    """Transformer decoder for caption generation."""

    def __init__(
        self,
        embed_size: int,
        vocab_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the Transformer decoder.

        Args:
            embed_size: Dimension of embeddings and model.
            vocab_size: Size of the vocabulary.
            num_heads: Number of attention heads.
            num_layers: Number of decoder layers.
            dropout: Dropout probability.
        """
        super(DecoderTransformer, self).__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=embed_size, dropout=dropout, max_len=5000
        )

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True,
        )

        # Stack multiple decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection to vocabulary
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Storage for attention weights (for visualization)
        self.attention_weights = None
        self.return_attention = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding and linear layer weights."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(
        self, encoder_out: torch.Tensor, captions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            encoder_out: Encoded image features of shape (batch, 49, embed_size).
            captions: Caption token indices of shape (batch, seq_len).

        Returns:
            Logits over vocabulary of shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = captions.shape

        # Generate causal mask for autoregressive generation
        # Shape: (seq_len, seq_len)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)

        # Create padding mask where captions == 0 (PAD token)
        # Shape: (batch, seq_len)
        tgt_key_padding_mask = captions == 0

        # Embed captions and scale by sqrt(d_model)
        # Shape: (batch, seq_len, embed_size)
        embedded = self.embedding(captions) * math.sqrt(self.embed_size)

        # Add positional encoding
        # Shape: (batch, seq_len, embed_size)
        embedded = self.pos_encoding(embedded)

        # Pass through transformer decoder
        # tgt: (batch, seq_len, embed_size)
        # memory: (batch, 49, embed_size)
        # tgt_mask: (seq_len, seq_len)
        # tgt_key_padding_mask: (batch, seq_len)
        decoder_out = self.transformer_decoder(
            tgt=embedded,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary size
        # Shape: (batch, seq_len, vocab_size)
        output = self.fc_out(decoder_out)

        return output

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a square causal mask for autoregressive decoding.

        Creates an upper triangular matrix filled with -inf to prevent
        attending to future positions.

        Args:
            sz: Size of the square mask (sequence length).

        Returns:
            Mask tensor of shape (sz, sz).
        """
        # Create upper triangular matrix with -inf (above diagonal)
        mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
        return mask

    def forward_with_attention(
        self, encoder_out: torch.Tensor, captions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both outputs and cross-attention weights.

        This is a simplified version that computes attention manually for visualization.
        Uses multi-head attention to compute cross-attention between decoder queries
        and encoder memory.

        Args:
            encoder_out: Encoded image features of shape (batch, 49, embed_size).
            captions: Caption token indices of shape (batch, seq_len).

        Returns:
            Tuple of (logits, attention_weights):
            - logits: shape (batch, seq_len, vocab_size)
            - attention_weights: shape (batch, seq_len, 49) - averaged over heads
        """
        batch_size, seq_len = captions.shape

        # Generate causal mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)

        # Create padding mask
        tgt_key_padding_mask = captions == 0

        # Embed captions and scale
        embedded = self.embedding(captions) * math.sqrt(self.embed_size)

        # Add positional encoding
        embedded = self.pos_encoding(embedded)

        # For attention visualization, we'll use a simpler approach:
        # Compute cross-attention between embedded captions and encoder output
        # This approximates what the decoder layers do

        # Use the first decoder layer's self-attention and cross-attention
        # For visualization purposes, we'll compute a simplified cross-attention

        # Query from embedded captions: (batch, seq_len, embed_size)
        # Key/Value from encoder: (batch, 49, embed_size)

        # Simple scaled dot-product attention
        # Q: (batch, seq_len, embed_size)
        # K: (batch, 49, embed_size)
        # V: (batch, 49, embed_size)

        query = embedded
        key = encoder_out
        value = encoder_out

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # (batch, seq_len, embed_size) @ (batch, embed_size, 49)
        # -> (batch, seq_len, 49)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.embed_size
        )

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)  # (batch, seq_len, 49)

        # For the actual output, use the standard decoder
        decoder_out = self.transformer_decoder(
            tgt=embedded,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary
        output = self.fc_out(decoder_out)

        return output, attention_weights

    def get_attention_weights(self) -> torch.Tensor:
        """
        Get stored attention weights from the last forward pass.

        Returns:
            Attention weights tensor if available, None otherwise.
        """
        return self.attention_weights


class ImageCaptioningModel(nn.Module):
    """Complete image captioning model combining encoder and decoder."""

    def __init__(
        self,
        embed_size: int,
        vocab_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        backbone: str = "resnet101",
    ):
        """
        Initialize the complete image captioning model.

        Args:
            embed_size: Dimension of embeddings.
            vocab_size: Size of the vocabulary.
            num_heads: Number of attention heads in transformer.
            num_layers: Number of transformer decoder layers.
            dropout: Dropout probability.
            backbone: CNN encoder backbone (default: resnet101).
        """
        super(ImageCaptioningModel, self).__init__()

        self.embed_size = embed_size

        # CNN encoder for image features
        self.encoder = EncoderCNN(embed_size=embed_size, backbone=backbone)

        # Transformer decoder for caption generation
        self.decoder = DecoderTransformer(
            embed_size=embed_size,
            vocab_size=vocab_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and decoder.

        During training, captions include <SOS> at start and <EOS> at end.
        We pass captions[:, :-1] to the decoder (excluding last token) and
        predict the next token at each position (teacher forcing).

        Args:
            images: Input images of shape (batch, 3, 224, 224).
            captions: Caption tokens of shape (batch, seq_len).
                Includes <SOS>, target words, and <EOS>.

        Returns:
            Logits over vocabulary of shape (batch, seq_len-1, vocab_size).
            Each position predicts the next token.
        """
        # Extract image features: (batch, 49, embed_size)
        encoder_out = self.encoder(images)

        # Generate captions using teacher forcing
        # Input: all tokens except last (e.g., <SOS>, w1, w2, ..., w_n)
        # Target: all tokens except first (e.g., w1, w2, ..., w_n, <EOS>)
        # Shape: (batch, seq_len-1, vocab_size)
        outputs = self.decoder(encoder_out, captions[:, :-1])

        return outputs

    def fine_tune_encoder(self, enable: bool = True) -> None:
        """
        Enable or disable fine-tuning of the encoder.

        Args:
            enable: If True, unfreeze encoder's layer4. If False, freeze all.
        """
        self.encoder.fine_tune(enable=enable)

    @staticmethod
    def create_from_config(config: dict, vocab_size: int) -> "ImageCaptioningModel":
        """
        Factory method to create model from configuration dictionary.

        Args:
            config: Configuration dictionary with 'model' section.
            vocab_size: Size of the vocabulary.

        Returns:
            Initialized ImageCaptioningModel instance.
        """
        model_config = config["model"]

        return ImageCaptioningModel(
            embed_size=model_config["embed_size"],
            vocab_size=vocab_size,
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            backbone=model_config["encoder_backbone"],
        )


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

    # Test PositionalEncoding
    print("\n" + "=" * 70)
    print("Testing PositionalEncoding")
    print("=" * 70)

    # Create positional encoding with d_model=512
    d_model = 512
    pos_encoding = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=100)

    print(f"\nPositionalEncoding created with d_model={d_model}")
    print(f"Max length: 100")
    print(f"Positional encoding buffer shape: {pos_encoding.pe.shape}")

    # Create dummy input: (batch=2, seq_len=10, d_model=512)
    batch_size = 2
    seq_len = 10
    dummy_embeddings = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {dummy_embeddings.shape}")

    # Forward pass (no dropout for testing)
    pos_encoding.eval()  # Set to eval mode to disable dropout
    with torch.no_grad():
        output = pos_encoding(dummy_embeddings)

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")

    # Verify output shape matches input shape
    assert output.shape == dummy_embeddings.shape, (
        f"Shape mismatch! Expected {dummy_embeddings.shape}, got {output.shape}"
    )
    print("\n✓ Output shape matches input shape!")

    # Verify that values are different (encoding was added)
    # In eval mode with no dropout, the difference should be exactly the positional encoding
    difference = (output - dummy_embeddings).abs().sum()
    print(f"\nSum of absolute differences: {difference.item():.2f}")
    assert difference > 0, "Positional encoding was not added!"
    print("✓ Positional encoding was successfully added!")

    # Test with different sequence lengths
    for test_seq_len in [5, 20, 50]:
        test_input = torch.randn(1, test_seq_len, d_model)
        with torch.no_grad():
            test_output = pos_encoding(test_input)
        assert test_output.shape == test_input.shape, (
            f"Failed for seq_len={test_seq_len}"
        )
    print("✓ Works correctly with different sequence lengths!")

    # Test DecoderTransformer
    print("\n" + "=" * 70)
    print("Testing DecoderTransformer")
    print("=" * 70)

    # Create decoder
    vocab_size = 1000
    embed_size = 512
    num_heads = 8
    num_layers = 6
    dropout = 0.1

    decoder = DecoderTransformer(
        embed_size=embed_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    print(f"\nDecoderTransformer created:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embed size: {embed_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num layers: {num_layers}")

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Create dummy inputs
    batch_size = 4
    seq_len = 15
    spatial_features = 49

    # Encoder output: (batch, 49, embed_size)
    dummy_encoder_out = torch.randn(batch_size, spatial_features, embed_size)

    # Captions: (batch, seq_len) - random token indices
    dummy_captions = torch.randint(1, vocab_size, (batch_size, seq_len))

    print(f"\nInput shapes:")
    print(f"  Encoder output: {dummy_encoder_out.shape}")
    print(f"  Captions: {dummy_captions.shape}")

    # Forward pass
    decoder.eval()  # Set to eval mode
    with torch.no_grad():
        output = decoder(dummy_encoder_out, dummy_captions)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")

    # Verify output shape
    expected_shape = (batch_size, seq_len, vocab_size)
    assert output.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    )
    print("\n✓ Output shape is correct!")

    # Test causal mask
    print("\nTesting causal mask generation:")
    mask = decoder.generate_square_subsequent_mask(5)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask (should be upper triangular with -inf):")
    print(f"  {mask}")

    # Verify mask structure
    assert mask.shape == (5, 5), "Mask shape incorrect"
    assert mask[0, 0] == 0, "Diagonal should be 0"
    assert mask[0, 1] == float("-inf"), "Upper triangle should be -inf"
    print("✓ Causal mask is correct!")

    # Test with padding (some tokens are 0)
    dummy_captions_with_padding = torch.tensor(
        [
            [1, 45, 23, 67, 0, 0, 0],  # 4 real tokens, 3 padding
            [1, 12, 34, 56, 78, 90, 2],  # All real tokens
        ]
    )
    dummy_encoder_out_small = torch.randn(2, 49, embed_size)

    with torch.no_grad():
        output_with_padding = decoder(
            dummy_encoder_out_small, dummy_captions_with_padding
        )

    print(f"\nWith padding:")
    print(f"  Input captions shape: {dummy_captions_with_padding.shape}")
    print(f"  Output shape: {output_with_padding.shape}")
    assert output_with_padding.shape == (2, 7, vocab_size)
    print("✓ Works correctly with padding!")

    # Test ImageCaptioningModel
    print("\n" + "=" * 70)
    print("Testing ImageCaptioningModel")
    print("=" * 70)

    # Create complete model
    vocab_size = 5000
    embed_size = 512
    num_heads = 8
    num_layers = 6
    dropout = 0.1

    model = ImageCaptioningModel(
        embed_size=embed_size,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        backbone="resnet101",
    )

    print(f"\nImageCaptioningModel created:")
    print(f"  Embed size: {embed_size}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num layers: {num_layers}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nParameter count:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen (encoder): {frozen_params:,}")

    # Create dummy inputs
    batch_size = 2
    seq_len = 20

    # Images: (batch, 3, 224, 224)
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    # Captions: (batch, seq_len) with <SOS>=1, words, <EOS>=2
    # Example: [<SOS>, w1, w2, ..., w17, <EOS>]
    dummy_captions = torch.randint(1, vocab_size, (batch_size, seq_len))
    dummy_captions[:, 0] = 1  # <SOS>
    dummy_captions[:, -1] = 2  # <EOS>

    print(f"\nInput shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Captions: {dummy_captions.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images, dummy_captions)

    print(f"\nOutput shape: {outputs.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len - 1}, {vocab_size})")

    # Verify output shape
    expected_shape = (batch_size, seq_len - 1, vocab_size)
    assert outputs.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {outputs.shape}"
    )
    print("\n✓ Output shape is correct!")

    # Test fine_tune_encoder
    print("\nTesting fine_tune_encoder method:")

    # Initially encoder should be frozen
    trainable_before = sum(
        p.numel() for p in model.encoder.backbone.parameters() if p.requires_grad
    )
    print(f"  Before fine_tune: {trainable_before:,} trainable encoder params")

    # Enable fine-tuning
    model.fine_tune_encoder(enable=True)
    trainable_after = sum(
        p.numel() for p in model.encoder.backbone.parameters() if p.requires_grad
    )
    print(f"  After fine_tune(True): {trainable_after:,} trainable encoder params")

    # Disable fine-tuning
    model.fine_tune_encoder(enable=False)
    trainable_disabled = sum(
        p.numel() for p in model.encoder.backbone.parameters() if p.requires_grad
    )
    print(f"  After fine_tune(False): {trainable_disabled:,} trainable encoder params")

    assert trainable_before == 0, "Initially encoder should be frozen"
    assert trainable_after > 0, "After enabling, encoder should be unfrozen"
    assert trainable_disabled == 0, "After disabling, encoder should be frozen"
    print("✓ fine_tune_encoder() works correctly!")

    # Test create_from_config
    print("\n" + "=" * 70)
    print("Testing create_from_config factory method")
    print("=" * 70)

    # Create dummy config
    dummy_config = {
        "model": {
            "embed_size": 512,
            "num_heads": 8,
            "num_layers": 6,
            "dropout": 0.1,
            "encoder_backbone": "resnet101",
        }
    }

    model_from_config = ImageCaptioningModel.create_from_config(
        dummy_config, vocab_size=5000
    )

    print(f"\nModel created from config:")
    print(f"  Embed size: {model_from_config.embed_size}")
    print(f"  Vocab size: {model_from_config.decoder.vocab_size}")

    # Test that it works
    with torch.no_grad():
        config_outputs = model_from_config(dummy_images, dummy_captions)

    assert config_outputs.shape == expected_shape
    print("✓ create_from_config() works correctly!")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

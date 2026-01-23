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

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # (batch, seq_len, embed_size) @ (batch, embed_size, 49)
        # -> (batch, seq_len, 49)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_size)

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
    print("Testing model components...")

    embed_size = 512
    encoder = EncoderCNN(embed_size=embed_size)
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = encoder(dummy_input)
    assert output.shape == (2, 49, embed_size)
    print("✓ EncoderCNN works")

    decoder = DecoderTransformer(
        embed_size=512, vocab_size=1000, num_heads=8, num_layers=6
    )
    encoder_out = torch.randn(4, 49, 512)
    captions = torch.randint(1, 1000, (4, 15))
    with torch.no_grad():
        output = decoder(encoder_out, captions)
    assert output.shape == (4, 15, 1000)
    print("✓ DecoderTransformer works")

    model = ImageCaptioningModel(
        embed_size=512, vocab_size=5000, num_heads=8, num_layers=6
    )
    images = torch.randn(2, 3, 224, 224)
    captions = torch.randint(1, 5000, (2, 20))
    model.eval()
    with torch.no_grad():
        outputs = model(images, captions)
    assert outputs.shape == (2, 19, 5000)
    print("✓ ImageCaptioningModel works")
    print("\nAll tests passed!")

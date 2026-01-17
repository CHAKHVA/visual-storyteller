"""Dataset class for Flickr8k image captioning."""

import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms


class FlickrDataset(Dataset):
    """Dataset for Flickr8k image captioning."""

    def __init__(
        self,
        image_dir: str,
        captions_df: pd.DataFrame,
        vocab: Any,
        transform: Any = None,
    ):
        """
        Initialize the Flickr dataset.

        Args:
            image_dir: Directory containing the image files.
            captions_df: DataFrame with 'image' and 'caption' columns.
                Each image can have multiple captions.
            vocab: Vocabulary instance for text processing.
            transform: Optional transform to apply to images.
        """
        self.image_dir = Path(image_dir)
        self.captions_df = captions_df
        self.vocab = vocab
        self.transform = transform

        # Get unique image list
        self.images = captions_df["image"].unique().tolist()

        # Create mapping: image_name -> list of captions
        # Group captions by image name
        self.image_to_captions = (
            captions_df.groupby("image")["caption"].apply(list).to_dict()
        )

    def __len__(self) -> int:
        """
        Return the number of unique images in the dataset.

        Returns:
            Number of unique images.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get an image and caption by index.

        Randomly selects one caption from the available captions for variety.
        Numericalizes the caption and adds special tokens.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            Tuple of (image_tensor, caption_tensor, image_name).
            caption_tensor includes <SOS> at start and <EOS> at end.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
        """
        # Get image filename
        image_name = self.images[idx]
        image_path = self.image_dir / image_name

        # Check if image exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Randomly select one caption from available captions
        captions = self.image_to_captions[image_name]
        caption = random.choice(captions)

        # Numericalize the caption
        caption_indices = self.vocab.numericalize(caption)

        # Prepend <SOS> and append <EOS> tokens
        caption_tensor = torch.tensor(
            [self.vocab.stoi["<SOS>"]] + caption_indices + [self.vocab.stoi["<EOS>"]],
            dtype=torch.long,
        )

        return image, caption_tensor, image_name


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Custom collate function to pad captions to the same length in a batch.

    Args:
        batch: List of tuples (image, caption, image_name) from dataset.

    Returns:
        Tuple of:
        - images: Stacked image tensor of shape (B, C, H, W)
        - captions: Padded caption tensor of shape (max_len, B)
        - image_names: List of image filenames
    """
    # Unzip batch into separate lists
    images, captions, image_names = zip(*batch)

    # Stack images into a single tensor (B, C, H, W)
    images = torch.stack(images, dim=0)

    # Pad captions to the same length
    # pad_sequence expects list of tensors and pads along the first dimension
    # Output shape: (max_len, B)
    captions = pad_sequence(captions, batch_first=False, padding_value=0)

    return images, captions, list(image_names)


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or validation.

    Args:
        train: If True, returns training transforms with augmentation.
               If False, returns validation transforms without augmentation.

    Returns:
        Composed torchvision transforms.
    """
    # ImageNet normalization statistics
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if train:
        # Training transforms with data augmentation
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
    else:
        # Validation/test transforms without augmentation
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )


if __name__ == "__main__":
    # Create dummy data
    print("=" * 50)
    print("Testing FlickrDataset")
    print("=" * 50)

    # Create a dummy dataframe with multiple captions per image
    dummy_df = pd.DataFrame(
        {
            "image": ["img1.jpg", "img1.jpg", "img2.jpg", "img2.jpg", "img3.jpg"],
            "caption": [
                "a dog in the park",
                "a brown dog playing outside",
                "a cat on a couch",
                "a sleeping cat",
                "a bird flying high in the sky",
            ],
        }
    )

    print(f"\nDummy DataFrame:")
    print(dummy_df)

    # Create dummy vocab with numericalize method
    class DummyVocab:
        def __init__(self):
            self.stoi = {
                "<PAD>": 0,
                "<SOS>": 1,
                "<EOS>": 2,
                "<UNK>": 3,
                "a": 4,
                "dog": 5,
                "cat": 6,
                "bird": 7,
                "in": 8,
                "the": 9,
                "park": 10,
                "on": 11,
            }

        def numericalize(self, text: str) -> list[int]:
            """Convert text to list of indices."""
            tokens = text.lower().split()
            return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

    vocab = DummyVocab()

    # Test transforms
    print("\n" + "=" * 50)
    print("Testing get_transforms()")
    print("=" * 50)

    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    print(f"\nTraining transforms: {train_transform}")
    print(f"\nValidation transforms: {val_transform}")

    # Create dataset (note: this will fail without real images)
    # For testing purposes, we'll just test the initialization
    print("\n" + "=" * 50)
    print("Testing Dataset Initialization")
    print("=" * 50)

    dataset = FlickrDataset(
        image_dir="caption_data/images",
        captions_df=dummy_df,
        vocab=vocab,
        transform=None,  # No transform for now
    )

    print(f"\nDataset length (unique images): {len(dataset)}")
    print(f"Unique images: {dataset.images}")

    print(f"\nImage-to-captions mapping:")
    for img_name, captions in dataset.image_to_captions.items():
        print(f"  {img_name}:")
        for cap in captions:
            print(f"    - {cap}")

    # Test collate_fn with dummy batches
    print("\n" + "=" * 50)
    print("Testing collate_fn()")
    print("=" * 50)

    # Create dummy batch with variable-length captions
    # Simulating what __getitem__ would return
    dummy_image_1 = torch.randn(3, 224, 224)
    dummy_image_2 = torch.randn(3, 224, 224)
    dummy_image_3 = torch.randn(3, 224, 224)

    # Captions with different lengths (including SOS and EOS)
    # Short caption: <SOS> a dog <EOS> = [1, 4, 5, 2]
    dummy_caption_1 = torch.tensor([1, 4, 5, 2], dtype=torch.long)
    # Medium caption: <SOS> a cat on the couch <EOS> = [1, 4, 6, 11, 9, 3, 2]
    dummy_caption_2 = torch.tensor([1, 4, 6, 11, 9, 3, 2], dtype=torch.long)
    # Long caption: <SOS> a bird flying high in the sky <EOS>
    dummy_caption_3 = torch.tensor([1, 4, 7, 3, 3, 8, 9, 3, 2], dtype=torch.long)

    dummy_batch = [
        (dummy_image_1, dummy_caption_1, "img1.jpg"),
        (dummy_image_2, dummy_caption_2, "img2.jpg"),
        (dummy_image_3, dummy_caption_3, "img3.jpg"),
    ]

    print(f"\nDummy batch caption lengths:")
    print(f"  Caption 1: {len(dummy_caption_1)} tokens")
    print(f"  Caption 2: {len(dummy_caption_2)} tokens")
    print(f"  Caption 3: {len(dummy_caption_3)} tokens")

    # Apply collate_fn
    images, captions, image_names = collate_fn(dummy_batch)

    print(f"\nAfter collate_fn:")
    print(f"  Images shape: {images.shape} (expected: [3, 3, 224, 224])")
    print(f"  Captions shape: {captions.shape} (expected: [9, 3] - max_len x batch)")
    print(f"  Image names: {image_names}")

    print(f"\nPadded captions (transposed for readability):")
    print(f"  Shape: {captions.T.shape} (batch x max_len)")
    print(captions.T)

    # Verify padding is correct (should be 0s)
    print(f"\nVerifying padding:")
    print(f"  Caption 1 has {(captions[:, 0] == 0).sum()} padding tokens")
    print(f"  Caption 2 has {(captions[:, 1] == 0).sum()} padding tokens")
    print(f"  Caption 3 has {(captions[:, 2] == 0).sum()} padding tokens")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

    print("\nNote: __getitem__ test skipped (requires actual images)")
    print("      In practice, use real images to test the full pipeline")

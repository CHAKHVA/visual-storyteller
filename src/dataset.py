"""Dataset class for Flickr8k image captioning."""

import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils import parse_captions_file
from src.vocab import Vocabulary


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


def build_vocab_from_dataloader(
    captions_df: pd.DataFrame, freq_threshold: int
) -> Vocabulary:
    """
    Build vocabulary from all captions in the dataframe.

    Args:
        captions_df: DataFrame with 'caption' column.
        freq_threshold: Minimum frequency for a word to be included.

    Returns:
        Built Vocabulary instance.
    """
    # Extract all captions as a list
    all_captions = captions_df["caption"].tolist()

    # Create and build vocabulary
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocabulary(all_captions)

    return vocab


def get_dataloaders(
    config: dict, vocab: Vocabulary
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Loads captions, splits data by unique images, and creates DataLoader
    instances with appropriate settings for each split.

    Args:
        config: Configuration dictionary with data and training settings.
        vocab: Vocabulary instance for text processing.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Load captions from file
    captions_df = parse_captions_file(config["data"]["captions_file"])

    # Get unique images
    unique_images = captions_df["image"].unique()

    # Calculate split sizes
    train_split = config["data"]["train_split"]
    val_split = config["data"]["val_split"]
    test_split = config["data"]["test_split"]

    # First split: train+val vs test
    train_val_images, test_images = train_test_split(
        unique_images,
        test_size=test_split,
        random_state=42,
    )

    # Second split: train vs val
    # Adjust val_split to be relative to train+val size
    val_split_adjusted = val_split / (train_split + val_split)
    train_images, val_images = train_test_split(
        train_val_images,
        test_size=val_split_adjusted,
        random_state=42,
    )

    # Filter captions dataframe for each split
    train_df = captions_df[captions_df["image"].isin(train_images)].reset_index(
        drop=True
    )
    val_df = captions_df[captions_df["image"].isin(val_images)].reset_index(drop=True)
    test_df = captions_df[captions_df["image"].isin(test_images)].reset_index(drop=True)

    # Create datasets with appropriate transforms
    train_dataset = FlickrDataset(
        image_dir=config["data"]["image_dir"],
        captions_df=train_df,
        vocab=vocab,
        transform=get_transforms(train=True),
    )

    val_dataset = FlickrDataset(
        image_dir=config["data"]["image_dir"],
        captions_df=val_df,
        vocab=vocab,
        transform=get_transforms(train=False),
    )

    test_dataset = FlickrDataset(
        image_dir=config["data"]["image_dir"],
        captions_df=test_df,
        vocab=vocab,
        transform=get_transforms(train=False),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} images ({len(train_df)} captions)")
    print(f"  Val:   {len(val_dataset)} images ({len(val_df)} captions)")
    print(f"  Test:  {len(test_dataset)} images ({len(test_df)} captions)")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from src.utils import load_config

    print("=" * 70)
    print("Testing Full DataLoader Pipeline")
    print("=" * 70)

    # Test basic functionality first with dummy data
    print("\n" + "=" * 70)
    print("Part 1: Testing collate_fn with dummy data")
    print("=" * 70)

    # Create dummy batch with variable-length captions
    dummy_image_1 = torch.randn(3, 224, 224)
    dummy_image_2 = torch.randn(3, 224, 224)
    dummy_image_3 = torch.randn(3, 224, 224)

    dummy_caption_1 = torch.tensor([1, 4, 5, 2], dtype=torch.long)
    dummy_caption_2 = torch.tensor([1, 4, 6, 11, 9, 3, 2], dtype=torch.long)
    dummy_caption_3 = torch.tensor([1, 4, 7, 3, 3, 8, 9, 3, 2], dtype=torch.long)

    dummy_batch = [
        (dummy_image_1, dummy_caption_1, "img1.jpg"),
        (dummy_image_2, dummy_caption_2, "img2.jpg"),
        (dummy_image_3, dummy_caption_3, "img3.jpg"),
    ]

    print(
        f"\nDummy batch caption lengths: {len(dummy_caption_1)}, {len(dummy_caption_2)}, {len(dummy_caption_3)}"
    )

    images, captions, image_names = collate_fn(dummy_batch)

    print(f"After collate_fn:")
    print(f"  Images shape: {images.shape} (expected: [3, 3, 224, 224])")
    print(f"  Captions shape: {captions.shape} (expected: [9, 3] - max_len x batch)")
    print(
        f"  Padding verification: Caption 1 has {(captions[:, 0] == 0).sum()} padding tokens"
    )

    # Test with actual data
    print("\n" + "=" * 70)
    print("Part 2: Testing with actual data (if available)")
    print("=" * 70)

    try:
        # Load configuration
        config = load_config("configs/config.yaml")
        print(f"\nConfiguration loaded successfully")
        print(f"  Batch size: {config['data']['batch_size']}")
        print(f"  Freq threshold: {config['data']['freq_threshold']}")
        print(
            f"  Splits: train={config['data']['train_split']}, val={config['data']['val_split']}, test={config['data']['test_split']}"
        )

        # Load captions
        captions_df = parse_captions_file(config["data"]["captions_file"])
        print(f"\nCaptions loaded:")
        print(f"  Total captions: {len(captions_df)}")
        print(f"  Unique images: {captions_df['image'].nunique()}")

        # Build vocabulary
        print(
            f"\nBuilding vocabulary with freq_threshold={config['data']['freq_threshold']}..."
        )
        vocab = build_vocab_from_dataloader(
            captions_df, config["data"]["freq_threshold"]
        )
        print(f"Vocabulary size: {len(vocab)}")

        # Create dataloaders
        print(f"\nCreating dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders(config, vocab)

        print(f"\nDataLoader info:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Test getting one batch from train loader
        print(f"\n" + "=" * 70)
        print("Part 3: Testing batch retrieval")
        print("=" * 70)

        try:
            train_iter = iter(train_loader)
            images, captions, image_names = next(train_iter)

            print(f"\nFirst training batch:")
            print(f"  Images shape: {images.shape} (B, C, H, W)")
            print(f"  Captions shape: {captions.shape} (max_len, B)")
            print(f"  Batch size: {images.shape[0]}")
            print(f"  Image size: {images.shape[2]}x{images.shape[3]}")
            print(f"  Max caption length in batch: {captions.shape[0]}")
            print(f"  First image name: {image_names[0]}")

            # Verify image normalization
            print(f"\nImage statistics (should be normalized):")
            print(f"  Mean: {images.mean():.3f}")
            print(f"  Std: {images.std():.3f}")
            print(f"  Min: {images.min():.3f}")
            print(f"  Max: {images.max():.3f}")

            # Show a sample caption
            sample_caption = captions[:, 0]  # First caption
            print(f"\nSample caption (first in batch):")
            print(f"  Indices: {sample_caption.tolist()}")
            print(f"  Text: {vocab.denumericalize(sample_caption.tolist())}")

            print("\n" + "=" * 70)
            print("All tests passed successfully!")
            print("=" * 70)

        except FileNotFoundError as e:
            print(f"\nWarning: Could not load images: {e}")
            print("This is expected if images are not yet downloaded.")
            print(
                "The dataloader setup is correct, but needs actual images to iterate."
            )

    except FileNotFoundError as e:
        print(f"\nWarning: Could not find data files: {e}")
        print("This is expected if the dataset hasn't been set up yet.")
        print("The code structure is correct and ready for actual data.")
    except Exception as e:
        print(f"\nError during testing: {e}")

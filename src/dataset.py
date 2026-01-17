"""Dataset class for Flickr8k image captioning."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        Get an image by index.

        Currently returns the image and its filename.
        Caption processing will be added later.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            Tuple of (image_tensor, image_name).

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

        return image, image_name


if __name__ == "__main__":
    from torchvision import transforms

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
                "a brown dog playing",
                "a cat on a couch",
                "a sleeping cat",
                "a bird flying",
            ],
        }
    )

    print(f"\nDummy DataFrame:")
    print(dummy_df)

    # Create dummy vocab (simplified)
    class DummyVocab:
        def __init__(self):
            self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    vocab = DummyVocab()

    # Create a simple transform
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Create dataset (note: this will fail without real images)
    # For testing purposes, we'll just test the initialization
    print(f"\nCreating dataset...")
    dataset = FlickrDataset(
        image_dir="caption_data/images",
        captions_df=dummy_df,
        vocab=vocab,
        transform=transform,
    )

    print(f"\nDataset length (unique images): {len(dataset)}")
    print(f"Unique images: {dataset.images}")

    print(f"\nImage-to-captions mapping:")
    for img_name, captions in dataset.image_to_captions.items():
        print(f"  {img_name}:")
        for cap in captions:
            print(f"    - {cap}")

    # Test that transforms are stored correctly
    print(f"\nTransform is set: {dataset.transform is not None}")

    # Note: We can't test __getitem__ without real images
    print(
        f"\nNote: __getitem__ test skipped (requires actual images in {dataset.image_dir})"
    )

    print("\n" + "=" * 50)
    print("Dataset initialization test passed!")
    print("=" * 50)

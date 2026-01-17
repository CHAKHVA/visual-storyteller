"""Utility functions for configuration, device management, and data loading."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Sets seeds for random, numpy, torch (CPU and CUDA), and enables
    deterministic behavior in cuDNN for consistent results across runs.

    Args:
        seed: Random seed value to use across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path_str: str) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    config_path = Path(config_path_str)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA if available, else CPU).

    Returns:
        torch.device: CUDA device if available, otherwise CPU device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_captions_file(filepath_str: str) -> pd.DataFrame:
    """
    Read captions CSV file and return as DataFrame.

    Expects a CSV file with columns 'image' and 'caption'.
    Each image can have multiple caption rows.

    Args:
        filepath: Path to the captions CSV file.

    Returns:
        DataFrame with columns 'image' and 'caption'.

    Raises:
        FileNotFoundError: If the captions file does not exist.
        ValueError: If required columns are missing.
    """
    filepath = Path(filepath_str)
    if not filepath.exists():
        raise FileNotFoundError(f"Captions file not found: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = {"image", "caption"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Captions file must contain columns {required_columns}. "
            f"Found: {set(df.columns)}"
        )

    return df


if __name__ == "__main__":
    # Test load_config with the config file
    config_path = "configs/config.yaml"
    config = load_config(config_path)

    print("Configuration loaded successfully!")
    print("\nData settings:")
    for key, value in config["data"].items():
        print(f"  {key}: {value}")

    print("\nModel settings:")
    for key, value in config["model"].items():
        print(f"  {key}: {value}")

    print("\nTraining settings:")
    for key, value in config["training"].items():
        print(f"  {key}: {value}")

    print("\nInference settings:")
    for key, value in config["inference"].items():
        print(f"  {key}: {value}")

    # Test device detection
    device = get_device()
    print(f"\nDetected device: {device}")

    # Test set_seed
    set_seed(42)
    print("Seed set to 42 for reproducibility")

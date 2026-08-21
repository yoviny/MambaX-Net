"""
Utility functions for improved error handling and validation in nnunet_train.py
These functions should be integrated into the main training script.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import torch


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration parameters and set defaults.

    Args:
        config: Configuration dictionary

    Returns:
        Validated configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = [
        "seed",
        "in_channels",
        "out_channels",
        "deep_supervision",
        "batch_size",
        "lr",
        "epochs",
        "n_fold",
        "patience",
        "scheduler",
    ]

    # Check required fields
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required configuration fields: {missing_fields}")

    # Validate data types and ranges
    validators = {
        "seed": lambda x: isinstance(x, int) and x >= 0,
        "in_channels": lambda x: isinstance(x, int) and x > 0,
        "out_channels": lambda x: isinstance(x, int) and x > 0,
        "batch_size": lambda x: isinstance(x, int) and 0 < x <= 64,
        "lr": lambda x: isinstance(x, (int, float)) and 0 < x <= 1.0,
        "epochs": lambda x: isinstance(x, int) and x > 0,
        "n_fold": lambda x: isinstance(x, int) and x >= 2,
        "patience": lambda x: isinstance(x, int) and x > 0,
        "deep_supervision": lambda x: isinstance(x, bool),
        "bf16": lambda x: isinstance(x, bool) if "bf16" in config else True,
        "custom_patch": lambda x: (
            isinstance(x, bool) if "custom_patch" in config else True
        ),
        "pretrained": lambda x: isinstance(x, bool) if "pretrained" in config else True,
    }

    for field, validator in validators.items():
        if field in config and not validator(config[field]):
            raise ValueError(f"Invalid value for {field}: {config[field]}")

    # Set defaults for optional fields
    defaults = {
        "bf16": False,
        "custom_patch": False,
        "step_size": 30,
        "pretrained": True,
    }

    for field, default_value in defaults.items():
        config.setdefault(field, default_value)

    return config


def create_directories_safely(directories: List[Union[str, Path]]) -> None:
    """
    Safely create directories with proper error handling.

    Args:
        directories: List of directory paths to create

    Raises:
        PermissionError: If unable to create directories
        OSError: If directory creation fails
    """
    for directory in directories:
        dir_path = Path(directory)
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Permission denied creating directory {dir_path}: {e}"
            )
        except OSError as e:
            raise OSError(f"Failed to create directory {dir_path}: {e}")


if __name__ == "__main__":
    # Example usage
    config = {
        "seed": 42,
        "in_channels": 1,
        "out_channels": 3,
        "deep_supervision": True,
        "batch_size": 2,
        "lr": 0.001,
        "epochs": 100,
        "n_fold": 5,
        "patience": 10,
        "scheduler": "poly",
    }

    try:
        validated_config = validate_config(config)
        print("Configuration validated successfully")
    except ValueError as e:
        print(f"Configuration validation failed: {e}")

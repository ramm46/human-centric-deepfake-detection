"""General-purpose helpers: device selection, reproducibility, checkpoints."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils.logger import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU.

    Mirrors the original notebook's device selection exactly:
    ``torch.device("cuda" if torch.cuda.is_available() else "cpu")``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    """Save a model's state dict, creating parent directories as needed."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("Model checkpoint saved to %s", checkpoint_path)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load a state dict into ``model`` in place and return it.

    Raises:
        FileNotFoundError: If the checkpoint does not exist, with a message
            pointing the user to ``train.py``.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run `python train.py` first to produce a trained model."
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint from %s", checkpoint_path)
    return model

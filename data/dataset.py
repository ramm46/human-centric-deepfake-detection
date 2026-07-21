"""Dataset and DataLoader construction.

Wraps ``torchvision.datasets.ImageFolder`` exactly as the notebook did
(binary classification with directories named by class), just without the
duplication of writing the same four lines in every cell.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from config import BATCH_SIZE
from data.transforms import get_eval_transform, get_train_transform
from utils.logger import get_logger

logger = get_logger(__name__)


def build_datasets(train_dir: Path, val_dir: Path):
    """Build train/validation ``ImageFolder`` datasets.

    Args:
        train_dir: Directory containing one subfolder per class for training.
        val_dir: Directory containing one subfolder per class for validation.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    train_dataset = datasets.ImageFolder(str(train_dir), transform=get_train_transform())
    val_dataset = datasets.ImageFolder(str(val_dir), transform=get_eval_transform())

    logger.info("Classes: %s", train_dataset.classes)
    logger.info("Train size: %d", len(train_dataset))
    logger.info("Validation size: %d", len(val_dataset))

    return train_dataset, val_dataset


def build_dataloaders(train_dir: Path, val_dir: Path, batch_size: int = BATCH_SIZE):
    """Build train/validation ``DataLoader`` objects.

    Args:
        train_dir: Directory containing one subfolder per class for training.
        val_dir: Directory containing one subfolder per class for validation.
        batch_size: Batch size for both loaders.

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset).
    """
    train_dataset, val_dataset = build_datasets(train_dir, val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, train_dataset, val_dataset

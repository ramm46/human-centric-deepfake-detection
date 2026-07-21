"""Evaluate a trained checkpoint on a validation/test set.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint-path checkpoints/resnet18_deepfake.pth --data-dir data_root/Validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CLASS_NAMES, DEFAULT_CHECKPOINT_PATH, VAL_DIR
from data.transforms import get_eval_transform
from models.model import build_model
from utils.helpers import get_device, load_checkpoint
from utils.logger import get_logger
from utils.metrics import accuracy, batch_correct_predictions, confusion_counts
from utils.visualization import plot_confusion_matrix

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the deepfake detection model.")
    parser.add_argument("--data-dir", type=Path, default=VAL_DIR, help="Evaluation data directory.")
    parser.add_argument(
        "--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Trained model checkpoint."
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        default=Path("checkpoints/confusion_matrix.png"),
        help="Where to save the confusion matrix plot.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device()

    from torchvision import datasets

    dataset = datasets.ImageFolder(str(args.data_dir), transform=get_eval_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size)

    model = build_model(pretrained=False)
    model = load_checkpoint(model, args.checkpoint_path, device)

    correct = torch.tensor(0)
    total = 0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += batch_correct_predictions(outputs, labels).cpu()
        total += labels.size(0)
        confusion += confusion_counts(outputs.cpu(), labels.cpu(), num_classes=len(CLASS_NAMES))

    acc = accuracy(correct, total)
    logger.info("Evaluation accuracy: %.4f (%d/%d)", acc, correct.item(), total)
    logger.info("Confusion matrix:\n%s", confusion)

    plot_confusion_matrix(confusion.numpy(), CLASS_NAMES, args.confusion_matrix_path)
    logger.info("Confusion matrix plot saved to %s", args.confusion_matrix_path)


if __name__ == "__main__":
    main()

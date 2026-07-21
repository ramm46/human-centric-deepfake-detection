"""Train the deepfake detection model.

Reproduces the final production training run from the research notebook:
ResNet18 (ImageNet-pretrained) with a linear classification head, Adam
optimizer, and the exact hyperparameters below. Usage:

    python train.py
    python train.py --data-dir /path/to/dataset --epochs 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from config import (
    BATCH_SIZE,
    DEFAULT_CHECKPOINT_PATH,
    LEARNING_RATE,
    LOG_EVERY_N_BATCHES,
    NUM_EPOCHS,
    RANDOM_SEED,
    TRAIN_DIR,
    VAL_DIR,
)
from data.dataset import build_dataloaders
from models.model import build_model
from models.trainer import Trainer
from utils.helpers import get_device, set_seed
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the deepfake detection model.")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="Training data directory.")
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR, help="Validation data directory.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Where to save the trained model weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)
    device = get_device()

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
        args.train_dir, args.val_dir, batch_size=args.batch_size
    )

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        log_every_n_batches=LOG_EVERY_N_BATCHES,
    )

    trainer.fit(train_loader, epochs=args.epochs)
    trainer.save(args.checkpoint_path)

    val_accuracy = trainer.evaluate(val_loader)
    logger.info("Post-training validation accuracy: %.4f", val_accuracy)


if __name__ == "__main__":
    main()

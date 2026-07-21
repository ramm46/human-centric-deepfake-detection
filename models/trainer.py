"""Training loop.

``Trainer.fit`` reproduces the final production training loop from the
notebook exactly: per-batch loss accumulation, a progress print every
``LOG_EVERY_N_BATCHES`` batches, and an end-of-epoch loss summary. The
original final run did not compute a per-epoch validation pass inside the
loop (validation was done separately afterward), so this preserves that
behavior. Validation is exposed separately via ``Trainer.evaluate`` for use
by evaluate.py, without changing what train.py actually does.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import get_logger
from utils.metrics import accuracy, batch_correct_predictions

logger = get_logger(__name__)


class Trainer:
    """Encapsulates the training and evaluation loops for the classifier."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        log_every_n_batches: int = 500,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.log_every_n_batches = log_every_n_batches

    def fit(self, train_loader: DataLoader, epochs: int) -> list[float]:
        """Run the training loop for ``epochs`` epochs.

        Args:
            train_loader: DataLoader yielding (images, labels) batches.
            epochs: Number of epochs to train for.

        Returns:
            List of per-epoch running loss totals (for plotting/logging).
        """
        epoch_losses: list[float] = []

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if batch_idx % self.log_every_n_batches == 0:
                    logger.info(
                        "Epoch %d Batch %d Loss %.4f", epoch + 1, batch_idx, loss.item()
                    )

            logger.info("Epoch %d finished. Total loss: %.4f", epoch + 1, running_loss)
            epoch_losses.append(running_loss)

        return epoch_losses

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> float:
        """Compute classification accuracy over a full DataLoader.

        Args:
            data_loader: DataLoader yielding (images, labels) batches.

        Returns:
            Accuracy as a float in [0, 1].
        """
        self.model.eval()
        correct = torch.tensor(0)
        total = 0

        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            correct += batch_correct_predictions(outputs, labels).cpu()
            total += labels.size(0)

        return accuracy(correct, total)

    def save(self, checkpoint_path: Path) -> None:
        """Save the current model weights to ``checkpoint_path``."""
        from utils.helpers import save_checkpoint

        save_checkpoint(self.model, checkpoint_path)

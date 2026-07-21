"""Metric computation helpers.

These wrap the same accuracy arithmetic that was inlined in the notebook's
training loops (``correct / total``) so it isn't duplicated across
train.py, evaluate.py, and models/trainer.py.
"""

from __future__ import annotations

import torch


def batch_correct_predictions(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Count correct predictions in a batch given raw model logits.

    Args:
        outputs: Raw model output of shape (batch, num_classes).
        labels: Ground-truth class indices of shape (batch,).

    Returns:
        Scalar tensor with the number of correct predictions in the batch.
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels)


def accuracy(correct: torch.Tensor, total: int) -> float:
    """Compute accuracy as a plain float in [0, 1]."""
    return (correct.double() / total).item()


def confusion_counts(
    outputs: torch.Tensor, labels: torch.Tensor, num_classes: int = 2
) -> torch.Tensor:
    """Return a (num_classes x num_classes) confusion matrix for one batch.

    Rows are true classes, columns are predicted classes.
    """
    _, preds = torch.max(outputs, dim=1)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_label, pred_label in zip(labels.view(-1), preds.view(-1)):
        matrix[true_label.long(), pred_label.long()] += 1
    return matrix

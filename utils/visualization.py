"""Plotting helpers for training curves and evaluation reports.

Purely additive tooling — not part of the original notebook's core logic,
and never called from the prediction/inference path, so it cannot affect
model outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curve(losses: list[float], output_path: Path) -> None:
    """Save a simple line plot of per-epoch training loss."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(matrix, class_names: list[str], output_path: Path) -> None:
    """Save a heatmap of a confusion matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

"""Image preprocessing pipelines.

IMPORTANT: The production checkpoint was trained with a single transform —
``Resize(224, 224)`` followed by ``ToTensor()`` — applied identically to
train, validation, and inference images, with NO normalization step. This
is preserved exactly as-is; changing it would silently break the trained
weights.
"""

from __future__ import annotations

from torchvision import transforms

from config import IMAGE_SIZE


def get_train_transform() -> transforms.Compose:
    """Transform used for training images (identical to eval transform)."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )


def get_eval_transform() -> transforms.Compose:
    """Transform used for validation and test images."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )


def get_inference_transform() -> transforms.Compose:
    """Transform used for single-image inference (predict.py, Streamlit app).

    Must stay identical to ``get_eval_transform`` — this is what the
    checkpoint was trained and validated against.
    """
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )

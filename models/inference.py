"""Single-image inference.

Reproduces the final prediction cells from the notebook exactly: load
image, apply the inference transform, run a forward pass, softmax the
logits, and report whichever class has the higher probability along with
its confidence percentage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from config import CLASS_NAMES
from data.transforms import get_inference_transform


@dataclass
class PredictionResult:
    """Container for a single image prediction."""

    prediction: str
    confidence: float
    fake_probability: float
    real_probability: float


def predict_image(model: torch.nn.Module, image_path: Path, device: torch.device) -> PredictionResult:
    """Run inference on a single image file.

    Args:
        model: A model already loaded with trained weights, in eval mode.
        image_path: Path to a JPEG/PNG image.
        device: Device the model lives on.

    Returns:
        A ``PredictionResult`` with the predicted label and confidence.
    """
    return predict_pil_image(model, Image.open(image_path).convert("RGB"), device)


def predict_pil_image(
    model: torch.nn.Module, image: Image.Image, device: torch.device
) -> PredictionResult:
    """Run inference on an already-loaded PIL image (used by the Streamlit app).

    Args:
        model: A model already loaded with trained weights, in eval mode.
        image: A PIL image, any mode (will be converted to RGB).
        device: Device the model lives on.

    Returns:
        A ``PredictionResult`` with the predicted label and confidence.
    """
    transform = get_inference_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    probabilities = F.softmax(output, dim=1)
    fake_prob = probabilities[0][0].item()
    real_prob = probabilities[0][1].item()

    prediction = CLASS_NAMES[0] if fake_prob > real_prob else CLASS_NAMES[1]
    confidence = max(fake_prob, real_prob) * 100

    return PredictionResult(
        prediction=prediction,
        confidence=confidence,
        fake_probability=fake_prob,
        real_probability=real_prob,
    )

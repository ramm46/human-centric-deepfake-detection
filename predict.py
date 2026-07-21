"""Run a single-image prediction from the command line.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --detailed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_CHECKPOINT_PATH
from explainability.explanation import detailed_explanation, rule_based_explanation
from models.inference import predict_image
from models.model import build_model
from utils.helpers import get_device, load_checkpoint
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether an image is real or a deepfake.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Trained model checkpoint."
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show the tiered risk-level explanation instead of the default short one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    model = build_model(pretrained=False)
    model = load_checkpoint(model, args.checkpoint_path, device)

    result = predict_image(model, args.image, device)

    print(f"Fake probability: {result.fake_probability:.4f}")
    print(f"Real probability: {result.real_probability:.4f}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2f}%")

    if args.detailed:
        risk_level, explanation = detailed_explanation(result.prediction, result.confidence)
        print(f"Risk Level: {risk_level}")
        print(f"Explanation: {explanation}")
    else:
        print(f"\nExplanation:\n{rule_based_explanation(result.prediction, result.confidence)}")


if __name__ == "__main__":
    main()

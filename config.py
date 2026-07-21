"""Central configuration for the deepfake detection project.

All paths, hyperparameters, and constants live here so that training,
evaluation, inference, and the Streamlit app stay in sync. Values mirror
the final production training run from the original research notebook —
nothing here changes model behavior, it only centralizes what used to be
duplicated across notebook cells.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT_PATH: Path = CHECKPOINT_DIR / "resnet18_deepfake.pth"

# Dataset root is environment-configurable so the project isn't tied to any
# single machine's folder layout (the original notebook hardcoded Google
# Drive paths). Falls back to a local ./data directory.
DATA_DIR: Path = Path(os.getenv("DEEPFAKE_DATA_DIR", PROJECT_ROOT / "data_root"))
TRAIN_DIR: Path = DATA_DIR / "Train"
VAL_DIR: Path = DATA_DIR / "Validation"

# --------------------------------------------------------------------------
# Model architecture
# --------------------------------------------------------------------------
# Matches the final training run in the source notebook: a plain ResNet18
# backbone with ImageNet weights and a single linear classification head.
MODEL_ARCHITECTURE: str = "resnet18"
PRETRAINED_WEIGHTS: str = "IMAGENET1K_V1"
NUM_CLASSES: int = 2
CLASS_NAMES: list[str] = ["Fake", "Real"]

# --------------------------------------------------------------------------
# Image preprocessing
# --------------------------------------------------------------------------
# The production model was trained WITHOUT normalization — only resize +
# tensor conversion. This is intentional and must match inference exactly.
IMAGE_SIZE: tuple[int, int] = (224, 224)
RANDOM_ROTATION_DEGREES: int = 10

# --------------------------------------------------------------------------
# Training hyperparameters (final production run)
# --------------------------------------------------------------------------
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-4
NUM_EPOCHS: int = 5
LOG_EVERY_N_BATCHES: int = 500

# --------------------------------------------------------------------------
# Explanation / risk thresholds
# --------------------------------------------------------------------------
HIGH_CONFIDENCE_THRESHOLD: float = 85.0
MODERATE_CONFIDENCE_THRESHOLD: float = 70.0

# --------------------------------------------------------------------------
# Optional third-party credentials
# --------------------------------------------------------------------------
# NEVER hardcode API keys. All secrets are read from the environment (a
# .env file works too if you load it with python-dotenv before import).
KAGGLE_USERNAME: str | None = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY: str | None = os.getenv("KAGGLE_KEY")
GOOGLE_GENAI_API_KEY: str | None = os.getenv("GOOGLE_GENAI_API_KEY")
NGROK_AUTHTOKEN: str | None = os.getenv("NGROK_AUTHTOKEN")

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
RANDOM_SEED: int = 42

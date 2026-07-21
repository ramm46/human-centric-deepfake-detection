"""Streamlit UI for the deepfake detection model.

This is a direct, modularized port of the app the original notebook
generated via ``%%writefile app.py``. The UI flow, wording, and behavior
are unchanged: upload an image, run it through the model, show the
prediction, confidence, and explanation.
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

from config import DEFAULT_CHECKPOINT_PATH
from explainability.explanation import rule_based_explanation
from models.inference import predict_pil_image
from models.model import build_model
from utils.helpers import get_device, load_checkpoint


@st.cache_resource
def load_model():
    """Load the trained model once per session (Streamlit caches the result)."""
    device = get_device()
    model = build_model(pretrained=False)
    model = load_checkpoint(model, DEFAULT_CHECKPOINT_PATH, device)
    return model, device


def main() -> None:
    st.title("Deepfake Detection System")

    model, device = load_model()

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        result = predict_pil_image(model, image, device)

        st.subheader(f"Prediction: {result.prediction}")
        st.write(f"Confidence: {result.confidence:.2f}%")

        st.subheader("AI Explanation")
        st.write(rule_based_explanation(result.prediction, result.confidence))


if __name__ == "__main__":
    main()

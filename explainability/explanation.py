"""Prediction explanation logic.

The notebook contains two independently-developed explanation approaches.
Both are preserved here, unchanged in wording and behavior:

1. ``rule_based_explanation`` — the simple two-branch explanation that is
   actually wired into the production Streamlit app. This is the default.
2. ``detailed_explanation`` — a richer three-tier (High/Moderate/Low risk)
   explanation from an earlier notebook cell. It was never connected to
   the final Streamlit app, but is kept available here (e.g. for
   predict.py's CLI output) rather than discarded.

A third, LLM-based explanation (Gemini) was also present in the notebook
as an experiment. It lives in ``llm_explanation.py`` since it has an
external dependency and requires an API key.
"""

from __future__ import annotations

from config import HIGH_CONFIDENCE_THRESHOLD, MODERATE_CONFIDENCE_THRESHOLD


def rule_based_explanation(prediction: str, confidence: float) -> str:
    """Explanation exactly matching the production Streamlit app's ``explain()``.

    Args:
        prediction: Either "Fake" or "Real".
        confidence: Confidence percentage (0-100).

    Returns:
        A short human-readable explanation string.
    """
    if prediction == "Fake":
        return (
            f"This image is predicted as FAKE with {confidence:.2f}% confidence.\n\n"
            "Deepfake images may contain unnatural facial textures,\n"
            "lighting inconsistencies, or blending artifacts."
        )

    return (
        f"This image is predicted as REAL with {confidence:.2f}% confidence.\n\n"
        "The facial features appear natural and consistent."
    )


def detailed_explanation(prediction: str, confidence: float) -> tuple[str, str]:
    """Tiered risk-level explanation from the notebook's standalone analysis cell.

    Args:
        prediction: Either "Fake" or "Real".
        confidence: Confidence percentage (0-100).

    Returns:
        Tuple of (risk_level, explanation_text).
    """
    if confidence > HIGH_CONFIDENCE_THRESHOLD:
        risk_level = "High"
        explanation = (
            "The model detected strong visual patterns associated with "
            "synthetic textures and facial inconsistencies."
        )
    elif confidence > MODERATE_CONFIDENCE_THRESHOLD:
        risk_level = "Moderate"
        explanation = (
            "The model identified some irregular texture patterns, but "
            "confidence is not extremely high."
        )
    else:
        risk_level = "Low"
        explanation = (
            "The prediction confidence is low. The model is uncertain due "
            "to limited distinguishing features."
        )

    return risk_level, explanation

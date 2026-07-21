"""Optional LLM-based explanation (Google Gemini).

This mirrors the experimental LLM explanation from the notebook. It is
NOT used by the default Streamlit app or CLI (which use the rule-based
explanation in ``explanation.py``, matching production behavior exactly).
It's kept here as an opt-in enhancement since the original notebook
invested real work in it.

Requires the ``GOOGLE_GENAI_API_KEY`` environment variable — the key is
never hardcoded. If the dependency isn't installed or the key isn't set,
callers get a clear error rather than a silent failure.
"""

from __future__ import annotations

from config import GOOGLE_GENAI_API_KEY


def llm_explanation(prediction: str, confidence: float) -> str:
    """Generate a natural-language explanation using Gemini.

    Args:
        prediction: Either "Fake" or "Real".
        confidence: Confidence percentage (0-100).

    Returns:
        A generated explanation string. Falls back to a static message
        (identical wording to the notebook's fallback branches) if no
        API key is configured, so callers never crash the app.
    """
    if prediction == "Fake":
        fallback = (
            f"The AI model classified this image as FAKE with {confidence:.2f}% confidence.\n\n"
            "Deepfake images often contain artifacts such as unnatural facial textures,\n"
            "inconsistent lighting, irregular blending around facial edges, or abnormal\n"
            "eye and mouth movements which the model may have detected."
        )
    else:
        fallback = (
            f"The AI model classified this image as REAL with {confidence:.2f}% confidence.\n\n"
            "Real images typically contain natural facial textures, consistent lighting,\n"
            "and normal structural patterns which the model recognized."
        )

    if not GOOGLE_GENAI_API_KEY:
        return fallback

    try:
        from google import genai
    except ImportError:
        return fallback

    try:
        client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)
        prompt = (
            f"The AI model classified this image as {prediction.upper()} with "
            f"{confidence:.2f}% confidence. Briefly explain, in 2-3 sentences, what "
            "visual cues a deepfake detector typically relies on for this kind of "
            "prediction."
        )
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        return response.text or fallback
    except Exception:  # noqa: BLE001 - external API call, degrade gracefully
        return fallback

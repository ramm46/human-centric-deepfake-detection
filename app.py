"""Root Streamlit entrypoint.

Run with:
    streamlit run app.py

This simply delegates to ``streamlit_app/app.py`` so the actual UI logic
lives in one place under ``streamlit_app/`` alongside the rest of the
modular codebase, while still supporting the conventional
``streamlit run app.py`` command from the project root.
"""

from streamlit_app.app import main

if __name__ == "__main__":
    main()

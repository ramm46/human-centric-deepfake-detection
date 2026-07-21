"""Project-wide logging configuration.

The original notebook relied entirely on ``print`` statements. This module
gives every entrypoint (train, evaluate, predict, the Streamlit app) a
consistent, leveled logger without changing what information is surfaced.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, creating handlers only once per name.

    Args:
        name: Usually ``__name__`` of the calling module.
        level: Logging level, defaults to INFO.

    Returns:
        A ready-to-use ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger

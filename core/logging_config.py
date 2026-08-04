"""Centralized logging configuration for the AI Meeting Assistant.

Call configure_logging() once during application startup before importing
project modules that use logging.

Individual modules simply create a module-level logger:

    import logging
    logger = logging.getLogger(__name__)

and rely on the root logging configuration installed here.
"""

import logging
import os

_CONFIGURED = False


def configure_logging() -> None:
    """Configure the application's root logger.

    Safe to call multiple times. The configuration is applied only once
    per process, making it suitable for Streamlit reruns.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Third-party libraries are noisy at INFO/DEBUG; keep them at WARNING
    # so the app's own log lines aren't drowned out.
    for noisy_logger in (
        "httpx",
        "urllib3",
        "chromadb",
        "sentence_transformers",
        "faster_whisper",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _CONFIGURED = True

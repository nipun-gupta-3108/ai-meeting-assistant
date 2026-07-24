"""Centralized logging configuration for the AI Meeting Assistant.

Call configure_logging() once, as early as possible in each entry point
(CLI and Streamlit), before importing any project modules that log at
import time. Individual modules just do:

    import logging
    logger = logging.getLogger(__name__)

and rely on the root configuration installed here.
"""

import logging
import os

_CONFIGURED = False


def configure_logging() -> None:
    """Install a single console handler + format for the whole app.

    Idempotent: safe to call multiple times (e.g. on every Streamlit
    rerun) — only configures handlers once per process.
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

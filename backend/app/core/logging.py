"""Structured logging setup, LOG_LEVEL-driven (PRD Appendix B)."""

from __future__ import annotations

import logging
import sys


def configure_logging(log_level: str) -> None:
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.handlers = [handler]

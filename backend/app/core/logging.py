"""Structured logging setup, LOG_LEVEL-driven (PRD Appendix B).

Every line carries the request id it was logged under (core/request_context.py), "-" outside a request. LOG_FORMAT=json
writes one JSON object per line for a log collector; the default is plain text.
"""

from __future__ import annotations

import logging
import sys

from app.core.request_context import JsonFormatter, RequestIdFilter


def configure_logging(log_level: str, log_format: str = "text") -> None:
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    if log_format.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
        )
    root.handlers = [handler]

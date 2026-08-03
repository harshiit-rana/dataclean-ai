"""Non-UTF-8 CSV detection and conversion. PRD FR-04, error table §14.1."""

from __future__ import annotations

import logging

import chardet
import pandas as pd

from app.core.errors import EmptyFileError, HeaderOnlyFileError, UnparseableCSVError

logger = logging.getLogger(__name__)


def detect_encoding(raw_bytes: bytes) -> str:
    result = chardet.detect(raw_bytes)
    encoding = result.get("encoding") or "utf-8"
    logger.info("Detected encoding %s (confidence %.2f)", encoding, result.get("confidence", 0))
    return encoding


def read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    """Decodes and parses uploaded CSV bytes, auto-detecting non-UTF-8 encodings (FR-04)
    and raising the plain-English errors from PRD §14.1 for the remaining failure modes."""
    if len(raw_bytes) == 0:
        raise EmptyFileError()

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(raw_bytes)
        try:
            text = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError, TypeError) as exc:
            raise UnparseableCSVError() from exc

    from io import StringIO

    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise UnparseableCSVError() from exc

    if df.shape[1] == 0:
        raise UnparseableCSVError()
    if df.shape[0] == 0:
        raise HeaderOnlyFileError()

    return df

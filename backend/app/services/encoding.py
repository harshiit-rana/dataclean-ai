"""Non-UTF-8 CSV detection and conversion. PRD FR-04, error table §14.1."""

from __future__ import annotations

import logging

import chardet
import pandas as pd

from app.core.errors import EmptyFileError, FileShapeTooLargeError, HeaderOnlyFileError, UnparseableCSVError

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

    from app.config import get_settings

    settings = get_settings()
    _check_header(text, settings.max_columns)
    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise UnparseableCSVError() from exc

    if df.shape[1] == 0:
        raise UnparseableCSVError()
    if df.shape[0] == 0:
        raise HeaderOnlyFileError()
    _check_shape(df, settings.max_rows, settings.max_cell_chars)

    return df


def _check_header(text: str, max_columns: int) -> None:
    """Refuses a file with too many columns before it is parsed: a wide file costs memory in the parse itself."""
    import csv
    from io import StringIO

    first_line = text.split("\n", 1)[0]
    try:
        header = next(csv.reader(StringIO(first_line)), [])
    except csv.Error:
        return  # left to the parser, which reports an unreadable file in its own words
    if len(header) > max_columns:
        raise FileShapeTooLargeError(
            f"The file has {len(header):,} columns; the limit is {max_columns:,}. Keep only the columns you need, "
            f"or raise MAX_COLUMNS if this server has the memory for it."
        )


def _check_shape(df: pd.DataFrame, max_rows: int, max_cell_chars: int) -> None:
    """Refuses a file with too many rows or a cell longer than any real value, straight after parsing."""
    if len(df) > max_rows:
        raise FileShapeTooLargeError(
            f"The file has {len(df):,} rows; the limit is {max_rows:,}. Split it, or raise MAX_ROWS if this server "
            f"has the memory for it."
        )
    for column in df.columns:
        series = df[column]
        if series.dtype != object:
            continue
        lengths = series.dropna().astype(str).str.len()
        if len(lengths) and int(lengths.max()) > max_cell_chars:
            row = int(lengths.idxmax()) + 1
            raise FileShapeTooLargeError(
                f"A cell in column '{column}' (row {row:,}) is {int(lengths.max()):,} characters long; the limit is "
                f"{max_cell_chars:,}. A value that long is usually a file that was not split into rows correctly."
            )

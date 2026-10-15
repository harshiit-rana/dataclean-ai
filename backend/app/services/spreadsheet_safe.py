"""CSV downloads that a spreadsheet cannot run as formulas.

Cleaned files and predictions are CSVs people open in Excel or Google Sheets, and a cell from the uploaded data that
begins with `=`, `+`, `-`, `@`, a tab or a carriage return is read there as a formula: `=HYPERLINK(...)` becomes a
link that sends data somewhere, `=cmd|...` worse (CSV or formula injection). Such cells are written with a leading
single quote, the OWASP guidance, which a spreadsheet shows as the text it was.

A cell that is a plain number is left alone wherever it is, so a negative number stays a number: `-5` is a value
to a spreadsheet, not a formula that can do anything. Only on the way out: the files the app reads itself -- the
session's cleaned.csv that models are trained on, the outputs of an application -- stay exactly as the data was,
and are escaped when served for download.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi.responses import Response

FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r")


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def risky(value) -> bool:
    """Whether a cell would be read by a spreadsheet as a formula."""
    return isinstance(value, str) and value.startswith(FORMULA_LEADERS) and not _is_number(value)


def neutralise(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A copy with every formula-leading text cell prefixed with a quote, and how many were."""
    out = frame.copy()
    count = 0
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        flagged = series.map(risky).astype(bool)
        if flagged.any():
            count += int(flagged.sum())
            out[column] = series.where(~flagged, "'" + series.astype(str))
    return out, count


def count_risky(frame: pd.DataFrame) -> int:
    """How many cells a download would neutralise."""
    return neutralise(frame)[1]


def csv_bytes(path: Path) -> bytes:
    """A stored CSV as download bytes, neutralised. Read as text, so every other value is written back exactly as
    it was stored."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    safe, _ = neutralise(frame)
    buffer = io.StringIO()
    safe.to_csv(buffer, index=False, lineterminator="\n")  # the same file on every platform
    return buffer.getvalue().encode("utf-8")


def csv_download(path: Path, filename: str) -> Response:
    """A stored CSV served for download, neutralised. Every CSV download goes through here."""
    return Response(
        csv_bytes(path),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

"""GET /api/preview/{session_id} -- REAL.

Serves the uploaded CSV's actual contents so the Review page can render the data visually
alongside the suggestion cards. Reads straight from the stored file and meta sidecar --
deliberately independent of the graph/checkpointer, so the table still renders even if a
profiling run had trouble.
"""

from __future__ import annotations

import math

import pandas as pd
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.errors import SessionNotFoundError, UnparseableCSVError
from app.models.api_schemas import PreviewResponse
from app.services.session_store import read_session_meta, session_dir

router = APIRouter()

PREVIEW_ROW_LIMIT = 100  # enough to eyeball the data; keeps the payload small


def _jsonable(value):
    """NaN/Inf are valid float values to pandas but not to JSON, and pandas' NA sentinels
    aren't Python natives -- normalize everything to None or a plain scalar."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar -> python scalar
        return value.item()
    return value


@router.get("/api/preview/{session_id}", response_model=PreviewResponse)
async def get_preview(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> PreviewResponse:
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    file_path = session_dir(settings.upload_dir, session_id) / "original.csv"
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise UnparseableCSVError() from exc

    preview = df.head(PREVIEW_ROW_LIMIT)
    rows = [{col: _jsonable(value) for col, value in record.items()} for record in preview.to_dict(orient="records")]

    goal_context = meta.get("goal_context") or {}
    critical_columns = goal_context.get("critical_columns", [])

    columns = [
        {
            "name": str(col),
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique(dropna=True)),
            "is_critical": str(col) in critical_columns,
        }
        for col in df.columns
    ]

    return PreviewResponse(
        session_id=session_id,
        columns=columns,
        rows=rows,
        total_rows=int(len(df)),
        total_columns=int(df.shape[1]),
        preview_row_count=len(rows),
    )

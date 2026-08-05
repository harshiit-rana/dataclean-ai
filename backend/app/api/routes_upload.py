"""POST /api/upload -- REAL. PRD FR-01..09, NFR-01, NFR-12, NFR-13."""

from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile

from app.config import Settings, get_settings
from app.core.errors import FileTooLargeError, UnsupportedFileTypeError
from app.models.api_schemas import UploadResponse
from app.services.encoding import read_csv_bytes
from app.services.session_access import META_KEY, issue_token
from app.services.session_store import new_session_id, save_uploaded_csv, write_session_meta

router = APIRouter()


@router.post("/api/upload", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise UnsupportedFileTypeError()

    raw_bytes = await file.read()
    if len(raw_bytes) > settings.max_file_size_bytes:
        raise FileTooLargeError(settings.max_file_size_mb)

    df = read_csv_bytes(raw_bytes)  # raises EmptyFileError / UnparseableCSVError / HeaderOnlyFileError

    session_id = new_session_id()
    save_uploaded_csv(settings.upload_dir, session_id, raw_bytes)

    # Issued unconditionally, enforced only when configured. Handing it out either way means
    # turning enforcement on is a config change rather than a migration.
    token, token_hash = issue_token()

    row_count, column_count = df.shape
    write_session_meta(
        settings.upload_dir,
        session_id,
        {
            "file_name": file.filename,
            "file_size_bytes": len(raw_bytes),
            "row_count": row_count,
            "column_count": column_count,
            "goal_context": None,
            META_KEY: token_hash,  # the hash only -- the token itself is never stored
        },
    )

    return UploadResponse(
        session_id=session_id,
        session_token=token,
        file_name=file.filename,
        row_count=row_count,
        column_count=column_count,
        columns=[str(col) for col in df.columns],
    )

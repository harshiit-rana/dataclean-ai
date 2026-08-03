"""App-level exceptions mapped to the {error, detail} response shape (PRD §10.3),
covering the error table in PRD §14."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.models.api_schemas import ErrorResponse


class AppError(Exception):
    """Base for all errors that should surface as a plain-English {error, detail} response."""

    status_code = 400

    def __init__(self, error: str, detail: str):
        self.error = error
        self.detail = detail
        super().__init__(detail)


# --- File upload errors (PRD §14.1) ------------------------------------------------------


class UnsupportedFileTypeError(AppError):
    def __init__(self):
        super().__init__("Unsupported file type", "Only CSV files are supported.")


class FileTooLargeError(AppError):
    def __init__(self, max_mb: int):
        super().__init__("File too large", f"File exceeds the {max_mb}MB limit.")


class UnparseableCSVError(AppError):
    def __init__(self):
        super().__init__("File could not be read as CSV", "File could not be read as CSV.")


class EmptyFileError(AppError):
    def __init__(self):
        super().__init__("Empty file", "File appears to be empty.")


class HeaderOnlyFileError(AppError):
    def __init__(self):
        super().__init__("No data rows", "File has no data rows.")


# --- Session errors (PRD §14.3) -----------------------------------------------------------


class SessionNotFoundError(AppError):
    status_code = 404

    def __init__(self, session_id: str):
        super().__init__("Session expired", f"Session {session_id} was not found. It may have expired.")


class CheckpointCorruptedError(AppError):
    status_code = 500

    def __init__(self):
        super().__init__("Session could not be restored", "The saved session state is corrupted. Please restart the session.")


class RedisUnavailableError(AppError):
    status_code = 503

    def __init__(self):
        super().__init__("Service temporarily unavailable", "Session storage is temporarily unavailable. Please try again shortly.")


# --- LLM / recoverable pipeline errors (PRD §14.2) -----------------------------------------


class RecoverableLLMError(AppError):
    status_code = 502

    def __init__(self, detail: str):
        super().__init__("AI service error", detail)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=exc.error, detail=exc.detail).model_dump(),
        )

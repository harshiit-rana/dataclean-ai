"""App-level exceptions mapped to the {error, detail} response shape (PRD §10.3),
covering the error table in PRD §14."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.request_context import current_request_id
from app.models.api_schemas import ErrorResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base for all errors that should surface as a plain-English {error, detail} response."""

    status_code = 400

    def __init__(self, error: str, detail: str, code: str | None = None):
        self.error = error
        self.detail = detail
        # A stable, machine-readable reason, for the few errors a page can offer a next step for.
        # Absent from the response when None, so every other error is exactly what it was.
        self.code = code
        super().__init__(detail)


# --- File upload errors (PRD §14.1) ------------------------------------------------------


class UnsupportedFileTypeError(AppError):
    def __init__(self):
        super().__init__("Unsupported file type", "Only CSV files are supported.")


class FileTooLargeError(AppError):
    def __init__(self, max_mb: int):
        super().__init__("File too large", f"File exceeds the {max_mb}MB limit.")


class FileShapeTooLargeError(AppError):
    """A file within the byte limit whose shape would still exhaust memory: too many columns, too many rows, or one
    enormous cell. 413 (content too large). The detail says which limit and how to raise it."""

    status_code = 413

    def __init__(self, detail: str):
        super().__init__("File too large to process", detail)


class UnparseableCSVError(AppError):
    def __init__(self):
        super().__init__("File could not be read as CSV", "File could not be read as CSV.")


class InvalidRuleError(AppError):
    """A business rule that cannot be evaluated as written.

    422 rather than 400: the request was well-formed JSON, but the rule it described is not
    one the engine can check. The message is the engine's own -- it says which column and what
    is wrong with it, which is what the user needs to fix it.
    """

    status_code = 422

    def __init__(self, detail: str):
        super().__init__("Rule could not be used", detail)


class EmptyFileError(AppError):
    def __init__(self):
        super().__init__("Empty file", "File appears to be empty.")


class HeaderOnlyFileError(AppError):
    def __init__(self):
        super().__init__("No data rows", "File has no data rows.")


class ModelArtifactRefusedError(AppError):
    """A stored model was not loaded: its file changed since it was saved, or this server's scikit-learn cannot be
    trusted to load it. The detail says which and what to do."""

    status_code = 409

    def __init__(self, detail: str, code: str):
        super().__init__("The saved model was not loaded", detail, code)


# --- Session errors (PRD §14.3) -----------------------------------------------------------


class SessionNotFoundError(AppError):
    status_code = 404

    def __init__(self, session_id: str):
        super().__init__("Session expired", f"Session {session_id} was not found. It may have expired.")


class SessionAccessDeniedError(AppError):
    status_code = 403

    def __init__(self):
        # Deliberately does not distinguish "wrong token" from "no token" -- both mean the
        # caller isn't the one who created the session, and saying which would help someone
        # probing for the difference.
        super().__init__(
            "Not your session",
            "This session belongs to a different browser. Sessions are private to whoever "
            "uploaded the file.",
        )


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
            content=ErrorResponse(
                error=exc.error, detail=exc.detail, code=exc.code, request_id=current_request_id(request)
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI's own validation errors, in the app's shape. Its default body put a list of objects in `detail`,
        which the page shows as text, so a malformed request surfaced as "[object Object]"."""
        problems = []
        for error in exc.errors()[:5]:
            where = ".".join(str(part) for part in error.get("loc", ()) if part not in ("body", "query", "path"))
            problems.append(f"{where}: {error.get('msg')}" if where else str(error.get("msg")))
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="Request not valid", detail="; ".join(problems) or "The request was not valid.",
                code="invalid_request", request_id=current_request_id(request),
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Unhandled errors otherwise return a bare "Internal Server Error" body with the
        traceback only in the server's stdout -- painful to debug behind Docker. Always log
        it, and in development echo the exception back so the cause is visible from the
        client too. Never leaks internals when APP_ENV isn't development."""
        logger.exception(
            "Unhandled error on %s %s", request.method, request.url.path,
            extra={"request_id": current_request_id(request)},
        )

        detail = "Something went wrong on our end. Please try again."
        if get_settings().app_env == "development":
            detail = f"{type(exc).__name__}: {exc}"

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error", detail=detail, request_id=current_request_id(request)
            ).model_dump(exclude_none=True),
        )

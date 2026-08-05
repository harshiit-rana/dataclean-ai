"""Per-session access tokens.

Session ids are server-generated UUIDs (NFR-13), which makes them unguessable but not
private: anything that learns one -- a log line, a shared screenshot, a proxy, a browser
history sync -- can then read that session's uploaded data and download its exports. On
localhost that doesn't matter. On a hosted instance it is the whole security model.

So on upload the server also issues a token, keeps only its hash, and requires it on every
request scoped to that session. This is deliberately not a user account system: there are no
users here, sessions are ephemeral, and inventing logins would be a much larger change for
the same guarantee. What it buys is that holding the id is no longer sufficient.

Off by default when APP_ENV is development, so `docker compose up` and a browser is still all
it takes to try the app.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

TOKEN_BYTES = 32
META_KEY = "access_token_hash"

HEADER_NAME = "X-Session-Token"


def issue_token() -> tuple[str, str]:
    """Returns (token, hash). Only the hash is ever stored, so a leaked meta.json doesn't
    hand over access to the session it describes."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    return token, hash_token(token)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(token: str | None, expected_hash: str | None) -> bool:
    """Constant-time comparison. The timing signal here is small, but the cost of not doing
    it is smaller still."""
    if not token or not expected_hash:
        return False
    return hmac.compare_digest(hash_token(token), expected_hash)

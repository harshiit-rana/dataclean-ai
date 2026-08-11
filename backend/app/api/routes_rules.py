"""GET/POST /api/rules/{session_id} -- REAL.

Business rules: what the data is *allowed* to contain, as opposed to what is statistically
unusual about it. See `services/business_rules.py` for why this is a separate idea from
everything else the app detects.

Two endpoints, and the split between them is the whole design. GET returns *proposals* --
guesses, with the evidence behind each one, that have no effect on anything. POST stores what
the user approved. Nothing between the two happens automatically, because a rule applied
without being read is a rule that can quietly mark a fifth of someone's rows as errors, and
that is the failure this app exists to prevent.

Stored in the session's meta sidecar rather than graph state: rules are set from the Clean and
Model pages, long after the graph has finished, and the readiness verdict reads them on every
request the same way it re-reads the file.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.core.errors import InvalidRuleError, SessionNotFoundError, UnparseableCSVError
from app.models.api_schemas import RulesResponse, SetRulesRequest
from app.services import business_rules
from app.services.session_store import read_session_meta, session_dir, write_session_meta

router = APIRouter()

META_KEY = "business_rules"


def _load(settings: Settings, session_id: str) -> tuple[pd.DataFrame, dict]:
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    directory = session_dir(settings.upload_dir, session_id)
    cleaned = directory / "cleaned.csv"
    path = cleaned if cleaned.exists() else directory / "original.csv"
    try:
        return pd.read_csv(path), meta
    except Exception as exc:
        raise UnparseableCSVError() from exc


def stored_rules(meta: dict | None) -> list[business_rules.Rule]:
    """Approved rules for a session, skipping any that no longer parse.

    Tolerant on purpose. A rule is stored as JSON and read back on every readiness call; if a
    stored rule became invalid -- an older shape, a hand-edited file -- the right behaviour is
    to ignore that one rule, not to make the readiness endpoint fail for the whole session.
    """
    if not meta:
        return []
    parsed: list[business_rules.Rule] = []
    for payload in meta.get(META_KEY) or []:
        try:
            parsed.append(business_rules.Rule.from_dict(payload))
        except (business_rules.InvalidRule, TypeError):
            continue
    return parsed


@router.get("/api/rules/{session_id}", response_model=RulesResponse)
async def get_rules(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> RulesResponse:
    """What is in force, and what the app would suggest adding."""
    df, meta = _load(settings, session_id)
    approved = stored_rules(meta)

    # A rule already approved is not proposed again, whether or not it currently passes.
    known = {rule.name for rule in approved}
    proposed = [r for r in business_rules.propose(df) if r.name not in known]

    return RulesResponse(
        session_id=session_id,
        rules=[rule.as_dict() for rule in approved],
        proposed=[rule.as_dict() for rule in proposed],
        checks=[check.as_dict() for check in business_rules.check_all(df, approved)],
    )


@router.post("/api/rules/{session_id}", response_model=RulesResponse)
async def set_rules(
    session_id: str,
    request: SetRulesRequest,
    settings: Settings = Depends(get_settings),
) -> RulesResponse:
    """Replace the session's rules with exactly what the user approved.

    A whole-list replace rather than add/remove endpoints, because the client always holds the
    complete set and partial updates would let the two drift apart. An invalid rule fails the
    request rather than being silently dropped -- the user is told their rule was rejected and
    why, which they cannot be if it vanishes.
    """
    df, meta = _load(settings, session_id)

    approved: list[business_rules.Rule] = []
    for payload in request.rules:
        try:
            approved.append(business_rules.Rule.from_dict(payload))
        except business_rules.InvalidRule as exc:
            raise InvalidRuleError(str(exc)) from exc
        except TypeError as exc:
            raise InvalidRuleError("That rule is missing something it needs.") from exc

    meta[META_KEY] = [rule.as_dict() for rule in approved]
    write_session_meta(settings.upload_dir, session_id, meta)

    known = {rule.name for rule in approved}
    proposed = [r for r in business_rules.propose(df) if r.name not in known]

    return RulesResponse(
        session_id=session_id,
        rules=[rule.as_dict() for rule in approved],
        proposed=[rule.as_dict() for rule in proposed],
        checks=[check.as_dict() for check in business_rules.check_all(df, approved)],
    )

"""Shared pytest fixtures. PRD §15.2."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(autouse=True)
def _test_env(monkeypatch, tmp_path):
    """Keeps tests from touching the repo's real ./uploads dir or requiring a real
    ANTHROPIC_API_KEY, and clears the lru_cache'd Settings singleton so env overrides
    actually take effect."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

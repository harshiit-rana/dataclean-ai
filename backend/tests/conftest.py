"""Shared pytest fixtures. PRD §15.2."""

from __future__ import annotations

import os
from pathlib import Path

# One thread per test worker when the suite runs in parallel (pytest-xdist). The forests fit with n_jobs=-1 and
# gradient boosting uses every OpenMP thread, so four workers on a four-core CI runner were sixteen-plus busy threads
# fighting over four cores, and the parallel run took longer than the serial one. Set before scikit-learn and numpy
# are imported, which read these at start-up. The fits are the same whatever the thread count.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ.setdefault(_variable, "1")

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
    # Development, said explicitly: the app requires session tokens unless it is told otherwise, and most tests
    # exercise the routes without them. The token tests set their own mode.
    monkeypatch.setenv("APP_ENV", "development")

    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

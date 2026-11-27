"""#141: on an empty Redis the backend and the worker set up the checkpointer at the same moment. The library checks
for its search index and then creates it, so one of them can be told the index already exists; that must not stop
startup, while any other setup error still must."""

from __future__ import annotations

import pytest
from redisvl.exceptions import RedisSearchError

from app.config import Settings
from app.graph import checkpointer as cp


class _FakeSaver:
    """Stands in for `AsyncRedisSaver.from_conn_string(...)`: raises the queued errors on entry, one per attempt."""

    def __init__(self, errors: list[Exception]):
        self.errors = errors
        self.entries = 0
        self.exits = 0

    def __call__(self, _url: str) -> "_FakeSaver":
        return self

    async def __aenter__(self) -> "_FakeSaver":
        self.entries += 1
        if self.errors:
            raise self.errors.pop(0)
        return self

    async def __aexit__(self, *_exc) -> None:
        self.exits += 1


def _patch(monkeypatch, fake: _FakeSaver) -> None:
    monkeypatch.setattr(cp.AsyncRedisSaver, "from_conn_string", fake)


async def test_index_created_by_the_other_process_is_retried(monkeypatch):
    fake = _FakeSaver([RedisSearchError("Failed to create index 'checkpoint' on Redis: SEARCH_INDEX_EXISTS Index already exists")])
    _patch(monkeypatch, fake)
    async with cp.build_checkpointer(Settings()) as saver:
        assert saver is fake
    assert fake.entries == 2
    assert fake.exits == 1


async def test_any_other_setup_error_still_fails_startup(monkeypatch):
    fake = _FakeSaver([RedisSearchError("Failed to create index 'checkpoint' on Redis: unknown command FT.CREATE")])
    _patch(monkeypatch, fake)
    with pytest.raises(RedisSearchError, match="unknown command"):
        async with cp.build_checkpointer(Settings()):
            pass
    assert fake.entries == 1


async def test_gives_up_after_a_few_attempts(monkeypatch):
    errors: list[Exception] = [RedisSearchError("Index already exists") for _ in range(cp._SETUP_ATTEMPTS)]
    fake = _FakeSaver(errors)
    _patch(monkeypatch, fake)
    with pytest.raises(RedisSearchError):
        async with cp.build_checkpointer(Settings()):
            pass
    assert fake.entries == cp._SETUP_ATTEMPTS


async def test_an_error_in_the_body_is_not_retried(monkeypatch):
    fake = _FakeSaver([])
    _patch(monkeypatch, fake)
    with pytest.raises(ValueError, match="from the app"):
        async with cp.build_checkpointer(Settings()):
            raise ValueError("from the app")
    assert fake.entries == 1
    assert fake.exits == 1

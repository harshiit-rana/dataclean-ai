"""One retention sweep at a time per deployment (#105)."""

from __future__ import annotations

import json
import threading
import time

from app.config import get_settings
from app.services import sweep_lock


def test_only_one_of_two_sweepers_gets_the_lock(tmp_path):
    assert sweep_lock.acquire(str(tmp_path), "api-1") is True
    assert sweep_lock.acquire(str(tmp_path), "api-2") is False


def test_many_at_once_still_give_exactly_one_winner(tmp_path):
    winners = []
    barrier = threading.Barrier(8)

    def race(n):
        barrier.wait()
        if sweep_lock.acquire(str(tmp_path), f"p{n}"):
            winners.append(n)

    threads = [threading.Thread(target=race, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1


def test_a_dead_sweepers_lock_is_broken_after_its_time(tmp_path):
    assert sweep_lock.acquire(str(tmp_path), "dead", now=time.time() - sweep_lock.TTL_SECONDS - 5)
    assert sweep_lock.acquire(str(tmp_path), "alive") is True
    held = json.loads((tmp_path / sweep_lock.LOCK_FILE).read_text())
    assert held["owner"] == "alive"
    assert not list(tmp_path.glob(f"{sweep_lock.LOCK_FILE}.stale.*"))


def test_a_live_lock_is_not_broken(tmp_path):
    sweep_lock.acquire(str(tmp_path), "alive")
    assert sweep_lock.acquire(str(tmp_path), "other") is False


def test_only_the_owner_releases_it(tmp_path):
    sweep_lock.acquire(str(tmp_path), "owner")
    sweep_lock.release(str(tmp_path), "someone-else")
    assert (tmp_path / sweep_lock.LOCK_FILE).exists()
    sweep_lock.release(str(tmp_path), "owner")
    assert not (tmp_path / sweep_lock.LOCK_FILE).exists()


def test_held_releases_on_the_way_out_and_says_who_had_it(tmp_path):
    with sweep_lock.held(str(tmp_path), "a") as mine:
        assert mine is True
        with sweep_lock.held(str(tmp_path), "b") as theirs:
            assert theirs is False
    assert not (tmp_path / sweep_lock.LOCK_FILE).exists()


def test_the_lock_file_is_not_taken_for_a_session(tmp_path):
    """The sweep walks folders; the lock is a file beside them, so it is never swept or counted."""
    from app.services.session_store import purge_expired_session_ids

    sweep_lock.acquire(str(tmp_path), "a", now=0)
    assert purge_expired_session_ids(str(tmp_path), 60) == []
    assert (tmp_path / sweep_lock.LOCK_FILE).exists()


def test_two_api_processes_do_not_both_sweep(monkeypatch):
    """The acceptance test of #105, at the level of one pass: while one holds the lock the other's pass does nothing."""
    import asyncio

    from app import main

    settings = get_settings()
    sweep_lock.acquire(settings.upload_dir, "the-other-api")
    calls = []
    monkeypatch.setattr(main, "purge_expired_session_ids", lambda *a, **k: calls.append(a) or [])
    assert asyncio.run(main.sweep_once(settings, None, owner="this-api")) is None
    assert calls == []
    sweep_lock.release(settings.upload_dir, "the-other-api")
    assert asyncio.run(main.sweep_once(settings, None, owner="this-api")) == []
    assert len(calls) == 1


def test_the_api_leaves_the_sweep_to_a_worker_when_there_is_one(monkeypatch):
    from app import main

    settings = get_settings()
    assert main.sweeps_in_api(settings) is True
    assert main.sweeps_in_api(settings.model_copy(update={"job_runner": "worker"})) is False
    assert main.sweeps_in_api(settings.model_copy(update={"upload_retention_seconds": 0})) is False


def test_the_worker_does_not_sweep_files_when_it_cannot_reach_redis(monkeypatch, tmp_path):
    """Files removed without their checkpoints would orphan the checkpoints for good: no pass at all instead."""
    import contextlib
    import os

    from redis.exceptions import ConnectionError as RedisConnectionError

    from app import worker
    from app.graph import checkpointer

    old = tmp_path / "expired-session"
    old.mkdir()
    (old / "meta.json").write_text("{}")
    for path in (old, old / "meta.json"):
        os.utime(path, (0, 0))

    @contextlib.asynccontextmanager
    async def unreachable(_settings):
        raise RedisConnectionError("no redis")
        yield  # pragma: no cover

    monkeypatch.setattr(checkpointer, "build_checkpointer", unreachable)
    settings = get_settings().model_copy(update={"upload_dir": str(tmp_path), "upload_retention_seconds": 60})
    assert worker.sweep(settings, "w") is None
    assert old.exists()

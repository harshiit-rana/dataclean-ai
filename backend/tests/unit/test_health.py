"""Liveness answers whatever else is down; readiness says whether Redis and the graph are there."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import health
from app.main import app

client = TestClient(app)


def test_liveness_is_constant():
    assert client.get("/health").json() == {"status": "ok"}


def test_readiness_fails_when_redis_is_down(monkeypatch):
    async def down(_url):
        return False

    monkeypatch.setattr(health, "_redis_answers", down)
    monkeypatch.setattr(app.state, "graph", object(), raising=False)
    response = client.get("/health/ready")
    assert response.status_code == 503 and response.json() == {"status": "not ready", "redis": "unreachable", "graph": "ok"}


def test_readiness_fails_without_the_graph(monkeypatch):
    async def up(_url):
        return True

    monkeypatch.setattr(health, "_redis_answers", up)
    monkeypatch.delattr(app.state, "graph", raising=False)
    response = client.get("/health/ready")
    assert response.status_code == 503 and response.json()["graph"] == "not built"


def test_ready_when_both_answer(monkeypatch):
    async def up(_url):
        return True

    monkeypatch.setattr(health, "_redis_answers", up)
    monkeypatch.setattr(app.state, "graph", object(), raising=False)
    response = client.get("/health/ready")
    assert response.status_code == 200 and response.json() == {"status": "ready", "redis": "ok", "graph": "ok"}


async def test_an_unreachable_redis_is_reported_within_the_timeout(monkeypatch):
    import time

    monkeypatch.setattr(health, "READY_TIMEOUT_SECONDS", 0.5)
    started = time.monotonic()
    assert await health._redis_answers("redis://127.0.0.1:1") is False
    assert time.monotonic() - started < 3

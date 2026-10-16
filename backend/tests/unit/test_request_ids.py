"""Every request has an id on its log lines, in its response header and in any error body."""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from app.core.logging import configure_logging
from app.core.request_context import JsonFormatter, RequestIdFilter
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestTheId:
    def test_each_response_carries_a_new_id(self):
        first = client.get("/health").headers["x-request-id"]
        second = client.get("/health").headers["x-request-id"]
        assert first and second and first != second

    def test_a_sensible_caller_id_is_kept(self):
        response = client.get("/health", headers={"X-Request-Id": "abc-123.DEF_9"})
        assert response.headers["x-request-id"] == "abc-123.DEF_9"

    def test_an_id_that_could_inject_into_a_log_line_is_replaced(self):
        response = client.get("/health", headers={"X-Request-Id": "evil\nINFO fake line"})
        assert response.headers["x-request-id"] != "evil\nINFO fake line"
        assert len(response.headers["x-request-id"]) == 16

    def test_an_error_body_carries_the_id(self):
        response = client.get("/api/model/no-such-session/result", headers={"X-Request-Id": "trace-42"})
        assert response.status_code == 404 and response.json()["request_id"] == "trace-42"


class TestTheLogs:
    def test_every_line_logged_during_a_request_carries_its_id(self, monkeypatch, caplog):
        from app.api import health

        def noisy():
            logging.getLogger("app.test").warning("inside the request")
            return {"status": "ok"}

        caplog.handler.addFilter(RequestIdFilter())
        with caplog.at_level(logging.WARNING):
            app.router.add_api_route("/api/_test_noisy", noisy)
            client.get("/api/_test_noisy", headers={"X-Request-Id": "line-7"})
        assert any(r.getMessage() == "inside the request" and r.request_id == "line-7" for r in caplog.records)
        assert health  # the module exists; the test route above is the noisy one

    def test_an_unexpected_error_is_logged_and_answered_with_the_same_id(self, caplog):
        def broken():
            raise RuntimeError("secret internals")

        caplog.handler.addFilter(RequestIdFilter())
        app.router.add_api_route("/api/_test_broken", broken)
        with caplog.at_level(logging.ERROR):
            response = client.get("/api/_test_broken", headers={"X-Request-Id": "boom-1"})
        assert response.status_code == 500 and response.json()["request_id"] == "boom-1"
        assert any("Unhandled error" in r.getMessage() and r.request_id == "boom-1" for r in caplog.records)

    def test_the_text_format_prints_the_id(self):
        configure_logging("INFO")
        handler = logging.getLogger().handlers[0]
        record = logging.LogRecord("x", logging.INFO, "f", 1, "hello", None, None)
        assert handler.filter(record) and "[-] x: hello" in handler.format(record)

    def test_the_json_format_is_one_object_per_line(self):
        record = logging.LogRecord("x", logging.INFO, "f", 1, "hello %s", ("there",), None)
        RequestIdFilter().filter(record)
        parsed = json.loads(JsonFormatter().format(record))
        assert parsed["message"] == "hello there" and parsed["request_id"] == "-" and parsed["level"] == "INFO"
        configure_logging("INFO", "json")
        assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)
        configure_logging("INFO")

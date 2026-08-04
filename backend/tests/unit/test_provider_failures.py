"""What happens when the AI provider fails part-way through a run.

Not hypothetical: both supported providers have free tiers with daily token caps, and one
was hit during a real end-to-end run. The stream died mid-chunk, the browser reported a
generic network error, and the whole session's work was lost with no indication of why.
"""

from __future__ import annotations

import json

import pytest

from app.graph.nodes.explanation_agent import _readable_failure
from app.services.sse_broker import _readable_failure as broker_message, stream_graph_as_sse


class FakeRateLimitError(Exception):
    """Stands in for groq.RateLimitError / anthropic.RateLimitError without importing either."""

    def __init__(self):
        super().__init__(
            "Error code: 429 - Rate limit reached for model ... on tokens per day (TPD)"
        )


class TestUserFacingMessages:
    def test_a_rate_limit_is_named_as_one(self):
        message = broker_message(FakeRateLimitError())
        assert "rate limit" in message.lower()
        # The thing a user most needs to know first.
        assert "nothing has been changed" in message.lower()

    def test_a_rate_limit_suggests_what_to_do(self):
        message = broker_message(FakeRateLimitError())
        assert "reset" in message or "switch provider" in message

    def test_a_bad_key_is_named_as_one(self):
        class AuthenticationError(Exception):
            pass

        assert "API key" in broker_message(AuthenticationError("401 invalid api_key"))

    def test_a_network_problem_is_named_as_one(self):
        class ConnectionError_(Exception):
            pass

        ConnectionError_.__name__ = "ConnectionError"
        assert "connection" in broker_message(ConnectionError_("boom")).lower()

    def test_an_unknown_error_still_says_something_concrete(self):
        message = broker_message(ValueError("something odd"))
        assert "something odd" in message

    def test_a_long_provider_message_is_truncated(self):
        message = broker_message(ValueError("x" * 5000))
        assert len(message) < 400


class TestPartialCardRun:
    def test_leads_with_what_the_user_still_has(self):
        """Four of twelve cards is a partial success. Reporting it as a failure throws away
        work the user could act on right now."""
        message = _readable_failure(FakeRateLimitError(), written=4, total=12)
        assert "4 suggestions already written" in message
        assert "ready to review" in message
        assert "4 of 12" in message

    def test_singular_reads_correctly(self):
        assert "1 suggestion already written is ready" in _readable_failure(
            FakeRateLimitError(), written=1, total=9
        )

    def test_says_so_plainly_when_nothing_was_written(self):
        message = _readable_failure(FakeRateLimitError(), written=0, total=9)
        assert "No suggestions were written." in message


class TestStreamSurvivesAFailure:
    @pytest.mark.asyncio
    async def test_a_failing_run_ends_with_an_error_event_not_a_dead_connection(self):
        """The generator raising closes the HTTP response mid-chunk, which the browser can
        only report as a generic network error."""

        class ExplodingGraph:
            async def astream(self, *_args, **_kwargs):
                yield ("custom", {"event": "agent_start", "agent": "profiling_agent"})
                raise FakeRateLimitError()

        frames = [
            frame async for frame in stream_graph_as_sse(ExplodingGraph(), {}, {}, "session-1")
        ]

        assert len(frames) == 2
        assert "agent_start" in frames[0]

        payload = json.loads(frames[1].split("data: ", 1)[1].strip())
        assert payload["event"] == "error"
        assert "rate limit" in payload["detail"].lower()

    @pytest.mark.asyncio
    async def test_a_healthy_run_still_signals_the_interrupt(self):
        class HealthyGraph:
            async def astream(self, *_args, **_kwargs):
                yield ("custom", {"event": "all_cards_complete", "total_cards": 3})

            async def aget_state(self, _config):
                class State:
                    next = ("human_approval",)

                return State()

        frames = [
            frame async for frame in stream_graph_as_sse(HealthyGraph(), {}, {}, "session-1")
        ]
        assert "all_cards_complete" in frames[0]
        assert "interrupt_reached" in frames[1]

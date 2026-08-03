"""ChatAnthropic factory. PRD §8.1 (Claude powers all four agents), FR-27/NFR-08 (retry once
on failure, surface a recoverable error rather than crashing the session), §14.2 (30s network
timeout, retry once)."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from app.config import Settings

LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 1  # "retry once" per FR-27/NFR-08/§14.2 -- not exponential backoff to N tries


def get_claude_client(settings: Settings, *, streaming: bool = False) -> ChatAnthropic:
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your own key "
            "from console.anthropic.com."
        )
    return ChatAnthropic(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        streaming=streaming,
    )

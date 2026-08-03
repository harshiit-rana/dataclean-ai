"""LLM client factory. PRD §8.1 (an LLM powers all four agents), FR-27/NFR-08 (retry once
on failure, surface a recoverable error rather than crashing the session), §14.2 (30s network
timeout, retry once).

Provider-agnostic by design: every node only ever calls the generic `.ainvoke()` /
`.astream()` methods LangChain's `BaseChatModel` interface provides (see
graph/nodes/profiling_agent.py, graph/nodes/explanation_agent.py), so swapping providers
here doesn't touch any node code. `LLM_PROVIDER` picks which one at runtime:

- "anthropic" (default): Claude, via ChatAnthropic. Best-verified for this app's two
  structured-output-dependent calls (Profiling Agent's JSON array, Explanation Agent's
  marker-formatted cards) -- see README's "LLM provider" section for the tradeoff.
- "groq": much cheaper / has a generous free tier, via ChatGroq. Recommended model is
  Llama 3.3 70B for the same reason Sonnet is the Anthropic default: it's the strongest
  instruction-follower on structured output among the options, and the Explanation Agent's
  marker format has no retry-on-parse-failure today, so a weaker model here silently
  produces blank suggestion cards rather than an error.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings

LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 1  # "retry once" per FR-27/NFR-08/§14.2 -- not exponential backoff to N tries


def get_llm_client(settings: Settings, *, streaming: bool = False) -> BaseChatModel:
    if settings.llm_provider == "groq":
        return _groq_client(settings, streaming=streaming)
    return _anthropic_client(settings, streaming=streaming)


def _anthropic_client(settings: Settings, *, streaming: bool) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your own key "
            "from console.anthropic.com, or set LLM_PROVIDER=groq to use Groq instead."
        )
    return ChatAnthropic(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        streaming=streaming,
    )


def _groq_client(settings: Settings, *, streaming: bool) -> BaseChatModel:
    from langchain_groq import ChatGroq

    if not settings.groq_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set. Copy .env.example to .env and "
            "add your own key from console.groq.com."
        )
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        streaming=streaming,
    )

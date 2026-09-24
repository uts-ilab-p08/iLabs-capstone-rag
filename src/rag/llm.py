"""LLM client — one code path for OpenRouter and for local Ollama.

Both expose the same OpenAI chat-completions API, so choosing between a hosted
model and a local one is configuration, not code:

    OpenRouter   LLM_BASE_URL=https://openrouter.ai/api/v1
                 LLM_MODEL=<vendor>/<model>        LLM_API_KEY=sk-or-...
    Ollama       LLM_BASE_URL=http://localhost:11434/v1
                 LLM_MODEL=qwen3:4b                LLM_API_KEY= (unused)

That is also how we compare models for the report: change LLM_MODEL, re-run.
"""

from __future__ import annotations

import re
from functools import lru_cache

from openai import OpenAI

from rag.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# Reasoning models (qwen3 and others) may return their private scratchpad
# wrapped in <think> tags. That is not part of the answer, so strip it.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """The model could not be reached or returned nothing usable."""


def _normalise_base_url(url: str) -> str:
    """Drop a trailing /chat/completions if someone pasted the full endpoint.

    The SDK appends the endpoint path itself, so a base URL that already ends
    in it produces .../chat/completions/chat/completions and a bare 404 that
    says nothing about the cause. Easy mistake, so absorb it here.
    """
    return re.sub(r"/+chat/completions/*$", "", url.strip())


def _extract_error(response) -> str | None:
    """Return an error message if the response carries one instead of a reply.

    OpenRouter reports rate limits, unavailable models and upstream provider
    failures as HTTP 200 with an `error` object in the body, so the SDK parses
    them into an object whose `choices` is None. Reading choices[0] blindly
    crashes with a confusing TypeError, so check first.
    """
    extra = getattr(response, "model_extra", None) or {}
    error = getattr(response, "error", None) or extra.get("error")
    if error:
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error)
    if not getattr(response, "choices", None):
        return "Response contained no choices"
    return None


def is_configured() -> bool:
    """True when a model has been chosen in .env."""
    return bool(LLM_MODEL)


def describe() -> str:
    return f"{LLM_MODEL or '(no model set)'} via {_normalise_base_url(LLM_BASE_URL)}"


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # Ollama ignores the key, but the SDK insists on a non-empty value.
    return OpenAI(base_url=_normalise_base_url(LLM_BASE_URL), api_key=LLM_API_KEY or "not-needed")


def complete(system: str, user: str, temperature: float = 0.0, max_tokens: int = 2000) -> str:
    """Send one prompt, return the reply text.

    temperature defaults to 0 so the same question gives the same answer — we
    are comparing models and measuring retrieval, and random variation would
    make those numbers meaningless.

    max_tokens is generous because reasoning models (qwen3, deepseek-r1 and
    similar) spend tokens thinking before they write anything. With a tight cap
    they hit the limit mid-thought and return an empty answer.
    """
    if not is_configured():
        raise LLMError("No LLM_MODEL set in .env")

    try:
        response = _client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as LLMError
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc

    error = _extract_error(response)
    if error:
        raise LLMError(error)

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    text = _THINK_BLOCK.sub("", text).strip()
    if not text:
        if choice.finish_reason == "length":
            raise LLMError(
                f"Hit the {max_tokens}-token limit before answering "
                "(a reasoning model may need a higher max_tokens)"
            )
        raise LLMError("Model returned an empty answer")
    return text

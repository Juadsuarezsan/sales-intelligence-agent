"""Thin LLM client used by every agent node.

The agent nodes depend on the :class:`LLMClient` protocol, not on the Anthropic
SDK, so tests can inject a scripted fake and the offline fallback simply
receives ``None``. :class:`AnthropicLLM` is the production implementation:
pinned model, explicit timeout, bounded retries with exponential backoff and
token accounting through :mod:`src.observability`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings
from src.observability import current_usage


class LLMOutputError(ValueError):
    """Raised when the model output cannot be parsed into the expected JSON."""


def recoverable_llm_errors() -> tuple[type[Exception], ...]:
    """Exception types after which a node may fall back to its offline heuristic.

    Covers unparseable output, schema violations and API failures (after the
    client's own retries). Anything else is a programming error and propagates.

    Returns:
        Tuple usable in an ``except`` clause.
    """
    from anthropic import APIError
    from pydantic import ValidationError

    return (LLMOutputError, ValidationError, APIError)


@dataclass(frozen=True)
class LLMResult:
    """Result of one completion.

    Attributes:
        text: Concatenated text blocks of the response.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        model: Model ID that served the request.
    """

    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient(Protocol):
    """Minimal completion interface shared by the production client and test fakes."""

    model: str

    async def complete(
        self, *, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResult:
        """Run one single-turn completion.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Completion cap.
            temperature: Sampling temperature.

        Returns:
            The completion and its token usage.
        """
        ...


class AnthropicLLM:
    """Production :class:`LLMClient` backed by the Anthropic Messages API.

    Args:
        api_key: Anthropic API key.
        model: Dated model ID.
        timeout_seconds: Per-request timeout.
        max_attempts: Total attempts (first try plus retries) on transient errors.
        tracing: Wrap the SDK client with the LangSmith tracer when ``True``.
        backoff_seconds: ``(min, max)`` bounds of the exponential backoff between attempts.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        tracing: bool = False,
        backoff_seconds: tuple[float, float] = (1.0, 10.0),
    ) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        client = AsyncAnthropic(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        if tracing:
            from langsmith.wrappers import wrap_anthropic

            client = wrap_anthropic(client)
        self._client = client

    async def complete(
        self, *, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResult:
        """Call the Messages API with bounded retries on transient failures.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Completion cap.
            temperature: Sampling temperature.

        Returns:
            The completion and its token usage.

        Raises:
            anthropic.APIError: When a non-retryable API error occurs or retries
                are exhausted.
        """
        from anthropic import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=1, min=self.backoff_seconds[0], max=self.backoff_seconds[1]
            ),
            retry=retry_if_exception_type(
                (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
            ),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    response = await self._client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
        except RetryError as exc:  # pragma: no cover - reraise=True makes this unreachable
            raise exc.last_attempt.exception() from exc  # type: ignore[misc]

        text = "".join(block.text for block in response.content if block.type == "text")
        result = LLMResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )
        tracker = current_usage()
        if tracker is not None:
            tracker.record_llm(result.input_tokens, result.output_tokens)
        logger.debug(
            f"llm model={result.model} in={result.input_tokens} out={result.output_tokens}"
        )
        return result


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response.

    Handles Markdown code fences and leading/trailing prose.

    Args:
        text: Raw model output.

    Returns:
        The parsed JSON object.

    Raises:
        LLMOutputError: If no JSON object can be decoded.
    """
    cleaned = _FENCE.sub("", text.strip()).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMOutputError("no JSON object found in model output")
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"invalid JSON in model output: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise LLMOutputError("model output is not a JSON object")
    return parsed


def build_llm(settings: Settings, *, model: str | None = None) -> LLMClient | None:
    """Build the production client from settings, or ``None`` when no key is set.

    Args:
        settings: Application settings.
        model: Override the model ID (used for the judge).

    Returns:
        An :class:`AnthropicLLM` or ``None`` (offline fallback).
    """
    if settings.anthropic_api_key is None:
        return None
    return AnthropicLLM(
        settings.anthropic_api_key,
        model or settings.anthropic_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        tracing=settings.tracing_enabled,
    )

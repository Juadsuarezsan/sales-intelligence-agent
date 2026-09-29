"""Shared fixtures: fixed seed, offline environment, fake LLM and sample profile."""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src.api.schemas import CompanyProfile, KeyPerson, Signal
from src.config import get_settings
from src.llm import LLMResult
from src.observability import current_usage

SEED = 20260516
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(SEED)


@pytest.fixture(autouse=True)
def offline_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Guarantee no key, no database and no network-backed tool in tests."""
    for key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY", "DATABASE_URL", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("USE_REAL_HN", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("RATE_LIMIT", "1000/minute")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeLLM:
    """Scripted :class:`src.llm.LLMClient` returning canned responses in order."""

    model = "fake-model"

    def __init__(self, responses: list[str] | None = None, *, default: str = "{}") -> None:
        self.responses = list(responses or [])
        self.default = default
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, *, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResult:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        text = self.responses.pop(0) if self.responses else self.default
        tracker = current_usage()
        if tracker is not None:
            tracker.record_llm(100, 50)
        return LLMResult(text=text, input_tokens=100, output_tokens=50, model=self.model)


class RoutedLLM(FakeLLM):
    """Fake LLM that picks the response by a keyword found in the system prompt."""

    ROUTES: dict[str, str] = {
        "plan research queries": "planner",
        "review research evidence": "reflector",
        "structured B2B company profile": "builder",
        "senior SDR": "extractor",
        "write short personalized": "writer",
        "strict evaluator": "judge",
    }

    def __init__(self, by_node: dict[str, str]) -> None:
        super().__init__()
        self.by_node = by_node

    async def complete(
        self, *, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResult:
        node = next((n for k, n in self.ROUTES.items() if k in system), "unknown")
        self.calls.append({"node": node, "user": user})
        tracker = current_usage()
        if tracker is not None:
            tracker.record_llm(200, 80)
        return LLMResult(
            text=self.by_node.get(node, "{}"), input_tokens=200, output_tokens=80, model=self.model
        )


def llm_json(**payload: Any) -> str:
    """Serialise a dict as the JSON string a model would return."""
    return json.dumps(payload)


@pytest.fixture
def profile() -> CompanyProfile:
    return CompanyProfile(
        name="Anthropic",
        one_liner="AI safety and research company building reliable, steerable AI systems",
        industry="B2B",
        website="https://anthropic.com",
        location="San Francisco, CA, USA",
        employees_est=800,
        funding_total_usd=2_000_000_000,
        last_funding_round="Series C",
        key_people=[KeyPerson(name="Dario Amodei", role="CEO")],
        recent_signals=[
            Signal(
                kind="funding",
                description="$2B Series C round closed",
                source_url="https://example.com/funding",
            ),
            Signal(
                kind="hiring",
                description="6 open roles on careers page",
                source_url="https://anthropic.com/careers",
            ),
        ],
    )


@pytest.fixture
def html_fixture() -> Any:
    def _load(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return _load

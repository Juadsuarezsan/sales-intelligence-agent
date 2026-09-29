"""Pydantic schemas shared by the agent, the HTTP API, the eval and the demo."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

SignalKind = Literal["funding", "hiring", "product_launch", "news", "leadership_change", "other"]


class Evidence(TypedDict, total=False):
    """One piece of raw evidence collected by a tool.

    Keys:
        title: Page or result title.
        url: Source URL.
        content: Text snippet.
        score: Relevance score in ``[0, 1]``.
        source: Tool that produced it (``web``, ``hn``, ``website``).
    """

    title: str
    url: str
    content: str
    score: float
    source: str


class KeyPerson(BaseModel):
    """A named person associated with the company."""

    name: str = Field(..., min_length=1, max_length=120)
    role: str | None = None
    linkedin_url: str | None = None


class Signal(BaseModel):
    """A dated buying signal (funding, hiring, launch...)."""

    kind: SignalKind
    description: str = Field(..., max_length=500)
    source_url: str = ""
    occurred_at: str | None = None


class CompanyProfile(BaseModel):
    """Structured profile produced by the Profile Builder node."""

    name: str = Field(..., min_length=1, max_length=200)
    one_liner: str = ""
    industry: str | None = None
    website: str | None = None
    employees_est: int | None = Field(default=None, ge=0)
    funding_total_usd: float | None = Field(default=None, ge=0)
    last_funding_round: str | None = None
    location: str | None = None
    key_people: list[KeyPerson] = Field(default_factory=list)
    recent_signals: list[Signal] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class PersonalizationHooks(BaseModel):
    """Output of the Personalization Extractor node."""

    pain_point: str = Field(..., max_length=500)
    recent_signal: str | None = Field(default=None, max_length=500)
    angle: str = Field(default="", max_length=500)
    evidence_refs: list[str] = Field(default_factory=list)


class OutreachEmail(BaseModel):
    """Cold outreach email produced by the Email Writer node."""

    subject: str = Field(..., max_length=200)
    greeting: str = Field(..., max_length=200)
    hook: str = Field(..., max_length=1000)
    value_prop: str = Field(..., max_length=1000)
    cta: str = Field(..., max_length=500)
    personalization_score: float = Field(default=0.0, ge=0.0, le=1.0)

    def body(self) -> str:
        """Render the email as plain text."""
        return f"{self.greeting}\n\n{self.hook}\n\n{self.value_prop}\n\n{self.cta}"


class QualityScores(BaseModel):
    """Output of the Quality Scorer (LLM-as-judge) node.

    Scores follow the numbered rubric in :mod:`src.agents.quality_scorer`,
    each on a 1-5 scale. ``judge`` tells whether a real LLM produced them or
    the deterministic keyword fallback did.
    """

    personalization: int = Field(..., ge=1, le=5)
    accuracy: int = Field(..., ge=1, le=5)
    cta_clarity: int = Field(..., ge=1, le=5)
    overall: float = Field(..., ge=1.0, le=5.0)
    rationale: str = ""
    judge: Literal["llm", "heuristic"] = "heuristic"


class ResearchRequest(BaseModel):
    """Body of ``POST /api/research``."""

    company_name: str = Field(..., min_length=2, max_length=120)
    domain: str | None = Field(
        default=None, max_length=253, pattern=r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    )
    seed_context: str | None = Field(default=None, max_length=2000)
    max_reflection_iterations: int | None = Field(default=None, ge=0, le=5)


class ResearchResponse(BaseModel):
    """Full output of one research run."""

    trace_id: str = ""
    company_name: str = ""
    profile: CompanyProfile
    hooks: PersonalizationHooks | None = None
    email: OutreachEmail
    scores: QualityScores | None = None
    tool_calls: int = 0
    iterations: int = 0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    mode: Literal["llm", "fallback"] = "fallback"
    cached: bool = False


class CompanySummary(BaseModel):
    """Compact row for ``GET /api/companies``."""

    company_name: str
    one_liner: str
    industry: str | None
    personalization_score: float
    cost_usd: float
    latency_ms: int
    mode: Literal["llm", "fallback"]


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: Literal["ok"]
    version: str
    model: str
    llm_enabled: bool
    web_search: Literal["tavily", "stub"]
    hackernews: Literal["algolia", "stub"]
    repository: Literal["postgres", "memory"]
    tracing_enabled: bool

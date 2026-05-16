"""Pydantic schemas for the sales intelligence agent."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KeyPerson(BaseModel):
    name: str
    role: str | None = None
    linkedin_url: str | None = None


class Signal(BaseModel):
    kind: Literal["funding", "hiring", "product_launch", "news", "leadership_change", "other"]
    description: str
    source_url: str
    occurred_at: str | None = None


class CompanyProfile(BaseModel):
    name: str
    one_liner: str = ""
    industry: str | None = None
    employees_est: int | None = None
    funding_total_usd: float | None = None
    last_funding_round: str | None = None
    location: str | None = None
    key_people: list[KeyPerson] = Field(default_factory=list)
    recent_signals: list[Signal] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class OutreachEmail(BaseModel):
    subject: str
    greeting: str
    hook: str
    value_prop: str
    cta: str
    personalization_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ResearchRequest(BaseModel):
    company_name: str = Field(..., min_length=2)
    domain: str | None = None
    seed_context: str | None = None


class ResearchResponse(BaseModel):
    profile: CompanyProfile
    email: OutreachEmail
    tool_calls: int = 0
    iterations: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0

"""Sales Intelligence API — research a company, return profile + outreach email."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from src.config import get_settings


class ResearchRequest(BaseModel):
    company_name: str = Field(..., min_length=2)
    domain: str | None = None
    seed_context: str | None = None


class CompanyProfile(BaseModel):
    name: str
    one_liner: str = ""
    industry: str | None = None
    employees_est: int | None = None
    funding_total_usd: float | None = None
    last_funding_round: str | None = None
    location: str | None = None
    key_people: list[dict] = []
    recent_signals: list[dict] = []
    pain_points: list[str] = []
    sources: list[str] = []


class OutreachEmail(BaseModel):
    subject: str
    greeting: str
    hook: str
    value_prop: str
    cta: str
    personalization_score: float = 0.0


class ResearchResponse(BaseModel):
    profile: CompanyProfile
    email: OutreachEmail
    tool_calls: int
    iterations: int
    latency_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="B2B Sales Intelligence Agent", version="0.1.0",
    description="Planner-executor-reflector loop over web search + HN + website fetch.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok", "version": "0.1.0", "stage": "scaffolding",
        "tavily_enabled": bool(s.tavily_api_key),
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }


@app.post("/api/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    return ResearchResponse(
        profile=CompanyProfile(name=req.company_name, one_liner="(not yet implemented)"),
        email=OutreachEmail(subject="", greeting="", hook="", value_prop="", cta=""),
        tool_calls=0, iterations=0, latency_ms=0,
    )

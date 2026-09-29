"""FastAPI transport layer for the B2B Sales Intelligence Agent.

Endpoints:

* ``GET /health`` - liveness plus which backends are active.
* ``POST /api/research`` - run the agent for one company (rate limited).
* ``GET /api/companies`` - list persisted results for the gallery.
* ``GET /api/companies/{name}`` - fetch one persisted result.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agents.orchestrator import build_components, build_graph, research_company
from src.api.schemas import (
    CompanySummary,
    HealthResponse,
    ResearchRequest,
    ResearchResponse,
)
from src.config import get_settings
from src.observability import configure_logging
from src.storage import PostgresRepository, ResearchRepository, build_repository

load_dotenv()

VERSION = "1.0.0"
TRACE_HEADER = "X-Trace-Id"

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the graph and open the repository once per process."""
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.components = build_components(settings)
    app.state.graph = build_graph(app.state.components)
    repository = build_repository(settings)
    if isinstance(repository, PostgresRepository):
        await repository.open()
    app.state.repository = repository
    logger.info(
        f"api ready version={VERSION} model={settings.anthropic_model} "
        f"llm={settings.llm_enabled} repository={repository.name}"
    )
    yield
    await repository.close()


app = FastAPI(
    title="B2B Sales Intelligence Agent",
    version=VERSION,
    description="Plan-execute-reflect agent producing company profiles and outreach emails.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", TRACE_HEADER],
)


def _repository() -> ResearchRepository:
    repo: ResearchRepository = app.state.repository
    return repo


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe that also reports which backends are active."""
    s = get_settings()
    components = app.state.components
    return HealthResponse(
        status="ok",
        version=VERSION,
        model=s.anthropic_model,
        llm_enabled=s.llm_enabled,
        web_search="tavily" if components.web.name == "tavily" else "stub",
        hackernews="algolia" if components.news.name == "algolia" else "stub",
        repository="postgres" if _repository().name == "postgres" else "memory",
        tracing_enabled=s.tracing_enabled,
    )


@app.post("/api/research", response_model=ResearchResponse)
@limiter.limit(lambda: get_settings().rate_limit)
async def research(request: Request, response: Response, body: ResearchRequest) -> ResearchResponse:
    """Research one company and write its outreach email.

    Results are cached in the repository by company name; pass a different
    ``seed_context`` or ``max_reflection_iterations`` to bypass the cache.
    """
    repo = _repository()
    if body.seed_context is None and body.max_reflection_iterations is None:
        cached = await repo.get(body.company_name)
        if cached is not None:
            response.headers[TRACE_HEADER] = cached.trace_id
            return cached.model_copy(update={"cached": True})

    graph = app.state.graph
    if body.max_reflection_iterations is not None:
        graph = build_graph(
            build_components(max_reflection_iterations=body.max_reflection_iterations)
        )
    try:
        result = await research_company(
            graph,
            company_name=body.company_name,
            context=body.seed_context,
            website=body.domain,
            trace_id=request.headers.get(TRACE_HEADER),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("research failed")
        raise HTTPException(status_code=500, detail="research failed; see server logs") from exc
    await repo.save(result)
    response.headers[TRACE_HEADER] = result.trace_id
    return result


@app.get("/api/companies", response_model=list[CompanySummary])
async def list_companies(limit: int = Query(default=100, ge=1, le=500)) -> list[CompanySummary]:
    """List the most recently researched companies (gallery feed)."""
    items = await _repository().list_recent(limit)
    return [
        CompanySummary(
            company_name=r.company_name,
            one_liner=r.profile.one_liner,
            industry=r.profile.industry,
            personalization_score=r.email.personalization_score,
            cost_usd=r.cost_usd,
            latency_ms=r.latency_ms,
            mode=r.mode,
        )
        for r in items
    ]


@app.get("/api/companies/{company_name}", response_model=ResearchResponse)
async def get_company(company_name: str) -> ResearchResponse:
    """Return one persisted research result."""
    result = await _repository().get(company_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"no research stored for {company_name!r}")
    return result

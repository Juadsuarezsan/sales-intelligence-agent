"""The two baselines from the spec plus the agent under evaluation.

* ``template_only``: no research at all; profile = name + website; template
  email. Cost is zero; this is the "mail merge" floor.
* ``single_search_email``: one web search, one profile build, one email,
  no reflection loop, no HN, no website fetch. This is the common
  "search + LLM" product.
* ``agent_reflection``: the full LangGraph loop from
  :mod:`src.agents.orchestrator` with a configurable reflection budget.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from eval.ground_truth import GroundTruthCompany
from src.agents.orchestrator import AgentComponents, build_graph, research_company
from src.api.schemas import CompanyProfile, ResearchResponse
from src.config import Settings
from src.observability import current_trace_id, start_run

SystemRunner = Callable[[GroundTruthCompany], Awaitable[ResearchResponse]]


async def template_only(
    company: GroundTruthCompany, c: AgentComponents, s: Settings
) -> ResearchResponse:
    """Baseline 1: no research, templated email.

    Args:
        company: Target company.
        c: Component bundle (only extractor/email writer/scorer are used, offline).
        s: Settings for cost computation.

    Returns:
        A response with ``mode="fallback"``.
    """
    start_run()
    t0 = time.perf_counter()
    profile = CompanyProfile(name=company.name, website=company.website or None)
    hooks = c.extractor.heuristic(profile)
    email = c.email_writer.template(profile, hooks)
    scores = c.scorer.heuristic(email, profile)
    return ResearchResponse(
        trace_id=current_trace_id(),
        company_name=company.name,
        profile=profile,
        hooks=hooks,
        email=email,
        scores=scores,
        tool_calls=0,
        iterations=0,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        cost_usd=0.0,
        mode="fallback",
    )


async def single_search_email(
    company: GroundTruthCompany, c: AgentComponents, s: Settings
) -> ResearchResponse:
    """Baseline 2: one web search, then profile → hooks → email → score.

    Args:
        company: Target company.
        c: Component bundle (its web search, builder, extractor, writer, scorer).
        s: Settings for cost computation.

    Returns:
        A response with exactly one tool call.
    """
    tracker = start_run()
    t0 = time.perf_counter()
    evidence = await c.web.search(f"{company.name} company", max_results=5)
    profile = await c.profile_builder.build(company.name, evidence, website=company.website or None)
    hooks = await c.extractor.extract(profile)
    email = await c.email_writer.write(profile, hooks)
    scores = await c.scorer.score(email, profile)
    return ResearchResponse(
        trace_id=current_trace_id(),
        company_name=company.name,
        profile=profile,
        hooks=hooks,
        email=email,
        scores=scores,
        tool_calls=1,
        iterations=1,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        input_tokens=tracker.input_tokens,
        output_tokens=tracker.output_tokens,
        cost_usd=tracker.cost_usd(
            price_input_per_mtok=s.price_input_per_mtok,
            price_output_per_mtok=s.price_output_per_mtok,
            cost_per_search_usd=s.tavily_cost_per_search_usd,
        ),
        mode="llm" if c.llm_enabled else "fallback",
    )


def agent_runner(
    c: AgentComponents, s: Settings, *, max_reflection_iterations: int
) -> SystemRunner:
    """Build a runner for the full agent with a given reflection budget.

    Args:
        c: Component bundle.
        s: Settings for cost computation.
        max_reflection_iterations: Reflection rounds allowed (ablation knob).

    Returns:
        Async callable ``company -> ResearchResponse``.
    """
    graph = build_graph(c, max_reflection_iterations=max_reflection_iterations)

    async def run(company: GroundTruthCompany) -> ResearchResponse:
        return await research_company(
            graph,
            company_name=company.name,
            website=company.website or None,
            settings=s,
            llm_enabled=c.llm_enabled,
        )

    return run

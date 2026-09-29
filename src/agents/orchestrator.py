"""LangGraph plan → execute → reflect loop, then build → hooks → email → score.

The graph is assembled from an :class:`AgentComponents` bundle so tests, the
eval harness and the ablation study can swap tools, LLM clients and loop
budgets without touching global settings.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langgraph.graph import END, StateGraph
from loguru import logger
from tavily.errors import BadRequestError, InvalidAPIKeyError, UsageLimitExceededError
from typing_extensions import TypedDict

from src.agents.email_writer import EmailWriter
from src.agents.personalization import PersonalizationExtractor
from src.agents.planner import ResearchPlanner
from src.agents.profile_builder import ProfileBuilder
from src.agents.quality_scorer import QualityScorer
from src.agents.reflector import Reflector
from src.api.schemas import (
    CompanyProfile,
    Evidence,
    OutreachEmail,
    PersonalizationHooks,
    QualityScores,
    ResearchResponse,
)
from src.config import Settings, get_settings
from src.llm import LLMClient, build_llm
from src.observability import log_node, start_run
from src.tools.hackernews import HackerNewsSearch, NewsSearch, StubHackerNews
from src.tools.web_search import StubWebSearch, TavilyWebSearch, WebSearch
from src.tools.website import StubWebsiteFetcher, WebsiteFetch, WebsiteFetcher

#: Errors a tool may raise that must not abort the research run.
TOOL_ERRORS: tuple[type[Exception], ...] = (
    asyncio.TimeoutError,
    httpx.HTTPError,
    OSError,
    ValueError,
    BadRequestError,
    InvalidAPIKeyError,
    UsageLimitExceededError,
)


class AgentState(TypedDict, total=False):
    """State threaded through the LangGraph nodes.

    Node names never coincide with these keys (LangGraph rejects that).
    """

    company_name: str
    context: str | None
    website: str | None
    planned_queries: list[str]
    executed_queries: list[str]
    evidence: list[Evidence]
    iteration: int
    tool_calls: int
    sufficient: bool
    missing: list[str]
    profile: CompanyProfile
    hooks: PersonalizationHooks
    email: OutreachEmail
    scores: QualityScores


@dataclass
class AgentComponents:
    """Everything the graph needs, already constructed.

    Attributes:
        planner: Query planner.
        reflector: Sufficiency judge.
        profile_builder: Evidence → profile.
        extractor: Profile → personalization hooks.
        email_writer: Profile + hooks → email.
        scorer: LLM-as-judge quality scorer.
        web: Web search tool.
        news: HackerNews tool.
        website: Website fetcher.
        max_reflection_iterations: Extra plan/execute rounds allowed after reflection.
        max_tool_calls: Hard cap of tool calls per company.
        website_fetch_enabled: Whether the executor fetches the website.
        queries_per_iteration: Searches executed per plan round.
        llm_enabled: Whether any node uses a real LLM (drives ``mode``).
    """

    planner: ResearchPlanner
    reflector: Reflector
    profile_builder: ProfileBuilder
    extractor: PersonalizationExtractor
    email_writer: EmailWriter
    scorer: QualityScorer
    web: WebSearch
    news: NewsSearch
    website: WebsiteFetch
    max_reflection_iterations: int = 3
    max_tool_calls: int = 8
    website_fetch_enabled: bool = True
    queries_per_iteration: int = 5
    llm_enabled: bool = False


def build_components(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    judge: LLMClient | None = None,
    web: WebSearch | None = None,
    news: NewsSearch | None = None,
    website: WebsiteFetch | None = None,
    max_reflection_iterations: int | None = None,
) -> AgentComponents:
    """Build the component bundle from settings, with optional overrides.

    Args:
        settings: Application settings (defaults to :func:`get_settings`).
        llm: Worker LLM client override (``None`` keeps the settings-derived one).
        judge: Judge LLM client override.
        web: Web search override.
        news: News search override.
        website: Website fetcher override.
        max_reflection_iterations: Loop budget override (ablation knob).

    Returns:
        The bundle.
    """
    s = settings or get_settings()
    worker = llm if llm is not None else build_llm(s)
    judge_llm = judge if judge is not None else build_llm(s, model=s.judge_model)

    if web is None:
        if s.tavily_api_key:
            logger.info("web search: tavily")
            web = TavilyWebSearch(s.tavily_api_key, timeout_seconds=s.tavily_timeout_seconds)
        else:
            logger.info("web search: stub (no TAVILY_API_KEY)")
            web = StubWebSearch()
    if news is None:
        news = (
            HackerNewsSearch(timeout_seconds=s.hn_timeout_seconds)
            if s.use_real_hn
            else StubHackerNews()
        )
    if website is None:
        website = (
            WebsiteFetcher(timeout_seconds=s.website_timeout_seconds)
            if s.website_fetch_enabled and s.tavily_api_key
            else StubWebsiteFetcher()
        )

    return AgentComponents(
        planner=ResearchPlanner(worker),
        reflector=Reflector(worker),
        profile_builder=ProfileBuilder(worker),
        extractor=PersonalizationExtractor(worker),
        email_writer=EmailWriter(worker),
        scorer=QualityScorer(judge_llm),
        web=web,
        news=news,
        website=website,
        max_reflection_iterations=(
            s.max_reflection_iterations
            if max_reflection_iterations is None
            else max_reflection_iterations
        ),
        max_tool_calls=s.max_tool_calls_per_company,
        website_fetch_enabled=s.website_fetch_enabled,
        llm_enabled=worker is not None,
    )


def _dedupe(evidence: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str]] = set()
    out: list[Evidence] = []
    for e in evidence:
        key = (e.get("url", ""), e.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def build_graph(
    components: AgentComponents | None = None,
    *,
    max_reflection_iterations: int | None = None,
) -> Any:
    """Compile the LangGraph state machine.

    Args:
        components: Component bundle (defaults to :func:`build_components`).
        max_reflection_iterations: Convenience override of the loop budget.

    Returns:
        The compiled graph (``ainvoke``-able).
    """
    c = components or build_components(max_reflection_iterations=max_reflection_iterations)
    if max_reflection_iterations is not None:
        c.max_reflection_iterations = max_reflection_iterations

    @log_node("plan")
    async def plan_node(state: AgentState) -> dict[str, Any]:
        queries = await c.planner.plan(
            state["company_name"],
            state.get("context"),
            missing=state.get("missing") or None,
            already_run=state.get("executed_queries") or None,
        )
        return {"planned_queries": queries, "iteration": state.get("iteration", 0) + 1}

    @log_node("execute")
    async def execute_node(state: AgentState) -> dict[str, Any]:
        evidence = list(state.get("evidence", []))
        executed = list(state.get("executed_queries", []))
        tool_calls = state.get("tool_calls", 0)
        first_round = state.get("iteration", 1) <= 1

        for q in state.get("planned_queries", [])[: c.queries_per_iteration]:
            if tool_calls >= c.max_tool_calls:
                logger.info(f"tool budget exhausted ({c.max_tool_calls}); skipping remaining")
                break
            try:
                evidence.extend(await c.web.search(q, max_results=4))
            except TOOL_ERRORS as exc:
                logger.warning(f"web search failed query={q!r} error={exc!r}")
            tool_calls += 1
            executed.append(q)

        if first_round and tool_calls < c.max_tool_calls:
            try:
                evidence.extend(await c.news.search(state["company_name"], limit=3))
            except TOOL_ERRORS as exc:
                logger.warning(f"news search failed error={exc!r}")
            tool_calls += 1

        site = state.get("website")
        if first_round and c.website_fetch_enabled and site and tool_calls < c.max_tool_calls:
            try:
                evidence.extend(await c.website.fetch(site))
            except TOOL_ERRORS as exc:
                logger.warning(f"website fetch failed site={site!r} error={exc!r}")
            tool_calls += 1

        return {
            "evidence": _dedupe(evidence),
            "tool_calls": tool_calls,
            "executed_queries": executed,
        }

    def route_after_execute(state: AgentState) -> str:
        if c.max_reflection_iterations <= 0:
            return "build"
        return "reflect"

    @log_node("reflect")
    async def reflect_node(state: AgentState) -> dict[str, Any]:
        verdict = await c.reflector.reflect(state["company_name"], state.get("evidence", []))
        return {
            "sufficient": verdict.sufficient,
            "missing": verdict.missing,
            "planned_queries": [] if verdict.sufficient else verdict.follow_up_queries,
        }

    def route_after_reflect(state: AgentState) -> str:
        rounds_done = state.get("iteration", 1)
        if state.get("sufficient"):
            return "build"
        if rounds_done - 1 >= c.max_reflection_iterations:
            return "build"
        if state.get("tool_calls", 0) >= c.max_tool_calls:
            return "build"
        if not state.get("planned_queries") and not state.get("missing"):
            return "build"
        return "plan"

    @log_node("build")
    async def build_profile_node(state: AgentState) -> dict[str, Any]:
        profile = await c.profile_builder.build(
            state["company_name"], state.get("evidence", []), website=state.get("website")
        )
        return {"profile": profile}

    @log_node("extract_hooks")
    async def extract_hooks_node(state: AgentState) -> dict[str, Any]:
        return {"hooks": await c.extractor.extract(state["profile"])}

    @log_node("write_email")
    async def write_email_node(state: AgentState) -> dict[str, Any]:
        return {"email": await c.email_writer.write(state["profile"], state.get("hooks"))}

    @log_node("score")
    async def score_node(state: AgentState) -> dict[str, Any]:
        return {"scores": await c.scorer.score(state["email"], state["profile"])}

    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("reflect", reflect_node)
    g.add_node("build", build_profile_node)
    g.add_node("extract_hooks", extract_hooks_node)
    g.add_node("write_email", write_email_node)
    g.add_node("score", score_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "execute")
    g.add_conditional_edges(
        "execute", route_after_execute, {"reflect": "reflect", "build": "build"}
    )
    g.add_conditional_edges("reflect", route_after_reflect, {"plan": "plan", "build": "build"})
    g.add_edge("build", "extract_hooks")
    g.add_edge("extract_hooks", "write_email")
    g.add_edge("write_email", "score")
    g.add_edge("score", END)
    return g.compile()


async def research_company(
    graph: Any,
    *,
    company_name: str,
    context: str | None = None,
    website: str | None = None,
    trace_id: str | None = None,
    settings: Settings | None = None,
    llm_enabled: bool | None = None,
) -> ResearchResponse:
    """Run the graph for one company and assemble the response.

    Args:
        graph: Compiled graph from :func:`build_graph`.
        company_name: Target company.
        context: Optional seed context.
        website: Optional known website/domain.
        trace_id: Optional externally supplied trace id.
        settings: Settings used for cost computation (defaults to the cached ones).
        llm_enabled: Override of the ``mode`` label; inferred from settings otherwise.

    Returns:
        The response with latency, tokens and cost filled in.
    """
    s = settings or get_settings()
    tracker = start_run(trace_id)
    t0 = time.perf_counter()
    state: AgentState = await graph.ainvoke(
        {"company_name": company_name, "context": context, "website": website, "iteration": 0}
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    mode = "llm" if (s.llm_enabled if llm_enabled is None else llm_enabled) else "fallback"
    from src.observability import current_trace_id

    response = ResearchResponse(
        trace_id=current_trace_id(),
        company_name=company_name,
        profile=state.get("profile", CompanyProfile(name=company_name)),
        hooks=state.get("hooks"),
        email=state.get(
            "email", OutreachEmail(subject="", greeting="", hook="", value_prop="", cta="")
        ),
        scores=state.get("scores"),
        tool_calls=state.get("tool_calls", 0),
        iterations=state.get("iteration", 0),
        latency_ms=latency_ms,
        input_tokens=tracker.input_tokens,
        output_tokens=tracker.output_tokens,
        cost_usd=tracker.cost_usd(
            price_input_per_mtok=s.price_input_per_mtok,
            price_output_per_mtok=s.price_output_per_mtok,
            cost_per_search_usd=s.tavily_cost_per_search_usd,
        ),
        mode=mode,
    )
    logger.info(
        f"research done company={company_name!r} iterations={response.iterations} "
        f"tool_calls={response.tool_calls} latency_ms={latency_ms} "
        f"tokens={response.input_tokens}/{response.output_tokens} cost_usd={response.cost_usd}"
    )
    return response

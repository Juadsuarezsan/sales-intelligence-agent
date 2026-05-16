"""LangGraph plan-execute-reflect loop for the sales intelligence agent."""
from __future__ import annotations

import os
import time
from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger
from typing_extensions import TypedDict

from src.agents.email_writer import EmailWriter
from src.agents.planner import ResearchPlanner
from src.agents.profile_builder import ProfileBuilder
from src.agents.reflector import Reflector
from src.api.schemas import CompanyProfile, OutreachEmail, ResearchResponse
from src.config import get_settings
from src.tools.hackernews import HackerNewsSearch, StubHackerNews
from src.tools.web_search import StubWebSearch, TavilyWebSearch, WebSearch


class AgentState(TypedDict, total=False):
    company_name: str
    context: str | None
    planned_queries: list[str]
    evidence: list[dict]
    iteration: int
    tool_calls: int
    sufficient: bool
    profile: CompanyProfile
    email: OutreachEmail
    started_at: float


def _build_web_search() -> WebSearch:
    s = get_settings()
    if s.tavily_api_key:
        logger.info("Web search: Tavily (real)")
        return TavilyWebSearch(s.tavily_api_key)
    logger.info("Web search: stub (no TAVILY_API_KEY set)")
    return StubWebSearch()


def build_graph():
    s = get_settings()
    planner = ResearchPlanner(model=s.anthropic_model, api_key=s.anthropic_api_key)
    web = _build_web_search()
    hn = HackerNewsSearch() if os.environ.get("USE_REAL_HN", "true").lower() != "false" else StubHackerNews()
    reflector = Reflector(model=s.anthropic_model, api_key=s.anthropic_api_key)
    profile_builder = ProfileBuilder(model=s.anthropic_model, api_key=s.anthropic_api_key)
    email_writer = EmailWriter(model=s.anthropic_model, api_key=s.anthropic_api_key)

    async def plan_node(state: AgentState) -> dict[str, Any]:
        queries = await planner.plan(state["company_name"], state.get("context"))
        return {"planned_queries": queries, "iteration": state.get("iteration", 0) + 1}

    async def execute_node(state: AgentState) -> dict[str, Any]:
        evidence = list(state.get("evidence", []))
        tool_calls = state.get("tool_calls", 0)
        for q in state["planned_queries"][:5]:
            try:
                hits = await web.search(q, max_results=4)
                evidence.extend(hits)
                tool_calls += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"web.search failed for '{q}': {exc}")
        # 1 HN check per company (only on first iteration to avoid spam)
        if state.get("iteration", 0) <= 1:
            try:
                hn_hits = await hn.search(state["company_name"], limit=3)
                for h in hn_hits:
                    evidence.append({"title": h.get("title", ""), "url": h.get("url", ""),
                                      "content": f"HN: {h.get('points')} points",
                                      "score": 0.6})
                tool_calls += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"HN search failed: {exc}")
        return {"evidence": evidence, "tool_calls": tool_calls}

    async def reflect_node(state: AgentState) -> dict[str, Any]:
        verdict = await reflector.reflect(state["company_name"], state.get("evidence", []))
        return {
            "sufficient": verdict.sufficient,
            "planned_queries": verdict.follow_up_queries if not verdict.sufficient else [],
        }

    def route(state: AgentState) -> str:
        if state.get("sufficient"):
            return "build"
        if state.get("iteration", 1) >= s.max_reflection_iterations:
            return "build"
        if not state.get("planned_queries"):
            return "build"
        return "plan"

    async def build_profile_node(state: AgentState) -> dict[str, Any]:
        profile = await profile_builder.build(state["company_name"], state.get("evidence", []))
        return {"profile": profile}

    async def write_email_node(state: AgentState) -> dict[str, Any]:
        email = await email_writer.write(state["profile"])
        return {"email": email}

    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("reflect", reflect_node)
    g.add_node("build", build_profile_node)
    g.add_node("email", write_email_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "reflect")
    g.add_conditional_edges("reflect", route, {"plan": "plan", "build": "build"})
    g.add_edge("build", "email")
    g.add_edge("email", END)
    return g.compile()


async def research_company(graph, *, company_name: str, context: str | None = None) -> ResearchResponse:
    t0 = time.perf_counter()
    state = await graph.ainvoke({"company_name": company_name, "context": context, "iteration": 0})
    return ResearchResponse(
        profile=state.get("profile", CompanyProfile(name=company_name)),
        email=state.get("email", OutreachEmail(subject="", greeting="", hook="", value_prop="", cta="")),
        tool_calls=state.get("tool_calls", 0),
        iterations=state.get("iteration", 0),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )

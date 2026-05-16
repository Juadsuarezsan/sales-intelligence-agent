"""Planner — decides which queries to run for a target company."""
from __future__ import annotations

import json
from tenacity import retry, stop_after_attempt, wait_exponential


SYSTEM = """You plan research queries for a B2B sales intelligence agent.

Given a target company name + optional context, produce a JSON list of 3-6
specific web search queries that will gather: funding info, leadership,
product/positioning, recent news, hiring signals.

Output JSON only, schema:
{ "queries": ["query1", "query2", ...] }

Keep queries short and specific. No quotes inside queries.
"""


class ResearchPlanner:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
    async def plan(self, company_name: str, context: str | None = None) -> list[str]:
        if not self.api_key:
            return self._default_queries(company_name)
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        chat = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0.3, max_tokens=300, timeout=15.0)
        user = f"<company>{company_name}</company>"
        if context:
            user += f"\n<context>{context}</context>"
        resp = await chat.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)])
        body = resp.content if isinstance(resp.content, str) else str(resp.content)
        body = body.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:].lstrip()
        try:
            raw = json.loads(body)
            return [str(q) for q in raw.get("queries", [])][:6]
        except Exception:
            return self._default_queries(company_name)

    @staticmethod
    def _default_queries(company_name: str) -> list[str]:
        return [
            f"{company_name} recent funding round",
            f"{company_name} CEO founder leadership",
            f"{company_name} product what does it do",
            f"{company_name} hiring engineering team",
            f"{company_name} news 2026",
        ]

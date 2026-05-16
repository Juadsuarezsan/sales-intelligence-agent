"""Profile builder — turns evidence into a CompanyProfile via Claude + Pydantic."""
from __future__ import annotations

import json
from typing import Sequence

from src.api.schemas import CompanyProfile, KeyPerson, Signal


SYSTEM = """You build a structured B2B company profile from web evidence.

Output ONLY valid JSON matching this schema:
{
  "name": "...",
  "one_liner": "10-15 word description",
  "industry": "...",
  "employees_est": <int> | null,
  "funding_total_usd": <number> | null,
  "last_funding_round": "Series B" | "Seed" | null,
  "location": "City, Country" | null,
  "key_people": [{"name":"...", "role":"...", "linkedin_url":null}],
  "recent_signals": [{"kind":"funding|hiring|product_launch|news|leadership_change|other",
                      "description":"...", "source_url":"...", "occurred_at":null}],
  "pain_points": ["..."],
  "sources": ["url1", "url2"]
}

Never invent data not present in the evidence. If a field is unknown, set null
or empty list.
"""


class ProfileBuilder:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    async def build(self, company_name: str, evidence: Sequence[dict]) -> CompanyProfile:
        if not self.api_key:
            return self._heuristic(company_name, list(evidence))
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        chat = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0, max_tokens=1200, timeout=20.0)
        ev = "\n".join(
            f"- title: {e.get('title','')}\n  url: {e.get('url','')}\n  content: {e.get('content','')[:300]}"
            for e in evidence[:12]
        )
        user = f"<company>{company_name}</company>\n<evidence>\n{ev}\n</evidence>"
        resp = await chat.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)])
        body = resp.content if isinstance(resp.content, str) else str(resp.content)
        body = body.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:].lstrip()
        try:
            raw = json.loads(body)
            return CompanyProfile(**raw)
        except Exception:
            return self._heuristic(company_name, list(evidence))

    @staticmethod
    def _heuristic(company_name: str, evidence: list[dict]) -> CompanyProfile:
        # Extract whatever signals we can without LLM
        sources = list(dict.fromkeys(e.get("url") for e in evidence if e.get("url")))
        signals = []
        for e in evidence[:4]:
            content = (e.get("content") or "").lower()
            kind = (
                "funding" if any(k in content for k in ("raised", "series ", "round"))
                else "leadership_change" if any(k in content for k in ("appoint", "promot"))
                else "product_launch" if any(k in content for k in ("launch", "announce"))
                else "news"
            )
            signals.append(Signal(
                kind=kind,
                description=e.get("title", "")[:160],
                source_url=e.get("url", ""),
            ))
        return CompanyProfile(
            name=company_name,
            one_liner="(profile built from heuristics — connect Anthropic API for deeper analysis)",
            sources=sources,
            recent_signals=signals,
        )

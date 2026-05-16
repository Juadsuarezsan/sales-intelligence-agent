"""Reflector — decides whether collected evidence is sufficient."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ReflectionVerdict:
    sufficient: bool
    missing: list[str]
    follow_up_queries: list[str]
    rationale: str


class Reflector:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    async def reflect(self, company_name: str, evidence: Sequence[dict]) -> ReflectionVerdict:
        if not self.api_key:
            return self._heuristic(company_name, list(evidence))
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        SYSTEM = """You review research evidence and decide if it is enough to write
        a personalized B2B outreach email. Output JSON only:

        {
          "sufficient": true | false,
          "missing": ["funding", "leadership", "product", ...],
          "follow_up_queries": ["...", "..."],
          "rationale": "one sentence"
        }

        Required for sufficiency: at least one of (funding OR product description)
        AND at least one of (leadership name OR recent signal).
        """
        chat = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0, max_tokens=300, timeout=15.0)
        ev = "\n".join(f"- {e.get('title','')}: {e.get('content','')[:160]}" for e in evidence[:10])
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
            return ReflectionVerdict(
                sufficient=bool(raw.get("sufficient", False)),
                missing=list(raw.get("missing", [])),
                follow_up_queries=list(raw.get("follow_up_queries", []))[:3],
                rationale=str(raw.get("rationale", "")),
            )
        except Exception:
            return self._heuristic(company_name, list(evidence))

    @staticmethod
    def _heuristic(company_name: str, evidence: list[dict]) -> ReflectionVerdict:
        text = " ".join(f"{e.get('title','')} {e.get('content','')}" for e in evidence).lower()
        has_funding = any(w in text for w in ("funding", "raised", "series ", "$"))
        has_leader = any(w in text for w in ("ceo", "founder", "cto"))
        has_product = any(w in text for w in ("platform", "saas", "product", "service"))
        sufficient = (has_funding or has_product) and (has_leader or len(evidence) >= 3)
        missing = []
        if not has_funding: missing.append("funding")
        if not has_leader:  missing.append("leadership")
        if not has_product: missing.append("product")
        follow_ups = [f"{company_name} {m}" for m in missing[:2]]
        return ReflectionVerdict(
            sufficient=sufficient, missing=missing, follow_up_queries=follow_ups,
            rationale=f"heuristic: funding={has_funding} leader={has_leader} product={has_product}",
        )

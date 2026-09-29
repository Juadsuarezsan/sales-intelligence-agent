"""Reflector node: decides whether the collected evidence is sufficient."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from loguru import logger

from src.api.schemas import Evidence
from src.llm import LLMClient, parse_json_object, recoverable_llm_errors

SYSTEM = """You review research evidence about a company and decide whether it is
enough to write a personalized B2B outreach email.

Sufficiency rule: at least one of (funding OR product description) AND at least
one of (leadership name OR recent signal such as hiring, launch, news).
Also flag contradictions between sources (e.g. two different headcounts).

Output JSON only:
{
  "sufficient": true | false,
  "missing": ["funding", "leadership", "product", "recent_signal"],
  "contradictions": ["..."],
  "follow_up_queries": ["...", "..."],
  "rationale": "one sentence"
}
"""

_FUNDING = ("funding", "raised", "series ", "seed round", "$")
_LEADER = ("ceo", "founder", "cto", "chief executive")
_PRODUCT = ("platform", "saas", "product", "service", "software", "api")
_SIGNAL = ("hiring", "open roles", "launch", "announce", "hn:", "points")


@dataclass
class ReflectionVerdict:
    """Outcome of one reflection step.

    Attributes:
        sufficient: Whether the loop can stop searching.
        missing: Topics still uncovered.
        follow_up_queries: Queries suggested for the next iteration.
        rationale: Short justification.
        contradictions: Conflicting facts spotted across sources.
    """

    sufficient: bool
    missing: list[str]
    follow_up_queries: list[str]
    rationale: str
    contradictions: list[str] = field(default_factory=list)


class Reflector:
    """Judges evidence sufficiency with the LLM or a keyword heuristic.

    Args:
        llm: Completion client; ``None`` selects the offline heuristic.
    """

    def __init__(self, llm: LLMClient | None) -> None:
        self.llm = llm

    async def reflect(self, company_name: str, evidence: Sequence[Evidence]) -> ReflectionVerdict:
        """Evaluate the evidence gathered so far.

        Args:
            company_name: Target company.
            evidence: Evidence items collected by the executor.

        Returns:
            The verdict.
        """
        if self.llm is None:
            return self.heuristic(company_name, list(evidence))
        lines = "\n".join(
            f"- [{e.get('source', '?')}] {e.get('title', '')}: {e.get('content', '')[:200]}"
            for e in evidence[:15]
        )
        user = f"<company>{company_name}</company>\n<evidence>\n{lines}\n</evidence>"
        try:
            result = await self.llm.complete(system=SYSTEM, user=user, max_tokens=400)
            raw = parse_json_object(result.text)
            return ReflectionVerdict(
                sufficient=bool(raw.get("sufficient", False)),
                missing=[str(m) for m in raw.get("missing", [])],
                follow_up_queries=[str(q) for q in raw.get("follow_up_queries", [])][:3],
                rationale=str(raw.get("rationale", "")),
                contradictions=[str(c) for c in raw.get("contradictions", [])],
            )
        except recoverable_llm_errors() as exc:
            logger.warning(f"reflector LLM failed, using heuristic: {exc!r}")
            return self.heuristic(company_name, list(evidence))

    @staticmethod
    def heuristic(company_name: str, evidence: list[Evidence]) -> ReflectionVerdict:
        """Keyword-based sufficiency check used offline and as fallback.

        Args:
            company_name: Target company.
            evidence: Evidence items.

        Returns:
            The verdict with ``rationale`` prefixed by ``heuristic:``.
        """
        text = " ".join(f"{e.get('title', '')} {e.get('content', '')}" for e in evidence).lower()
        has_funding = any(w in text for w in _FUNDING)
        has_leader = any(w in text for w in _LEADER)
        has_product = any(w in text for w in _PRODUCT)
        has_signal = any(w in text for w in _SIGNAL)
        sufficient = (has_funding or has_product) and (has_leader or has_signal)
        missing: list[str] = []
        if not has_funding:
            missing.append("funding")
        if not has_leader:
            missing.append("leadership")
        if not has_product:
            missing.append("product")
        if not has_signal:
            missing.append("recent signal")
        return ReflectionVerdict(
            sufficient=sufficient,
            missing=missing,
            follow_up_queries=[f"{company_name} {m}" for m in missing[:2]],
            rationale=(
                f"heuristic: funding={has_funding} leader={has_leader} "
                f"product={has_product} signal={has_signal}"
            ),
        )

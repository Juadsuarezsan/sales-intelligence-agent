"""Planner node: decides which search queries to run for a target company."""

from __future__ import annotations

from loguru import logger

from src.llm import LLMClient, parse_json_object, recoverable_llm_errors

SYSTEM = """You plan research queries for a B2B sales intelligence agent.

Given a target company name, optional context and the list of information that
is still missing, produce 3-6 specific web search queries that will gather:
funding, leadership, product/positioning, recent news and hiring signals.

Output JSON only:
{ "queries": ["query 1", "query 2", "..."] }

Keep queries short and specific. No quotes inside queries. Never repeat a
query that was already run.
"""

_DEFAULT_TEMPLATES: tuple[str, ...] = (
    "{name} recent funding round",
    "{name} CEO founder leadership",
    "{name} product what does it do",
    "{name} hiring engineering team",
    "{name} news 2026",
)


class ResearchPlanner:
    """Produces search queries with the LLM or a deterministic template set.

    Args:
        llm: Completion client; ``None`` selects the offline heuristic.
    """

    def __init__(self, llm: LLMClient | None) -> None:
        self.llm = llm

    async def plan(
        self,
        company_name: str,
        context: str | None = None,
        *,
        missing: list[str] | None = None,
        already_run: list[str] | None = None,
    ) -> list[str]:
        """Return the next batch of queries.

        Args:
            company_name: Target company.
            context: Optional seed context from the caller (domain, notes).
            missing: Information gaps reported by the reflector.
            already_run: Queries executed in previous iterations.

        Returns:
            Between 1 and 6 queries.
        """
        if self.llm is None:
            return self.default_queries(company_name, missing=missing, already_run=already_run)
        user = f"<company>{company_name}</company>"
        if context:
            user += f"\n<context>{context}</context>"
        if missing:
            user += f"\n<missing>{', '.join(missing)}</missing>"
        if already_run:
            user += "\n<already_run>\n" + "\n".join(already_run) + "\n</already_run>"
        try:
            result = await self.llm.complete(
                system=SYSTEM, user=user, max_tokens=300, temperature=0.3
            )
            raw = parse_json_object(result.text)
            queries = [str(q).strip() for q in raw.get("queries", []) if str(q).strip()]
        except recoverable_llm_errors() as exc:
            logger.warning(f"planner LLM failed, using default queries: {exc!r}")
            return self.default_queries(company_name, missing=missing, already_run=already_run)
        if not queries:
            logger.warning("planner LLM returned no queries, using defaults")
            return self.default_queries(company_name, missing=missing, already_run=already_run)
        return queries[:6]

    @staticmethod
    def default_queries(
        company_name: str,
        *,
        missing: list[str] | None = None,
        already_run: list[str] | None = None,
    ) -> list[str]:
        """Deterministic query set used offline and as fallback.

        Args:
            company_name: Target company.
            missing: If given, one follow-up query per missing topic is produced.
            already_run: Queries to exclude.

        Returns:
            Query list (never empty unless everything was already run).
        """
        seen = set(already_run or [])
        if missing:
            follow_ups = [f"{company_name} {topic}" for topic in missing[:3]]
            fresh = [q for q in follow_ups if q not in seen]
            if fresh:
                return fresh
        base = [t.format(name=company_name) for t in _DEFAULT_TEMPLATES]
        fresh = [q for q in base if q not in seen]
        return fresh or base[:1]

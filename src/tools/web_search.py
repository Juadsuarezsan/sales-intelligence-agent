"""Web search tool: Tavily in production, deterministic stub offline.

Both implementations satisfy the :class:`WebSearch` protocol and return a
list of :class:`~src.api.schemas.Evidence` dictionaries so the rest of the
graph never depends on the provider's response shape.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.api.schemas import Evidence
from src.observability import current_usage


class WebSearch(Protocol):
    """Provider-agnostic web search."""

    name: str

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        """Search the web.

        Args:
            query: Free-text query.
            max_results: Upper bound on returned results.

        Returns:
            Evidence items ordered by relevance.
        """
        ...


class TavilyWebSearch:
    """Tavily-backed search with an explicit timeout and bounded retries.

    Args:
        api_key: Tavily API key.
        timeout_seconds: Wall-clock timeout per search (Tavily's client has none).
    """

    name = "tavily"

    def __init__(self, api_key: str, *, timeout_seconds: float = 10.0) -> None:
        from tavily import AsyncTavilyClient

        self._client = AsyncTavilyClient(api_key=api_key)
        self.timeout_seconds = timeout_seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError, OSError)),
        reraise=True,
    )
    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        """Run one Tavily search.

        Args:
            query: Free-text query.
            max_results: Upper bound on returned results.

        Returns:
            Evidence items with ``source="web"``.

        Raises:
            asyncio.TimeoutError: If Tavily does not answer within the timeout
                after all retries.
        """
        out: dict[str, Any] = await asyncio.wait_for(
            self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
            ),
            timeout=self.timeout_seconds,
        )
        tracker = current_usage()
        if tracker is not None:
            tracker.record_search()
        results: list[Evidence] = []
        for r in out.get("results", []):
            results.append(
                Evidence(
                    title=str(r.get("title", "")),
                    url=str(r.get("url", "")),
                    content=str(r.get("content", "")),
                    score=float(r.get("score", 0.0)),
                    source="web",
                )
            )
        logger.debug(f"tavily query={query!r} results={len(results)}")
        return results


class StubWebSearch:
    """Deterministic, offline search results keyed on query keywords.

    The stub is what CI, the notebook and the offline eval run use. Its
    results are canned text and do not describe the real company, which is
    why every metric computed on top of it is labelled "fallback".
    """

    name = "stub"

    async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        """Return canned results matching keywords in ``query``.

        Args:
            query: Free-text query.
            max_results: Upper bound on returned results.

        Returns:
            At least one evidence item.
        """
        q = query.lower()
        canned: list[Evidence] = []
        if "funding" in q or "series" in q or "raised" in q:
            canned.append(
                Evidence(
                    title=f"{query.title()} raises Series B",
                    url="https://techcrunch.com/example-funding",
                    content=(
                        "In a recent Series B round, the company raised $40M led by "
                        "Andreessen Horowitz to expand its go-to-market team."
                    ),
                    score=0.91,
                    source="web",
                )
            )
        if "ceo" in q or "founder" in q or "leadership" in q:
            canned.append(
                Evidence(
                    title=f"Leadership of {query}",
                    url="https://example.com/about",
                    content=(
                        "The CEO is John Doe (former VP Engineering at Acme). "
                        "The CTO is Jane Smith."
                    ),
                    score=0.85,
                    source="web",
                )
            )
        if "product" in q or "what does" in q:
            canned.append(
                Evidence(
                    title=f"{query.title()} product overview",
                    url="https://example.com/product",
                    content=(
                        "The company offers a SaaS platform for AI workflows targeting "
                        "mid-market enterprises."
                    ),
                    score=0.88,
                    source="web",
                )
            )
        if "hiring" in q or "careers" in q:
            canned.append(
                Evidence(
                    title=f"{query.title()} is hiring",
                    url="https://example.com/careers",
                    content="Open roles: 6 engineering positions including a Head of Platform.",
                    score=0.80,
                    source="web",
                )
            )
        if not canned:
            canned.append(
                Evidence(
                    title=f"About {query}",
                    url="https://example.com",
                    content=f"Generic placeholder result for query: {query}",
                    score=0.45,
                    source="web",
                )
            )
        return canned[:max_results]

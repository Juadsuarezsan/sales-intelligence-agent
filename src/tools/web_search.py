"""Web search tool — Tavily in production, deterministic stub for tests."""
from __future__ import annotations

from typing import Protocol

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class WebSearch(Protocol):
    name: str
    async def search(self, query: str, max_results: int = 5) -> list[dict]: ...


class TavilyWebSearch:
    name = "tavily"

    def __init__(self, api_key: str) -> None:
        from tavily import AsyncTavilyClient
        self.client = AsyncTavilyClient(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        out = await self.client.search(query=query, max_results=max_results,
                                         search_depth="basic", include_answer=False)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "content": r.get("content", ""), "score": float(r.get("score", 0.0))}
            for r in out.get("results", [])
        ]


class StubWebSearch:
    """Deterministic fake results — for offline tests + CI."""

    name = "stub"

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        q = query.lower()
        canned: list[dict] = []
        if "funding" in q or "series" in q:
            canned.append({
                "title": f"{query.title()} raises Series B",
                "url": "https://techcrunch.com/example-funding",
                "content": f"In a recent Series B round, the company raised $40M led by Andreessen Horowitz.",
                "score": 0.91,
            })
        if "ceo" in q or "founder" in q:
            canned.append({
                "title": f"Leadership of {query}",
                "url": "https://example.com/about",
                "content": f"The CEO is John Doe (former VP Engineering at Acme). The CTO is Jane Smith.",
                "score": 0.85,
            })
        if "product" in q or "what does" in q:
            canned.append({
                "title": f"{query.title()} product overview",
                "url": "https://example.com/product",
                "content": f"The company offers a SaaS platform for AI workflows targeting mid-market enterprises.",
                "score": 0.88,
            })
        # always return at least one result so the agent can progress
        if not canned:
            canned.append({
                "title": f"About {query}",
                "url": "https://example.com",
                "content": f"Generic placeholder result for query: {query}",
                "score": 0.45,
            })
        return canned[:max_results]

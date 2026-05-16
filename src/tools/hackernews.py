"""HackerNews Algolia API — free, no auth. Returns recent mentions of a company."""
from __future__ import annotations

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class HackerNewsSearch:
    BASE = "https://hn.algolia.com/api/v1/search"

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
    async def search(self, company: str, limit: int = 5) -> list[dict]:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(self.BASE, params={"query": company, "tags": "story", "hitsPerPage": limit})
            r.raise_for_status()
            data = r.json()
        return [
            {"title": h.get("title", ""), "url": h.get("url", ""),
             "author": h.get("author"), "points": h.get("points", 0),
             "created_at": h.get("created_at", "")}
            for h in data.get("hits", [])
        ]


class StubHackerNews:
    async def search(self, company: str, limit: int = 5) -> list[dict]:
        return [
            {"title": f"Show HN: {company} launches new feature",
             "url": "https://news.ycombinator.com/item?id=12345",
             "author": "demo_user", "points": 142, "created_at": "2026-04-15"},
            {"title": f"Ask HN: experiences with {company}?",
             "url": "https://news.ycombinator.com/item?id=12346",
             "author": "another", "points": 88, "created_at": "2026-03-02"},
        ][:limit]

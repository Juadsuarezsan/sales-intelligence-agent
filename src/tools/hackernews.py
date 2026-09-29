"""HackerNews mentions through the free Algolia search API."""

from __future__ import annotations

from typing import Protocol

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.api.schemas import Evidence


class NewsSearch(Protocol):
    """Provider-agnostic news/mention search."""

    name: str

    async def search(self, company: str, limit: int = 5) -> list[Evidence]:
        """Return recent mentions of ``company``."""
        ...


class HackerNewsSearch:
    """HN Algolia client with explicit timeout and bounded retries.

    Args:
        timeout_seconds: HTTP timeout per request.
        base_url: Override for tests.
    """

    name = "algolia"
    BASE = "https://hn.algolia.com/api/v1/search"

    def __init__(self, *, timeout_seconds: float = 8.0, base_url: str | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url or self.BASE

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def search(self, company: str, limit: int = 5) -> list[Evidence]:
        """Search HN stories mentioning ``company``.

        Args:
            company: Company name.
            limit: Maximum hits.

        Returns:
            Evidence items with ``source="hn"``.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
            httpx.TransportError: When the network fails after retries.
        """
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            r = await client.get(
                self.base_url,
                params={"query": company, "tags": "story", "hitsPerPage": limit},
            )
            r.raise_for_status()
            data = r.json()
        hits: list[Evidence] = []
        for h in data.get("hits", []):
            points = int(h.get("points") or 0)
            hits.append(
                Evidence(
                    title=str(h.get("title") or ""),
                    url=str(h.get("url") or ""),
                    content=f"HN: {points} points, posted {h.get('created_at', '')[:10]}",
                    score=min(1.0, 0.5 + points / 500),
                    source="hn",
                )
            )
        logger.debug(f"hn company={company!r} hits={len(hits)}")
        return hits


class StubHackerNews:
    """Deterministic HN results for offline runs."""

    name = "stub"

    async def search(self, company: str, limit: int = 5) -> list[Evidence]:
        """Return two canned stories.

        Args:
            company: Company name.
            limit: Maximum hits.

        Returns:
            Canned evidence items with ``source="hn"``.
        """
        return [
            Evidence(
                title=f"Show HN: {company} launches new feature",
                url="https://news.ycombinator.com/item?id=12345",
                content="HN: 142 points, posted 2026-04-15",
                score=0.78,
                source="hn",
            ),
            Evidence(
                title=f"Ask HN: experiences with {company}?",
                url="https://news.ycombinator.com/item?id=12346",
                content="HN: 88 points, posted 2026-03-02",
                score=0.68,
                source="hn",
            ),
        ][:limit]

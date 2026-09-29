import httpx
import pytest
import respx

from src.tools.hackernews import HackerNewsSearch, StubHackerNews

URL = "https://hn.algolia.com/api/v1/search"


@respx.mock
async def test_hackernews_maps_hits() -> None:
    respx.get(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "hits": [
                    {
                        "title": "Show HN: Acme",
                        "url": "https://acme.com",
                        "points": 250,
                        "created_at": "2026-05-01T10:00:00Z",
                    },
                    {"title": "Ask HN: Acme?", "url": None, "points": None},
                ]
            },
        )
    )
    hits = await HackerNewsSearch(timeout_seconds=1).search("Acme", limit=2)
    assert hits[0]["source"] == "hn" and hits[0]["score"] == 1.0
    assert "posted 2026-05-01" in hits[0]["content"]
    assert hits[1]["url"] == "" and hits[1]["score"] == 0.5


@respx.mock
async def test_hackernews_raises_on_http_error() -> None:
    respx.get(URL).mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        await HackerNewsSearch(timeout_seconds=1).search("Acme")


async def test_stub_hackernews_limit() -> None:
    hits = await StubHackerNews().search("Acme", limit=1)
    assert len(hits) == 1 and "Acme" in hits[0]["title"]

import pytest
from src.tools.web_search import StubWebSearch


@pytest.mark.asyncio
async def test_funding_query_returns_funding_result():
    s = StubWebSearch()
    out = await s.search("Acme Corp recent funding round")
    assert len(out) >= 1
    assert any("series" in r["content"].lower() or "$" in r["content"] for r in out)


@pytest.mark.asyncio
async def test_always_returns_at_least_one_result():
    s = StubWebSearch()
    out = await s.search("totally random query")
    assert len(out) >= 1

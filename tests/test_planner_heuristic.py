import pytest
from src.agents.planner import ResearchPlanner


@pytest.mark.asyncio
async def test_default_queries_cover_funding_leadership_product():
    p = ResearchPlanner(model="claude-sonnet-4-5", api_key=None)
    queries = await p.plan("Acme Corp")
    joined = " ".join(queries).lower()
    assert "funding" in joined
    assert "ceo" in joined or "founder" in joined or "leadership" in joined
    assert "product" in joined
    assert 3 <= len(queries) <= 6

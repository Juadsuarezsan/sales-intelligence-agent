import pytest
from src.agents.reflector import Reflector


@pytest.mark.asyncio
async def test_sufficient_when_funding_and_leader_present():
    r = Reflector(model="x", api_key=None)
    evidence = [
        {"title": "Series B funding", "content": "raised $40M led by Andreessen Horowitz"},
        {"title": "About", "content": "CEO is John Doe, former VP at Acme"},
        {"title": "Product", "content": "a SaaS platform for AI workflows"},
    ]
    v = await r.reflect("TestCo", evidence)
    assert v.sufficient is True


@pytest.mark.asyncio
async def test_insufficient_with_no_evidence():
    r = Reflector(model="x", api_key=None)
    v = await r.reflect("TestCo", [])
    assert v.sufficient is False
    assert "funding" in v.missing
    assert "leadership" in v.missing
    assert "product" in v.missing


@pytest.mark.asyncio
async def test_follow_up_queries_for_missing():
    r = Reflector(model="x", api_key=None)
    v = await r.reflect("TestCo", [{"title": "x", "content": "founded by someone"}])
    assert v.follow_up_queries

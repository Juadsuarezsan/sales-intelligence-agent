import pytest
from src.agents.orchestrator import build_graph, research_company


@pytest.mark.asyncio
async def test_full_research_returns_profile_and_email(monkeypatch):
    # Force stub HN to avoid hitting the real API in CI
    monkeypatch.setenv("USE_REAL_HN", "false")
    graph = build_graph()
    out = await research_company(graph, company_name="Anthropic")
    assert out.profile.name == "Anthropic"
    assert out.email.subject
    assert out.iterations >= 1
    assert out.tool_calls >= 1

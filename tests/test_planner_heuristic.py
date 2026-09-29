from src.agents.planner import ResearchPlanner
from tests.conftest import FakeLLM, llm_json


async def test_default_queries_cover_funding_leadership_product() -> None:
    p = ResearchPlanner(llm=None)
    queries = await p.plan("Acme Corp")
    joined = " ".join(queries).lower()
    assert "funding" in joined
    assert "ceo" in joined or "founder" in joined or "leadership" in joined
    assert "product" in joined
    assert 3 <= len(queries) <= 6


async def test_default_follow_ups_target_missing_topics_and_skip_repeats() -> None:
    p = ResearchPlanner(llm=None)
    queries = await p.plan("Acme", missing=["funding", "leadership"], already_run=["Acme funding"])
    assert queries == ["Acme leadership"]


async def test_llm_queries_are_used_when_valid() -> None:
    llm = FakeLLM([llm_json(queries=["Acme Series B 2026", "Acme CEO interview", ""])])
    queries = await ResearchPlanner(llm).plan("Acme", context="robotics")
    assert queries == ["Acme Series B 2026", "Acme CEO interview"]
    assert "<context>robotics</context>" in llm.calls[0]["user"]


async def test_llm_garbage_falls_back_to_defaults() -> None:
    queries = await ResearchPlanner(FakeLLM(["not json at all"])).plan("Acme")
    assert queries == ResearchPlanner.default_queries("Acme")


async def test_llm_empty_list_falls_back_to_defaults() -> None:
    queries = await ResearchPlanner(FakeLLM([llm_json(queries=[])])).plan("Acme")
    assert len(queries) == 5

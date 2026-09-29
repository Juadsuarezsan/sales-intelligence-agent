from src.agents.reflector import Reflector
from src.api.schemas import Evidence
from tests.conftest import FakeLLM, llm_json


async def test_sufficient_when_funding_and_leader_present() -> None:
    r = Reflector(llm=None)
    evidence = [
        Evidence(title="Series B funding", content="raised $40M led by Andreessen Horowitz"),
        Evidence(title="About", content="CEO is John Doe, former VP at Acme"),
        Evidence(title="Product", content="a SaaS platform for AI workflows"),
    ]
    v = await r.reflect("TestCo", evidence)
    assert v.sufficient is True


async def test_insufficient_with_no_evidence() -> None:
    v = await Reflector(llm=None).reflect("TestCo", [])
    assert v.sufficient is False
    assert {"funding", "leadership", "product"} <= set(v.missing)


async def test_follow_up_queries_for_missing() -> None:
    v = await Reflector(llm=None).reflect(
        "TestCo", [Evidence(title="x", content="founded by someone")]
    )
    assert v.follow_up_queries


async def test_llm_verdict_is_parsed() -> None:
    llm = FakeLLM(
        [
            llm_json(
                sufficient=False,
                missing=["funding"],
                contradictions=["headcount 50 vs 500"],
                follow_up_queries=["TestCo funding", "TestCo raise", "x", "y"],
                rationale="no funding info",
            )
        ]
    )
    v = await Reflector(llm).reflect("TestCo", [Evidence(title="t", content="c")])
    assert v.sufficient is False
    assert v.missing == ["funding"]
    assert v.contradictions == ["headcount 50 vs 500"]
    assert len(v.follow_up_queries) == 3


async def test_llm_failure_falls_back_to_heuristic() -> None:
    v = await Reflector(FakeLLM(["<html>"])).reflect("TestCo", [])
    assert v.rationale.startswith("heuristic:")

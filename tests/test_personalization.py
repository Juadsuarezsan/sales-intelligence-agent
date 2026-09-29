from src.agents.personalization import PersonalizationExtractor
from src.api.schemas import CompanyProfile, Signal
from tests.conftest import FakeLLM, llm_json


async def test_heuristic_prefers_funding_signal_and_industry_pain(profile: CompanyProfile) -> None:
    hooks = await PersonalizationExtractor(llm=None).extract(profile)
    assert hooks.recent_signal is not None and hooks.recent_signal.startswith("funding:")
    assert hooks.evidence_refs == ["https://example.com/funding"]
    assert "go-to-market" in hooks.pain_point  # B2B industry map
    assert "Anthropic" in hooks.angle


async def test_heuristic_without_signals_uses_generic_pain() -> None:
    p = CompanyProfile(name="Quiet Co")
    hooks = await PersonalizationExtractor(llm=None).extract(p)
    assert hooks.recent_signal is None
    assert hooks.pain_point


async def test_heuristic_hiring_only_signal() -> None:
    p = CompanyProfile(
        name="Hiring Co",
        recent_signals=[Signal(kind="hiring", description="10 open roles", source_url="")],
    )
    hooks = await PersonalizationExtractor(llm=None).extract(p)
    assert hooks.recent_signal == "hiring: 10 open roles"
    assert "growing team" in hooks.pain_point
    assert hooks.evidence_refs == []


async def test_llm_hooks_parsed_and_invalid_falls_back(profile: CompanyProfile) -> None:
    good = FakeLLM([llm_json(pain_point="p", recent_signal="s", angle="a", evidence_refs=["u"])])
    hooks = await PersonalizationExtractor(good).extract(profile)
    assert (hooks.pain_point, hooks.recent_signal, hooks.angle) == ("p", "s", "a")

    bad = FakeLLM([llm_json(recent_signal="missing pain point")])
    hooks = await PersonalizationExtractor(bad).extract(profile)
    assert hooks.recent_signal is not None and hooks.recent_signal.startswith("funding:")

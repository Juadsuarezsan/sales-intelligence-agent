from src.agents.quality_scorer import JUDGE_SYSTEM, RUBRIC, QualityScorer
from src.api.schemas import CompanyProfile, OutreachEmail
from tests.conftest import FakeLLM, llm_json


def _email(hook: str, cta: str, value_prop: str = "We build workflow tooling.") -> OutreachEmail:
    return OutreachEmail(subject="s", greeting="Hi,", hook=hook, value_prop=value_prop, cta=cta)


def test_rubric_is_numbered_with_anchors() -> None:
    for criterion in ("1. PERSONALIZATION", "2. ACCURACY", "3. CTA_CLARITY"):
        assert criterion in RUBRIC
    assert RUBRIC.count("5 =") == 3 and RUBRIC.count("1 =") == 3
    assert "Example 5" in RUBRIC and RUBRIC in JUDGE_SYSTEM


async def test_heuristic_scores_generic_email_low(profile: CompanyProfile) -> None:
    s = await QualityScorer(llm=None).score(_email("Quick intro.", "Thoughts?"), profile)
    assert s.judge == "heuristic"
    assert s.personalization == 1
    assert s.cta_clarity == 3  # a question mark but no format/time
    assert s.accuracy == 5  # no factual claims


async def test_heuristic_scores_specific_accurate_email_high(profile: CompanyProfile) -> None:
    s = await QualityScorer(llm=None).score(
        _email(
            "Congrats on Anthropic's $2B Series C and the 6 open roles.",
            "Worth a 20-minute call next week?",
            "B2B teams like yours automate research with us.",
        ),
        profile,
    )
    assert s.personalization == 5 and s.accuracy == 4 and s.cta_clarity == 4
    assert s.overall == round((5 + 4 + 4) / 3, 2)


async def test_heuristic_penalises_unsupported_claims(profile: CompanyProfile) -> None:
    s = await QualityScorer(llm=None).score(
        _email("Congrats on the $900M Series D!", "Call?"), profile
    )
    assert s.accuracy == 3
    s2 = await QualityScorer(llm=None).score(
        _email("Congrats on the $900M Series D and $1B Series E!", ""), profile
    )
    assert s2.accuracy == 1 and s2.cta_clarity == 1


async def test_llm_judge_scores_are_parsed(profile: CompanyProfile) -> None:
    llm = FakeLLM([llm_json(personalization=4, accuracy=5, cta_clarity=3, rationale="ok")])
    s = await QualityScorer(llm).score(_email("h", "c"), profile)
    assert s.judge == "llm" and (s.personalization, s.accuracy, s.cta_clarity) == (4, 5, 3)
    assert s.overall == 4.0


async def test_llm_judge_malformed_falls_back(profile: CompanyProfile) -> None:
    missing_key = FakeLLM([llm_json(personalization=4)])
    s = await QualityScorer(missing_key).score(_email("h", "c"), profile)
    assert s.judge == "heuristic"
    out_of_range = FakeLLM([llm_json(personalization=9, accuracy=5, cta_clarity=3)])
    s = await QualityScorer(out_of_range).score(_email("h", "c"), profile)
    assert s.judge == "heuristic"

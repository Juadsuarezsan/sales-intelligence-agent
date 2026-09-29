from src.agents.email_writer import EmailWriter, score_personalization
from src.api.schemas import CompanyProfile, OutreachEmail, PersonalizationHooks
from tests.conftest import FakeLLM, llm_json


def test_high_personalization_when_email_mentions_facts(profile: CompanyProfile) -> None:
    email = OutreachEmail(
        subject="AI safety partnership",
        greeting="Hi team,",
        hook="Saw Anthropic's recent $2B Series C funding round.",
        value_prop="We help B2B teams ship workflow tooling 30% faster.",
        cta="Open to a 20-min chat?",
    )
    assert score_personalization(email, profile) >= 0.4


def test_low_personalization_for_generic_email(profile: CompanyProfile) -> None:
    email = OutreachEmail(
        subject="generic outreach",
        greeting="Hi team,",
        hook="Quick intro.",
        value_prop="We build software for businesses.",
        cta="Coffee?",
    )
    assert score_personalization(email, profile) < 0.3


async def test_template_email_falls_back_without_llm(profile: CompanyProfile) -> None:
    e = await EmailWriter(llm=None).write(profile)
    assert e.subject and e.cta
    assert e.greeting == "Hi Dario,"
    assert e.body().count("\n\n") == 3


async def test_template_uses_hooks(profile: CompanyProfile) -> None:
    hooks = PersonalizationHooks(pain_point="manual QA", recent_signal="funding: $2B Series C")
    e = EmailWriter.template(profile, hooks)
    assert "$2B Series C" in e.hook
    assert "manual QA" in e.value_prop
    assert e.personalization_score > 0


async def test_llm_email_is_validated_and_scored(profile: CompanyProfile) -> None:
    llm = FakeLLM(
        [
            llm_json(
                subject="Series C congrats",
                greeting="Hi Dario,",
                hook="Congrats on Anthropic's $2B Series C.",
                value_prop="We help B2B teams automate research.",
                cta="20 minutes next week?",
            )
        ]
    )
    e = await EmailWriter(llm, offering="a research copilot").write(profile)
    assert e.subject == "Series C congrats"
    assert e.personalization_score >= 0.4
    assert "<our_offering>a research copilot</our_offering>" in llm.calls[0]["user"]


async def test_llm_invalid_email_falls_back_to_template(profile: CompanyProfile) -> None:
    e = await EmailWriter(FakeLLM([llm_json(subject="only subject")])).write(profile)
    assert e.cta == "Worth a 20-minute conversation next week?"

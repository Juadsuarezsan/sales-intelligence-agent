import pytest
from src.agents.email_writer import EmailWriter, _score_personalization
from src.api.schemas import CompanyProfile, KeyPerson, OutreachEmail, Signal


def _profile() -> CompanyProfile:
    return CompanyProfile(
        name="Anthropic", industry="AI safety", location="San Francisco",
        key_people=[KeyPerson(name="Dario Amodei", role="CEO")],
        recent_signals=[
            Signal(kind="funding", description="$2B Series C round closed", source_url="https://x.com/a"),
        ],
    )


def test_high_personalization_when_email_mentions_facts():
    p = _profile()
    email = OutreachEmail(
        subject="AI safety partnership",
        greeting="Hi team,",
        hook="Saw Anthropic's recent $2B Series C funding round.",
        value_prop="We help AI safety teams ship workflow tooling 30% faster.",
        cta="Open to a 20-min chat?",
    )
    score = _score_personalization(email, p)
    assert score >= 0.4


def test_low_personalization_for_generic_email():
    p = _profile()
    email = OutreachEmail(
        subject="generic outreach",
        greeting="Hi team,",
        hook="Quick intro.",
        value_prop="We build software for businesses.",
        cta="Coffee?",
    )
    assert _score_personalization(email, p) < 0.3


@pytest.mark.asyncio
async def test_template_email_falls_back_without_api_key():
    w = EmailWriter(model="x", api_key=None)
    e = await w.write(_profile())
    assert e.subject
    assert e.cta

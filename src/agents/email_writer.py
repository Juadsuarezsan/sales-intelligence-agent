"""Email Writer node: produces the :class:`OutreachEmail` from profile + hooks."""

from __future__ import annotations

from loguru import logger

from src.api.schemas import CompanyProfile, OutreachEmail, PersonalizationHooks
from src.llm import LLMClient, parse_json_object, recoverable_llm_errors

DEFAULT_OFFERING = "an AI workflow platform"

WRITER_SYSTEM = """You write short personalized B2B cold outreach emails.

Output JSON ONLY with this schema:
{
  "subject": "5-7 words max",
  "greeting": "Hi {first_name}, | Hello team,",
  "hook": "1 sentence that references the recent_signal from the hooks",
  "value_prop": "1-2 sentences on what we offer and why it fits THIS company's pain_point",
  "cta": "Specific 1-sentence call-to-action (a meeting, a demo, a reply)"
}

Rules:
- Reference at least one concrete fact from the profile in the hook (funding round,
  product, recent news). Do NOT invent facts that are not in the profile.
- Keep under 90 words total. Match the company's industry tone.
"""


class EmailWriter:
    """Writes the outreach email with the LLM or a template.

    Args:
        llm: Completion client; ``None`` selects the template.
        offering: What "we" sell, injected into the value proposition.
    """

    def __init__(self, llm: LLMClient | None, *, offering: str = DEFAULT_OFFERING) -> None:
        self.llm = llm
        self.offering = offering

    async def write(
        self, profile: CompanyProfile, hooks: PersonalizationHooks | None = None
    ) -> OutreachEmail:
        """Produce the email.

        Args:
            profile: Structured company profile.
            hooks: Personalization hooks; derived trivially when ``None``.

        Returns:
            A validated email with a keyword-based ``personalization_score``.
        """
        if hooks is None:
            hooks = PersonalizationHooks(pain_point="manual work slows the team down")
        if self.llm is None:
            return self.template(profile, hooks, offering=self.offering)
        user = (
            f"<profile>\n{profile.model_dump_json(indent=2)}\n</profile>\n"
            f"<hooks>\n{hooks.model_dump_json(indent=2)}\n</hooks>\n"
            f"<our_offering>{self.offering}</our_offering>"
        )
        try:
            result = await self.llm.complete(
                system=WRITER_SYSTEM, user=user, max_tokens=500, temperature=0.7
            )
            raw = parse_json_object(result.text)
            email = OutreachEmail.model_validate(raw)
        except recoverable_llm_errors() as exc:
            logger.warning(f"email writer LLM failed, using template: {exc!r}")
            return self.template(profile, hooks, offering=self.offering)
        email.personalization_score = score_personalization(email, profile)
        return email

    @staticmethod
    def template(
        profile: CompanyProfile,
        hooks: PersonalizationHooks,
        *,
        offering: str = DEFAULT_OFFERING,
    ) -> OutreachEmail:
        """Deterministic template used offline, as fallback and as the template baseline.

        Args:
            profile: Structured company profile.
            hooks: Personalization hooks.
            offering: What "we" sell.

        Returns:
            The templated email.
        """
        if hooks.recent_signal:
            hook = f"I noticed {profile.name}'s recent {hooks.recent_signal}."
        else:
            hook = f"I came across {profile.name} while researching the space."
        first_name = profile.key_people[0].name.split()[0] if profile.key_people else None
        greeting = f"Hi {first_name}," if first_name else "Hi team,"
        email = OutreachEmail(
            subject=f"{offering[:40]} for {profile.name}"[:200],
            greeting=greeting,
            hook=hook[:1000],
            value_prop=(
                f"We build {offering} for teams where {hooks.pain_point.rstrip('.')}; "
                f"customers cut that manual work by a measurable margin within a quarter."
            )[:1000],
            cta="Worth a 20-minute conversation next week?",
        )
        email.personalization_score = score_personalization(email, profile)
        return email


def score_personalization(email: OutreachEmail, profile: CompanyProfile) -> float:
    """Keyword-overlap personalization score in ``[0, 1]``.

    Counts how many profile facts (name, industry, signals, people) appear in
    the hook and value proposition. It is a cheap proxy; the LLM-as-judge in
    :mod:`src.agents.quality_scorer` is the real metric.

    Args:
        email: Email to score.
        profile: Profile the email was written from.

    Returns:
        Score between 0 and 1.
    """
    text = " ".join([email.hook, email.value_prop]).lower()
    score = 0.0
    if profile.name and profile.name.lower() in text:
        score += 0.2
    if profile.industry and profile.industry.lower() in text:
        score += 0.2
    for s in profile.recent_signals[:3]:
        keywords = [w.lower() for w in s.description.split()[:3] if len(w) > 2]
        if keywords and any(k in text for k in keywords):
            score += 0.2
    for p in profile.key_people[:2]:
        if p.name.lower() in text:
            score += 0.1
    return min(1.0, round(score, 3))

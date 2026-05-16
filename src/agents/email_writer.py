"""Email writer + LLM-as-judge personalization scorer."""
from __future__ import annotations

import json

from src.api.schemas import CompanyProfile, OutreachEmail


WRITER_SYSTEM = """You write short personalized B2B cold outreach emails.

Output JSON ONLY with this schema:
{
  "subject": "5-7 words max",
  "greeting": "Hi {first_name}, | Hello team,",
  "hook": "1 sentence that references a specific recent signal in the profile",
  "value_prop": "1-2 sentences on what we offer and why it fits THIS company",
  "cta": "Specific 1-sentence call-to-action (a meeting, a demo, a reply)"
}

Rules:
- Reference at least one concrete fact from the profile in the hook (funding round,
  product, recent news). Do NOT invent facts that aren't in the profile.
- Keep under 90 words total.
- Match the company's industry tone.
"""


class EmailWriter:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    async def write(self, profile: CompanyProfile, our_offering: str = "an AI workflow platform") -> OutreachEmail:
        if not self.api_key:
            return self._template(profile, our_offering)
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        chat = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0.7, max_tokens=400, timeout=15.0)
        prof_text = profile.model_dump_json(indent=2)
        user = f"<profile>\n{prof_text}\n</profile>\n<our_offering>{our_offering}</our_offering>"
        resp = await chat.ainvoke([SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=user)])
        body = resp.content if isinstance(resp.content, str) else str(resp.content)
        body = body.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:].lstrip()
        try:
            raw = json.loads(body)
            email = OutreachEmail(**raw)
            email.personalization_score = _score_personalization(email, profile)
            return email
        except Exception:
            return self._template(profile, our_offering)

    @staticmethod
    def _template(profile: CompanyProfile, our_offering: str) -> OutreachEmail:
        signal_text = ""
        if profile.recent_signals:
            s = profile.recent_signals[0]
            signal_text = f"saw your recent {s.kind.replace('_',' ')} — {s.description}"
        else:
            signal_text = f"came across {profile.name} while researching the space"
        return OutreachEmail(
            subject=f"{our_offering[:40]} for {profile.name}",
            greeting="Hi team,",
            hook=f"I {signal_text}.",
            value_prop=f"We build {our_offering}. Companies like yours typically reduce manual ops by 30-50%.",
            cta="Worth a 20-minute conversation next week?",
            personalization_score=0.4,
        )


def _score_personalization(email: OutreachEmail, profile: CompanyProfile) -> float:
    """Heuristic personalization score 0..1: how many profile facts appear in the email."""
    text = " ".join([email.hook, email.value_prop]).lower()
    score = 0.0
    facts = []
    if profile.name and profile.name.lower() in text:
        score += 0.2
    if profile.industry and profile.industry.lower() in text:
        score += 0.2
        facts.append("industry")
    for s in profile.recent_signals[:3]:
        keyword = s.description.split()[0:3]
        if any(k.lower() in text for k in keyword):
            score += 0.2
            facts.append(f"signal:{s.kind}")
    for p in profile.key_people[:2]:
        if p.name.lower() in text:
            score += 0.1
    return min(1.0, score)

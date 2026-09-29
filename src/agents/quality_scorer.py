"""Quality Scorer node: LLM-as-judge with an explicit, numbered rubric.

The rubric below is the single source of truth for what "personalization",
"accuracy" and "CTA clarity" mean in this project. It is rendered verbatim
into the judge prompt and documented in ``eval/RESULTS.md``. When no LLM is
configured, a deterministic keyword approximation of the same three criteria
is used and the result is labelled ``judge="heuristic"``.
"""

from __future__ import annotations

import re

from loguru import logger

from src.api.schemas import CompanyProfile, OutreachEmail, QualityScores
from src.llm import LLMClient, parse_json_object, recoverable_llm_errors

RUBRIC = """Score the outreach email on three criteria, each an integer from 1 to 5.

1. PERSONALIZATION - does the email prove it was written for THIS company?
   1 = fully generic; could be sent to anyone ("I came across your company").
   2 = mentions the company name only.
   3 = references the industry or a vague fact ("you are growing fast").
   4 = references one specific, verifiable fact from the profile (a funding
       round, a named product, a named executive, an open role).
   5 = weaves two or more specific facts into a coherent reason to reach out now.
   Example 5: "Congrats on the $40M Series B - with six open platform roles,
   onboarding those engineers onto one workflow will matter."

2. ACCURACY - is every factual claim supported by the profile?
   1 = contains a fabricated or contradicted fact (wrong round, wrong CEO).
   2 = several unsupported claims presented as facts.
   3 = one unsupported claim or an exaggeration of a real fact.
   4 = all claims supported; minor imprecision in wording.
   5 = every claim traceable to a profile field or signal, no exaggeration.
   Example 1: profile says Seed round, email congratulates on Series C.

3. CTA_CLARITY - is the call to action specific and low-friction?
   1 = no call to action.
   2 = vague ("let me know your thoughts").
   3 = asks for something but without format or time ("can we talk?").
   4 = one concrete ask with a format or duration ("20-minute call next week").
   5 = concrete ask, duration, timeframe and an easy opt-out or alternative.
   Example 5: "Open to a 20-minute call Tuesday or Thursday? Happy to send a
   two-minute video instead if easier."
"""

JUDGE_SYSTEM = (
    "You are a strict evaluator of B2B cold outreach emails. Apply the rubric "
    "literally; do not reward fluency.\n\n" + RUBRIC + "\nOutput JSON only:\n"
    '{"personalization": <1-5>, "accuracy": <1-5>, "cta_clarity": <1-5>, '
    '"rationale": "two sentences citing the rubric levels"}'
)

_MONEY = r"\$\s?\d+(?:\.\d+)?\s?(?:billion|million|bn|m|b)\b"
_ROUND = r"series\s[a-h]\b"
#: One factual claim: a money amount optionally followed by its round, or a bare round.
_CLAIM = re.compile(rf"{_MONEY}(?:\s+(?:in\s+(?:a\s+|its\s+)?)?{_ROUND})?|{_ROUND}", re.I)
_CLAIM_PARTS = re.compile(rf"{_MONEY}|{_ROUND}", re.I)
_CTA_TIME = re.compile(
    r"\b(\d+[- ]?min(?:ute)?s?|next week|this week|tuesday|thursday|monday|"
    r"wednesday|friday|tomorrow)\b",
    re.I,
)
_CTA_FORMAT = re.compile(r"\b(call|chat|conversation|demo|meeting|reply|video)\b", re.I)


class QualityScorer:
    """Scores emails with the rubric via LLM, or a deterministic approximation.

    Args:
        llm: Judge completion client; ``None`` selects the heuristic.
    """

    def __init__(self, llm: LLMClient | None) -> None:
        self.llm = llm

    async def score(self, email: OutreachEmail, profile: CompanyProfile) -> QualityScores:
        """Score one email.

        Args:
            email: Email to judge.
            profile: Profile it was written from (ground for accuracy).

        Returns:
            Scores with ``judge="llm"`` or ``judge="heuristic"``.
        """
        if self.llm is None:
            return self.heuristic(email, profile)
        user = (
            f"<profile>\n{profile.model_dump_json(indent=2)}\n</profile>\n"
            f"<email>\nSubject: {email.subject}\n\n{email.body()}\n</email>"
        )
        try:
            result = await self.llm.complete(system=JUDGE_SYSTEM, user=user, max_tokens=400)
            raw = parse_json_object(result.text)
            p, a, c = (int(raw["personalization"]), int(raw["accuracy"]), int(raw["cta_clarity"]))
            return QualityScores(
                personalization=p,
                accuracy=a,
                cta_clarity=c,
                overall=round((p + a + c) / 3, 2),
                rationale=str(raw.get("rationale", "")),
                judge="llm",
            )
        except (KeyError, TypeError) as exc:
            logger.warning(f"quality scorer returned malformed scores, using heuristic: {exc!r}")
            return self.heuristic(email, profile)
        except recoverable_llm_errors() as exc:
            logger.warning(f"quality scorer LLM failed, using heuristic: {exc!r}")
            return self.heuristic(email, profile)

    @staticmethod
    def heuristic(email: OutreachEmail, profile: CompanyProfile) -> QualityScores:
        """Deterministic approximation of the rubric.

        Personalization counts specific profile facts present in the email;
        accuracy checks that money amounts / rounds mentioned exist in the
        profile; CTA clarity checks for a format and a time reference.

        Args:
            email: Email to judge.
            profile: Profile it was written from.

        Returns:
            Scores labelled ``judge="heuristic"``.
        """
        text = f"{email.hook} {email.value_prop}".lower()
        facts = 0
        if profile.name.lower() in text:
            facts += 1
        if profile.industry and profile.industry.lower() in text:
            facts += 1
        for s in profile.recent_signals[:5]:
            keywords = [w.lower() for w in s.description.split()[:4] if len(w) > 3]
            if keywords and any(k in text for k in keywords):
                facts += 1
        for p in profile.key_people[:3]:
            if p.name.lower() in text:
                facts += 1
        personalization = 1 if facts == 0 else 2 if facts == 1 else 4 if facts == 2 else 5

        profile_text = profile.model_dump_json().lower().replace(" ", "")
        claims = _CLAIM.findall(email.body())
        unsupported = sum(
            1
            for claim in claims
            if any(
                part.lower().replace(" ", "") not in profile_text
                for part in _CLAIM_PARTS.findall(claim)
            )
        )
        accuracy = 5 if not claims else 4 if unsupported == 0 else 3 if unsupported == 1 else 1

        cta = email.cta
        has_format, has_time = bool(_CTA_FORMAT.search(cta)), bool(_CTA_TIME.search(cta))
        if not cta.strip():
            cta_clarity = 1
        elif has_format and has_time:
            cta_clarity = 4
        elif has_format or has_time or "?" in cta:
            cta_clarity = 3
        else:
            cta_clarity = 2

        return QualityScores(
            personalization=personalization,
            accuracy=accuracy,
            cta_clarity=cta_clarity,
            overall=round((personalization + accuracy + cta_clarity) / 3, 2),
            rationale=(
                f"heuristic: facts={facts} claims={len(claims)} unsupported={unsupported} "
                f"cta_format={has_format} cta_time={has_time}"
            ),
            judge="heuristic",
        )

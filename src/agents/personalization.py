"""Personalization Extractor node: picks the hook (pain point + recent signal)."""

from __future__ import annotations

from loguru import logger

from src.api.schemas import CompanyProfile, PersonalizationHooks
from src.llm import LLMClient, parse_json_object, recoverable_llm_errors

SYSTEM = """You are a senior SDR preparing a cold email. From the structured company
profile, extract the single best personalization hook.

Output JSON only:
{
  "pain_point": "one sentence: the operational problem this company most likely has now",
  "recent_signal": "one sentence quoting the most recent, specific signal, or null",
  "angle": "one sentence connecting the signal and pain point to an AI workflow platform",
  "evidence_refs": ["source url(s) that support the signal"]
}

Rules: only use facts present in the profile. Prefer funding > product launch >
hiring > leadership change > news when several signals exist.
"""

_SIGNAL_PRIORITY = ("funding", "product_launch", "hiring", "leadership_change", "news", "other")

_INDUSTRY_PAIN: dict[str, str] = {
    "fintech": "reconciling compliance reviews and onboarding checks by hand slows growth",
    "healthcare": "clinical and administrative teams lose hours to manual documentation",
    "b2b": "go-to-market teams spend more time on manual research than on selling",
    "consumer": "support and ops volume grows faster than headcount",
    "industrials": "field and back-office workflows still run on spreadsheets and email",
    "education": "student-facing staff are buried in repetitive administrative work",
    "real estate and construction": "project coordination relies on manual status chasing",
    "government": "case processing depends on slow, paper-heavy workflows",
}


class PersonalizationExtractor:
    """Derives :class:`PersonalizationHooks` from a profile.

    Args:
        llm: Completion client; ``None`` selects the offline heuristic.
    """

    def __init__(self, llm: LLMClient | None) -> None:
        self.llm = llm

    async def extract(self, profile: CompanyProfile) -> PersonalizationHooks:
        """Extract the hooks.

        Args:
            profile: Structured company profile.

        Returns:
            Validated hooks; heuristic when the LLM output is unusable.
        """
        if self.llm is None:
            return self.heuristic(profile)
        user = f"<profile>\n{profile.model_dump_json(indent=2)}\n</profile>"
        try:
            result = await self.llm.complete(system=SYSTEM, user=user, max_tokens=400)
            raw = parse_json_object(result.text)
            return PersonalizationHooks.model_validate(raw)
        except recoverable_llm_errors() as exc:
            logger.warning(f"personalization LLM failed, using heuristic: {exc!r}")
            return self.heuristic(profile)

    @staticmethod
    def heuristic(profile: CompanyProfile) -> PersonalizationHooks:
        """Rule-based hook selection used offline and as fallback.

        Args:
            profile: Structured company profile.

        Returns:
            Hooks built from the highest-priority signal and an industry pain map.
        """
        signal = None
        for kind in _SIGNAL_PRIORITY:
            signal = next((s for s in profile.recent_signals if s.kind == kind), None)
            if signal is not None:
                break

        industry = (profile.industry or "").lower()
        pain = next((p for k, p in _INDUSTRY_PAIN.items() if k in industry), None)
        if pain is None:
            if profile.pain_points:
                pain = profile.pain_points[0]
            elif signal is not None and signal.kind == "hiring":
                pain = "a growing team needs repeatable processes instead of heroics"
            elif signal is not None and signal.kind == "funding":
                pain = "post-raise growth targets outpace what manual operations can deliver"
            else:
                pain = "manual research and follow-up eat into the team's selling time"

        recent = None
        refs: list[str] = []
        if signal is not None:
            recent = f"{signal.kind.replace('_', ' ')}: {signal.description}"
            if signal.source_url:
                refs.append(signal.source_url)
        trigger = f"just showed {signal.kind.replace('_', ' ')}" if signal else "is scaling"
        angle = (
            f"Because {profile.name} {trigger}, automating the workflows behind that growth "
            f"frees the team from the manual work behind '{pain.rstrip('.')}'."
        )
        return PersonalizationHooks(
            pain_point=pain, recent_signal=recent, angle=angle[:500], evidence_refs=refs
        )

"""Profile Builder node: turns evidence into a validated :class:`CompanyProfile`."""

from __future__ import annotations

import re
from collections.abc import Sequence

from loguru import logger

from src.api.schemas import CompanyProfile, Evidence, KeyPerson, Signal
from src.llm import LLMClient, parse_json_object, recoverable_llm_errors

SYSTEM = """You build a structured B2B company profile from web evidence.

Output ONLY valid JSON matching this schema:
{
  "name": "...",
  "one_liner": "10-15 word description of what the company does",
  "industry": "one of: B2B, Consumer, Healthcare, Fintech, Industrials, Education,
               Real Estate and Construction, Government, or a short free text",
  "website": "https://..." | null,
  "employees_est": <int> | null,
  "funding_total_usd": <number> | null,
  "last_funding_round": "Series B" | "Seed" | null,
  "location": "City, Region, Country" | null,
  "key_people": [{"name": "...", "role": "...", "linkedin_url": null}],
  "recent_signals": [{"kind": "funding|hiring|product_launch|news|leadership_change|other",
                      "description": "...", "source_url": "...", "occurred_at": null}],
  "pain_points": ["..."],
  "sources": ["url1", "url2"]
}

Never invent data not present in the evidence. If a field is unknown, set null
or an empty list. Copy URLs verbatim from the evidence into sources.
"""

_MONEY = re.compile(r"\$\s?(\d+(?:\.\d+)?)\s?(billion|million|bn|m|b)\b", re.IGNORECASE)
_ROUND = re.compile(r"\b(pre-seed|seed|series [a-h])\b", re.IGNORECASE)
_PERSON = re.compile(
    r"\b(CEO|CTO|COO|CFO|founder|co-founder)\s+(?:is|:)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})"
)
_LOCATION = re.compile(
    r"\b(?:headquartered|based|located)\s+in\s+([A-Z][A-Za-z]+(?:,? [A-Z][A-Za-z]+){0,3})"
)
_ROLES_OPEN = re.compile(r"open roles detected:\s*(\d+)", re.IGNORECASE)
_TEAM = re.compile(r"\b(\d{1,4})\s+(?:employees|people|team members)\b", re.IGNORECASE)


def _money_to_usd(amount: str, unit: str) -> float:
    value = float(amount)
    unit = unit.lower()
    if unit in ("billion", "bn", "b"):
        return value * 1_000_000_000
    return value * 1_000_000


class ProfileBuilder:
    """Builds a :class:`CompanyProfile` with the LLM or regex heuristics.

    Args:
        llm: Completion client; ``None`` selects the offline heuristic.
    """

    def __init__(self, llm: LLMClient | None) -> None:
        self.llm = llm

    async def build(
        self,
        company_name: str,
        evidence: Sequence[Evidence],
        *,
        website: str | None = None,
    ) -> CompanyProfile:
        """Produce the profile.

        Args:
            company_name: Target company.
            evidence: All evidence collected by the executor.
            website: Known website, if the caller supplied one.

        Returns:
            A validated profile. Falls back to the heuristic when the LLM
            output is unusable.
        """
        if self.llm is None:
            return self.heuristic(company_name, list(evidence), website=website)
        lines = "\n".join(
            f"- source: {e.get('source', '?')}\n  title: {e.get('title', '')}\n"
            f"  url: {e.get('url', '')}\n  content: {e.get('content', '')[:400]}"
            for e in evidence[:15]
        )
        user = f"<company>{company_name}</company>\n"
        if website:
            user += f"<website>{website}</website>\n"
        user += f"<evidence>\n{lines}\n</evidence>"
        try:
            result = await self.llm.complete(system=SYSTEM, user=user, max_tokens=1500)
            raw = parse_json_object(result.text)
            raw.setdefault("name", company_name)
            if website and not raw.get("website"):
                raw["website"] = website
            return CompanyProfile.model_validate(raw)
        except recoverable_llm_errors() as exc:
            logger.warning(f"profile builder LLM failed, using heuristic: {exc!r}")
            return self.heuristic(company_name, list(evidence), website=website)

    @staticmethod
    def heuristic(
        company_name: str, evidence: list[Evidence], *, website: str | None = None
    ) -> CompanyProfile:
        """Regex-based extraction used offline and as fallback.

        It pulls funding amounts and rounds, C-level names, a headquarters
        location, an open-roles count and a one-liner from the homepage meta
        description when the website fetcher supplied one.

        Args:
            company_name: Target company.
            evidence: Evidence items.
            website: Known website, if any.

        Returns:
            A profile whose ``one_liner`` is marked ``[heuristic]`` when no
            description could be extracted.
        """
        sources = list(dict.fromkeys(e.get("url", "") for e in evidence if e.get("url")))
        text = " ".join(f"{e.get('title', '')}. {e.get('content', '')}" for e in evidence)

        funding_total: float | None = None
        money = _MONEY.search(text)
        if money:
            funding_total = _money_to_usd(money.group(1), money.group(2))
        round_match = _ROUND.search(text)
        last_round = round_match.group(1).title() if round_match else None

        people: list[KeyPerson] = []
        for role, name in _PERSON.findall(text):
            if all(p.name != name for p in people):
                people.append(KeyPerson(name=name, role=role.upper() if len(role) <= 3 else role))

        loc_match = _LOCATION.search(text)
        location = loc_match.group(1).strip(" .,") if loc_match else None

        team = _TEAM.search(text)
        employees = int(team.group(1)) if team else None

        signals: list[Signal] = []
        for e in evidence:
            content = e.get("content", "")
            low = content.lower()
            url = e.get("url", "")
            if len(signals) >= 5:
                break
            if _MONEY.search(content) or _ROUND.search(content):
                signals.append(
                    Signal(kind="funding", description=e.get("title", "")[:160], source_url=url)
                )
            elif (m := _ROLES_OPEN.search(content)) and int(m.group(1)) > 0:
                signals.append(
                    Signal(
                        kind="hiring",
                        description=f"{m.group(1)} open roles on careers page",
                        source_url=url,
                    )
                )
            elif "launch" in low or "announce" in low:
                signals.append(
                    Signal(
                        kind="product_launch",
                        description=e.get("title", "")[:160],
                        source_url=url,
                    )
                )
            elif "appoint" in low or "promot" in low or "joins as" in low:
                signals.append(
                    Signal(
                        kind="leadership_change",
                        description=e.get("title", "")[:160],
                        source_url=url,
                    )
                )
            elif e.get("source") == "hn":
                signals.append(
                    Signal(kind="news", description=e.get("title", "")[:160], source_url=url)
                )

        one_liner = ""
        for e in evidence:
            if e.get("source") == "website" and e.get("content"):
                first = e["content"].split(" Headings:")[0].strip()
                if len(first) >= 20:
                    one_liner = first[:200]
                    break
        if not one_liner:
            one_liner = "[heuristic] profile built without LLM; connect ANTHROPIC_API_KEY"

        return CompanyProfile(
            name=company_name,
            one_liner=one_liner,
            website=website,
            employees_est=employees,
            funding_total_usd=funding_total,
            last_funding_round=last_round,
            location=location,
            key_people=people[:4],
            recent_signals=signals,
            sources=sources,
        )

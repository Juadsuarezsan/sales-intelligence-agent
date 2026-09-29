"""Ground-truth loader for the 200-company YC eval set.

The file ``data/eval/yc_ground_truth_200.json`` is produced by
``scripts/download_data.py`` from the public ``yc-oss/api`` dataset. Every
field comes from Y Combinator's own company directory, which is the
"validated against a reliable source" ground truth the spec asks for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = REPO_ROOT / "data" / "eval" / "yc_ground_truth_200.json"


@dataclass(frozen=True)
class GroundTruthCompany:
    """One YC company with the metadata fields used for scoring.

    Attributes:
        name: Company name.
        website: Canonical website URL.
        industry: Top-level YC industry (``B2B``, ``Fintech``...).
        location: ``all_locations`` string from YC (``City, Region, Country``).
        team_size: Headcount reported to YC.
        one_liner: YC one-line description.
        batch: YC batch label.
        status: ``Active``, ``Acquired``...
        long_description: YC long description.
        tags: YC tags.
        yc_url: Company page on ycombinator.com.
    """

    name: str
    website: str
    industry: str
    location: str
    team_size: int | None
    one_liner: str
    batch: str = ""
    status: str = ""
    long_description: str = ""
    tags: list[str] = field(default_factory=list)
    yc_url: str = ""


def load_ground_truth(path: Path = DEFAULT_GROUND_TRUTH) -> list[GroundTruthCompany]:
    """Load the eval set.

    Args:
        path: JSON file written by ``scripts/download_data.py``.

    Returns:
        Companies in file order (already deterministic).

    Raises:
        FileNotFoundError: If the eval set has not been generated yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python scripts/download_data.py` to build the eval set"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    companies = raw["companies"] if isinstance(raw, dict) else raw
    return [
        GroundTruthCompany(
            name=c["name"],
            website=c.get("website", ""),
            industry=c.get("industry", ""),
            location=c.get("location", ""),
            team_size=c.get("team_size"),
            one_liner=c.get("one_liner", ""),
            batch=c.get("batch", ""),
            status=c.get("status", ""),
            long_description=c.get("long_description", ""),
            tags=list(c.get("tags") or []),
            yc_url=c.get("yc_url", ""),
        )
        for c in companies
    ]

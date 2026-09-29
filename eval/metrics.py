"""Field-level information accuracy and distribution helpers.

Accuracy is measured as precision/recall over the four profile fields that
have a YC counterpart and are not supplied as input. A predicted field counts
as *correct* only when it matches the ground truth under the rule documented
in :func:`field_matches`;
``None``/empty predictions are neither correct nor wrong (they lower recall,
not precision).
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlparse

from eval.ground_truth import GroundTruthCompany
from src.api.schemas import CompanyProfile

#: Scored fields. ``website`` is deliberately excluded: the eval hands it to
#: every system as input (so the fetcher can work), so it measures nothing.
FIELDS: tuple[str, ...] = ("industry", "location", "employees_est", "one_liner")

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "for", "of", "to", "in", "on", "with", "your", "that", "is"}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2}


def _host(url: str) -> str:
    candidate = url if "://" in url else "https://" + url
    host = urlparse(candidate).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def field_matches(name: str, predicted: object, truth: GroundTruthCompany) -> bool:
    """Decide whether a predicted field value matches the ground truth.

    Rules:
        * ``industry``: case-insensitive containment either way.
        * ``location``: the predicted city (first comma-separated token) appears
          in YC's ``all_locations`` or vice versa.
        * ``employees_est``: within ±25 % of YC ``team_size``.
        * ``website``: same registrable host (``www.`` stripped).
        * ``one_liner``: Jaccard similarity of content tokens >= 0.3 against the
          YC one-liner.

    Args:
        name: Field name from :data:`FIELDS`.
        predicted: Value on the profile.
        truth: Ground-truth record.

    Returns:
        ``True`` when the prediction is correct.
    """
    if predicted is None or predicted == "":
        return False
    if name == "industry":
        p, t = str(predicted).lower().strip(), truth.industry.lower().strip()
        return bool(p and t) and (p in t or t in p)
    if name == "location":
        p_city = str(predicted).split(",")[0].strip().lower()
        t = truth.location.lower()
        return bool(p_city) and (p_city in t or t.split(",")[0].strip() in str(predicted).lower())
    if name == "employees_est":
        if truth.team_size is None or not isinstance(predicted, int):
            return False
        return abs(predicted - truth.team_size) <= max(1, round(0.25 * truth.team_size))
    if name == "website":
        return bool(truth.website) and _host(str(predicted)) == _host(truth.website)
    if name == "one_liner":
        a, b = _tokens(str(predicted)), _tokens(truth.one_liner)
        if not a or not b:
            return False
        return len(a & b) / len(a | b) >= 0.3
    raise ValueError(f"unknown field {name!r}")


def _truth_has(name: str, truth: GroundTruthCompany) -> bool:
    value = {
        "industry": truth.industry,
        "location": truth.location,
        "employees_est": truth.team_size,
        "website": truth.website,
        "one_liner": truth.one_liner,
    }[name]
    return value not in (None, "")


@dataclass
class FieldScores:
    """Per-company accuracy counts.

    Attributes:
        correct: Predicted fields that match.
        predicted: Non-empty predicted fields.
        expected: Fields with a ground-truth value.
        per_field: ``field -> "tp" | "fp" | "fn" | "tn"``.
    """

    correct: int = 0
    predicted: int = 0
    expected: int = 0
    per_field: dict[str, str] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        """Correct / predicted (1.0 when nothing was predicted)."""
        return self.correct / self.predicted if self.predicted else 1.0

    @property
    def recall(self) -> float:
        """Correct / expected (1.0 when nothing was expected)."""
        return self.correct / self.expected if self.expected else 1.0


def score_profile(profile: CompanyProfile, truth: GroundTruthCompany) -> FieldScores:
    """Compare a profile with its ground truth field by field.

    Args:
        profile: Predicted profile.
        truth: Ground-truth record.

    Returns:
        Counts and per-field outcomes.
    """
    scores = FieldScores()
    for name in FIELDS:
        predicted = getattr(profile, name)
        has_pred = predicted not in (None, "") and not str(predicted).startswith("[heuristic]")
        has_truth = _truth_has(name, truth)
        if has_truth:
            scores.expected += 1
        if has_pred:
            scores.predicted += 1
        if has_pred and has_truth and field_matches(name, predicted, truth):
            scores.correct += 1
            scores.per_field[name] = "tp"
        elif has_pred:
            scores.per_field[name] = "fp"
        elif has_truth:
            scores.per_field[name] = "fn"
        else:
            scores.per_field[name] = "tn"
    return scores


def aggregate_accuracy(items: Sequence[FieldScores]) -> dict[str, float]:
    """Micro-average precision/recall/F1 over companies.

    Args:
        items: Per-company scores.

    Returns:
        ``{"precision", "recall", "f1"}`` rounded to 4 decimals.
    """
    correct = sum(i.correct for i in items)
    predicted = sum(i.predicted for i in items)
    expected = sum(i.expected for i in items)
    precision = correct / predicted if predicted else 0.0
    recall = correct / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def distribution(values: Sequence[float]) -> dict[str, float]:
    """Summary statistics of a numeric sample.

    Args:
        values: Sample (may be empty).

    Returns:
        ``mean``, ``p50``, ``p95``, ``min``, ``max`` (all ``0.0`` when empty).
    """
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "mean": round(statistics.fmean(ordered), 4),
        "p50": round(statistics.median(ordered), 4),
        "p95": round(ordered[p95_index], 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }

import json
from pathlib import Path

import pytest

from eval import run as eval_run
from eval.baselines import single_search_email, template_only
from eval.ground_truth import GroundTruthCompany, load_ground_truth
from eval.metrics import (
    FIELDS,
    aggregate_accuracy,
    distribution,
    field_matches,
    score_profile,
)
from eval.report import PENDING, render_results
from src.agents.orchestrator import build_components
from src.api.schemas import CompanyProfile
from src.config import Settings

TRUTH = GroundTruthCompany(
    name="Rescale",
    website="https://rescale.com",
    industry="B2B",
    location="San Francisco, CA, USA",
    team_size=250,
    one_liner="High Performance Computing Built for the Cloud",
    batch="Winter 2012",
)


def test_field_matching_rules() -> None:
    assert field_matches("industry", "b2b software", TRUTH)
    assert not field_matches("industry", "Healthcare", TRUTH)
    assert field_matches("location", "San Francisco, California", TRUTH)
    assert not field_matches("location", "Berlin", TRUTH)
    assert field_matches("employees_est", 300, TRUTH)  # within 25 %
    assert not field_matches("employees_est", 400, TRUTH)
    assert field_matches("one_liner", "cloud high performance computing platform", TRUTH)
    assert not field_matches("one_liner", "a bakery in Ohio", TRUTH)
    assert not field_matches("industry", None, TRUTH)
    with pytest.raises(ValueError):
        field_matches("nope", "x", TRUTH)


def test_score_profile_precision_recall() -> None:
    profile = CompanyProfile(
        name="Rescale",
        industry="B2B",
        location="Berlin",
        one_liner="[heuristic] nothing",
    )
    fs = score_profile(profile, TRUTH)
    assert fs.per_field == {
        "industry": "tp",
        "location": "fp",
        "employees_est": "fn",
        "one_liner": "fn",
    }
    assert fs.precision == 0.5 and fs.recall == 0.25
    agg = aggregate_accuracy([fs, fs])
    assert agg == {"precision": 0.5, "recall": 0.25, "f1": pytest.approx(0.3333, abs=1e-4)}
    assert aggregate_accuracy([]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_distribution_stats() -> None:
    assert distribution([]) == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    d = distribution([1, 2, 3, 4, 100])
    assert d["p50"] == 3 and d["max"] == 100 and d["mean"] == 22


def test_ground_truth_file_has_200_complete_records() -> None:
    companies = load_ground_truth()
    assert len(companies) == 200
    assert all(c.website and c.industry and c.location and c.one_liner for c in companies)
    assert all(isinstance(c.team_size, int) and c.team_size > 0 for c in companies)
    assert len({c.name for c in companies}) == 200


def test_ground_truth_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ground_truth(tmp_path / "nope.json")


async def test_baselines_run_offline() -> None:
    settings = Settings()
    c = build_components(settings)
    t = await template_only(TRUTH, c, settings)
    assert t.tool_calls == 0 and t.cost_usd == 0.0 and t.email.subject
    s = await single_search_email(TRUTH, c, settings)
    assert s.tool_calls == 1 and s.iterations == 1 and s.profile.website == TRUTH.website


async def test_run_eval_and_render(tmp_path: Path) -> None:
    settings = Settings()
    companies = load_ground_truth()[:3]
    report = await eval_run.run_eval(
        companies=companies, components=build_components(settings), settings=settings
    )
    names = [s["system"] for s in report["systems"]]
    assert names == [
        "template_only",
        "single_search_email",
        "agent_reflection_0",
        "agent_reflection_1",
        "agent_reflection_2",
        "agent_reflection_3",
    ]
    assert report["mode"].startswith("fallback")
    assert report["fields"] == list(FIELDS)
    agent = report["systems"][-1]
    assert agent["n"] == 3 and len(agent["rows"]) == 3
    md = render_results(report, run_file="x.json")
    assert "| Template only |" in md and "Agent loop con reflection (este)" in md
    assert PENDING in md and "1. PERSONALIZATION" in md
    assert "no representan la calidad del sistema" in md


def test_eval_cli_writes_run_and_results(tmp_path: Path) -> None:
    out = eval_run.main(
        [
            "--limit",
            "2",
            "--name",
            "unit",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--results",
            str(tmp_path / "RESULTS.md"),
        ]
    )
    assert out.exists() and out.name.endswith("-unit.json")
    data = json.loads(out.read_text())
    assert data["n_companies"] == 2
    assert (tmp_path / "RESULTS.md").read_text().startswith("# Resultados de evaluación")

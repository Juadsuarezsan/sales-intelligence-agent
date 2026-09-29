import json
from pathlib import Path

import httpx
import respx

from scripts import batch_run, download_data

RAW = [
    {
        "id": 3,
        "name": "Gamma",
        "slug": "gamma",
        "website": "https://gamma.io",
        "industry": "B2B",
        "subindustry": "B2B -> Infra",
        "all_locations": "Austin, TX, USA",
        "team_size": 12,
        "one_liner": "Infra for all",
        "long_description": "…",
        "batch": "Winter 2024",
        "status": "Active",
        "stage": "Early",
        "tags": ["Infra"],
        "isHiring": True,
        "launched_at": 1,
        "url": "https://www.ycombinator.com/companies/gamma",
    },
    {"id": 1, "name": "Alpha", "status": "Inactive", "website": "https://a.io"},
    {
        "id": 2,
        "name": "Beta",
        "status": "Active",
        "website": "",
        "industry": "B2B",
        "all_locations": "x",
        "team_size": 3,
        "one_liner": "y",
    },
    {
        "id": 4,
        "name": "Delta",
        "status": "Active",
        "website": "https://d.io",
        "industry": "Fintech",
        "all_locations": "Lima, Peru",
        "team_size": 40,
        "one_liner": "Payments",
        "batch": "S23",
        "url": "u",
    },
]


def test_eligibility_and_deterministic_sample() -> None:
    assert [c["name"] for c in RAW if download_data.eligible(c)] == ["Gamma", "Delta"]
    sample = download_data.build_eval_set(RAW, sample_size=1, seed=1)
    assert len(sample) == 1 and "location" in sample[0] and "yc_url" in sample[0]
    assert download_data.build_eval_set(RAW, sample_size=1, seed=1) == sample


@respx.mock
def test_download_data_cli_writes_eval_set_and_manifest(tmp_path: Path) -> None:
    respx.get(download_data.RAW_URL).mock(return_value=httpx.Response(200, json=RAW))
    raw = tmp_path / "raw.json"
    out = tmp_path / "eval.json"
    manifest = tmp_path / "MANIFEST.txt"
    code = download_data.main(
        [
            "--raw",
            str(raw),
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--sample-size",
            "5",
        ]
    )
    assert code == 0 and raw.exists()
    data = json.loads(out.read_text())
    assert data["n"] == 2 and [c["name"] for c in data["companies"]] == ["Gamma", "Delta"]
    text = manifest.read_text()
    assert "raw_sha256: " + download_data.sha256_of(raw) in text
    assert "eval_sha256: " + download_data.sha256_of(out) in text
    assert "eligible_records: 2" in text


def test_batch_run_writes_outputs_and_demo(tmp_path: Path) -> None:
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"companies": download_data.build_eval_set(RAW)}))
    out_dir = tmp_path / "outputs"
    demo = tmp_path / "demo.json"
    code = batch_run.main(
        [
            "--ground-truth",
            str(gt),
            "--limit",
            "2",
            "--concurrency",
            "2",
            "--out",
            str(out_dir),
            "--demo",
            str(demo),
        ]
    )
    assert code == 0
    assert sorted(p.name for p in out_dir.glob("*.json")) == ["delta.json", "gamma.json"]
    payload = json.loads(demo.read_text())
    assert payload["mode"].startswith("fallback") and len(payload["cases"]) == 2
    assert payload["cases"][0]["result"]["profile"]["name"] == "Gamma"
    assert payload["cases"][0]["yc"]["industry"] == "B2B"

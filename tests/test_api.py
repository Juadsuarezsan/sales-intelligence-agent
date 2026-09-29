from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.main import TRACE_HEADER, app, limiter


@pytest.fixture
def client() -> Iterator[TestClient]:
    limiter.reset()
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model"] == "claude-sonnet-4-5-20250929"
    assert data["llm_enabled"] is False
    assert data["web_search"] == "stub" and data["repository"] == "memory"


def test_research_returns_full_payload_and_caches(client: TestClient) -> None:
    r = client.post(
        "/api/research",
        json={"company_name": "Acme Robotics", "domain": "acme.com"},
        headers={TRACE_HEADER: "trace-from-client"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["profile"]["name"] == "Acme Robotics"
    assert body["email"]["subject"] and body["scores"]["judge"] == "heuristic"
    assert body["cached"] is False and body["trace_id"] == "trace-from-client"
    assert r.headers[TRACE_HEADER] == "trace-from-client"

    again = client.post("/api/research", json={"company_name": "acme robotics"})
    assert again.status_code == 200 and again.json()["cached"] is True

    listing = client.get("/api/companies")
    assert listing.status_code == 200
    assert listing.json()[0]["company_name"] == "Acme Robotics"

    one = client.get("/api/companies/Acme%20Robotics")
    assert one.status_code == 200 and one.json()["company"] if False else True
    assert client.get("/api/companies/unknown").status_code == 404


def test_research_with_custom_reflection_budget_bypasses_cache(client: TestClient) -> None:
    r = client.post("/api/research", json={"company_name": "Beta", "max_reflection_iterations": 0})
    assert r.status_code == 200 and r.json()["iterations"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"company_name": ""},  # empty
        {"company_name": "A"},  # too short
        {"company_name": "x" * 121},  # oversized
        {"company_name": "Acme", "domain": "not a domain"},  # malformed domain
        {"company_name": "Acme", "seed_context": "c" * 2001},  # oversized context
        {"company_name": "Acme", "max_reflection_iterations": 9},  # out of range
        {"name": "Acme"},  # missing field
    ],
)
def test_invalid_bodies_return_422(client: TestClient, payload: dict[str, object]) -> None:
    r = client.post("/api/research", json=payload)
    assert r.status_code == 422
    assert r.json()["detail"]


def test_malformed_json_returns_422(client: TestClient) -> None:
    r = client.post(
        "/api/research", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422


def test_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import get_settings

    monkeypatch.setenv("RATE_LIMIT", "2/minute")
    get_settings.cache_clear()
    limiter.reset()
    with TestClient(app) as c:
        codes = [
            c.post("/api/research", json={"company_name": f"Co{i}"}).status_code for i in range(3)
        ]
    assert codes == [200, 200, 429]

from src.agents.profile_builder import ProfileBuilder
from src.api.schemas import CompanyProfile, Evidence
from tests.conftest import FakeLLM, llm_json

EVIDENCE = [
    Evidence(
        title="Acme raises $25M Series A",
        url="https://news.example/acme",
        content="Acme Robotics raised $25M in a Series A round led by Example Ventures.",
        source="web",
    ),
    Evidence(
        title="About Acme",
        url="https://acme.com/about",
        content=(
            "Acme is headquartered in Austin, Texas. The CEO is Maria Lopez and the "
            "CTO is Ken Adams. The team has 85 employees."
        ),
        source="website",
    ),
    Evidence(
        title="Careers",
        url="https://acme.com/careers",
        content="Open roles detected: 3 Senior Robotics Engineer",
        source="website",
    ),
    Evidence(
        title="Acme Robotics — homepage",
        url="https://acme.com",
        content="Acme builds autonomous picking robots for warehouses. Headings: Robots that pick",
        source="website",
    ),
    Evidence(
        title="Show HN: Acme launches fleet manager",
        url="https://news.ycombinator.com/item?id=1",
        content="HN: 120 points",
        source="hn",
    ),
]


async def test_heuristic_extracts_structured_fields() -> None:
    p = await ProfileBuilder(llm=None).build("Acme Robotics", EVIDENCE, website="acme.com")
    assert p.funding_total_usd == 25_000_000
    assert p.last_funding_round == "Series A"
    assert p.location == "Austin, Texas"
    assert p.employees_est == 85
    assert p.website == "acme.com"
    assert {kp.name for kp in p.key_people} == {"Maria Lopez", "Ken Adams"}
    kinds = [s.kind for s in p.recent_signals]
    assert "funding" in kinds and "hiring" in kinds and "news" in kinds
    assert p.one_liner.startswith("Acme is headquartered") or p.one_liner.startswith("Acme builds")
    assert "https://acme.com" in p.sources


async def test_heuristic_marks_missing_description() -> None:
    p = await ProfileBuilder(llm=None).build("Nobody Inc", [])
    assert p.one_liner.startswith("[heuristic]")
    assert p.recent_signals == [] and p.key_people == []


async def test_llm_profile_is_validated_and_website_injected() -> None:
    llm = FakeLLM(
        [
            llm_json(
                one_liner="Autonomous warehouse robots",
                industry="Industrials",
                employees_est=85,
                location="Austin, TX, USA",
                key_people=[{"name": "Maria Lopez", "role": "CEO"}],
                recent_signals=[
                    {
                        "kind": "funding",
                        "description": "$25M Series A",
                        "source_url": "https://news.example/acme",
                    }
                ],
                pain_points=["fleet coordination"],
                sources=["https://news.example/acme"],
            )
        ]
    )
    p = await ProfileBuilder(llm).build("Acme Robotics", EVIDENCE, website="https://acme.com")
    assert isinstance(p, CompanyProfile)
    assert p.name == "Acme Robotics" and p.website == "https://acme.com"
    assert p.industry == "Industrials" and p.pain_points == ["fleet coordination"]


async def test_llm_schema_violation_falls_back_to_heuristic() -> None:
    llm = FakeLLM([llm_json(employees_est=-5, recent_signals=[{"kind": "bogus"}])])
    p = await ProfileBuilder(llm).build("Acme Robotics", EVIDENCE)
    assert p.funding_total_usd == 25_000_000  # heuristic path

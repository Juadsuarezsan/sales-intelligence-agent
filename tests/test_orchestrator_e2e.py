import httpx
import pytest

from src.agents.orchestrator import build_components, build_graph, research_company
from src.api.schemas import Evidence
from src.config import Settings
from src.tools.hackernews import StubHackerNews
from src.tools.web_search import StubWebSearch
from src.tools.website import StubWebsiteFetcher
from tests.conftest import FakeLLM, RoutedLLM, llm_json


async def test_full_research_returns_profile_email_hooks_and_scores() -> None:
    out = await research_company(build_graph(), company_name="Anthropic", website="anthropic.com")
    assert out.profile.name == "Anthropic"
    assert out.email.subject and out.hooks is not None and out.scores is not None
    assert out.iterations >= 1 and out.tool_calls >= 1
    assert out.mode == "fallback" and out.cost_usd == 0.0
    assert len(out.trace_id) == 32
    assert out.profile.website == "anthropic.com"
    assert any(s.kind == "hiring" for s in out.profile.recent_signals)


async def test_zero_reflection_iterations_skips_reflector() -> None:
    class ExplodingReflector:
        async def reflect(self, *_a: object, **_k: object) -> None:
            raise AssertionError("reflector must not run with 0 iterations")

    c = build_components(max_reflection_iterations=0)
    c.reflector = ExplodingReflector()  # type: ignore[assignment]
    out = await research_company(build_graph(c), company_name="Acme")
    assert out.iterations == 1


async def test_reflection_loop_runs_follow_up_rounds_and_respects_budget() -> None:
    class NeverEnough:
        async def reflect(self, company: str, evidence: list[Evidence]) -> object:
            from src.agents.reflector import ReflectionVerdict

            return ReflectionVerdict(
                sufficient=False,
                missing=["funding"],
                follow_up_queries=[f"{company} funding {len(evidence)}"],
                rationale="never",
            )

    c = build_components(max_reflection_iterations=2)
    c.reflector = NeverEnough()  # type: ignore[assignment]
    c.max_tool_calls = 100
    out = await research_company(build_graph(c), company_name="Acme")
    assert out.iterations == 3  # initial round + 2 reflection-triggered rounds

    c2 = build_components(max_reflection_iterations=3)
    c2.reflector = NeverEnough()  # type: ignore[assignment]
    c2.max_tool_calls = 6
    out2 = await research_company(build_graph(c2), company_name="Acme")
    assert out2.tool_calls <= 6


async def test_tool_failures_do_not_abort_the_run() -> None:
    class BrokenSearch:
        name = "broken"

        async def search(self, query: str, max_results: int = 5) -> list[Evidence]:
            raise httpx.ConnectError("down")

    class BrokenNews:
        name = "broken"

        async def search(self, company: str, limit: int = 5) -> list[Evidence]:
            raise TimeoutError()

    c = build_components(web=BrokenSearch(), news=BrokenNews(), website=StubWebsiteFetcher())
    out = await research_company(build_graph(c), company_name="Acme", website="acme.com")
    assert out.profile.name == "Acme"
    assert out.tool_calls >= 1


async def test_llm_path_end_to_end_reports_tokens_and_cost() -> None:
    llm = RoutedLLM(
        {
            "planner": llm_json(queries=["Acme funding", "Acme CEO"]),
            "reflector": llm_json(sufficient=True, missing=[], follow_up_queries=[]),
            "builder": llm_json(
                one_liner="Warehouse robots",
                industry="Industrials",
                location="Austin, TX",
                employees_est=85,
                key_people=[{"name": "Maria Lopez", "role": "CEO"}],
                recent_signals=[
                    {"kind": "funding", "description": "$25M Series A", "source_url": "https://n"}
                ],
                sources=["https://n"],
            ),
            "extractor": llm_json(pain_point="fleet ops", recent_signal="$25M Series A", angle="a"),
            "writer": llm_json(
                subject="Series A",
                greeting="Hi Maria,",
                hook="Congrats on Acme's $25M Series A.",
                value_prop="Industrials teams automate fleet ops with us.",
                cta="20-minute call next week?",
            ),
            "judge": llm_json(personalization=5, accuracy=5, cta_clarity=4, rationale="r"),
        }
    )
    settings = Settings(PRICE_INPUT_PER_MTOK=3.0, PRICE_OUTPUT_PER_MTOK=15.0)
    c = build_components(settings, llm=llm, judge=llm, web=StubWebSearch(), news=StubHackerNews())
    out = await research_company(
        build_graph(c), company_name="Acme", settings=settings, llm_enabled=True
    )
    nodes = [call["node"] for call in llm.calls]
    assert nodes == ["planner", "reflector", "builder", "extractor", "writer", "judge"]
    assert out.mode == "llm"
    assert out.profile.industry == "Industrials" and out.email.greeting == "Hi Maria,"
    assert out.scores is not None and out.scores.judge == "llm"
    assert out.input_tokens == 6 * 200 and out.output_tokens == 6 * 80
    assert out.cost_usd == pytest.approx((1200 * 3 + 480 * 15) / 1e6)


async def test_llm_garbage_everywhere_still_completes() -> None:
    c = build_components(llm=FakeLLM(default="garbage"), judge=FakeLLM(default="garbage"))
    out = await research_company(build_graph(c), company_name="Acme")
    assert out.email.cta and out.scores is not None and out.scores.judge == "heuristic"

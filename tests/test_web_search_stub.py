import asyncio
from typing import Any

import pytest
from pytest_mock import MockerFixture
from tenacity import wait_fixed

from src.observability import start_run
from src.tools.web_search import StubWebSearch, TavilyWebSearch


async def test_funding_query_returns_funding_result() -> None:
    out = await StubWebSearch().search("Acme Corp recent funding round")
    assert len(out) >= 1
    assert any("series" in r["content"].lower() or "$" in r["content"] for r in out)


async def test_always_returns_at_least_one_result() -> None:
    out = await StubWebSearch().search("totally random query")
    assert len(out) >= 1 and out[0]["source"] == "web"


async def test_tavily_maps_results_and_counts_paid_search(mocker: MockerFixture) -> None:
    fake_client = mocker.Mock()
    fake_client.search = mocker.AsyncMock(
        return_value={
            "results": [
                {"title": "T", "url": "https://u", "content": "c", "score": 0.9},
                {"title": "T2", "url": "https://u2", "content": "c2"},
            ]
        }
    )
    mocker.patch("tavily.AsyncTavilyClient", return_value=fake_client)
    tracker = start_run()
    tool = TavilyWebSearch("tvly-test", timeout_seconds=1)
    out = await tool.search("Acme funding", max_results=2)
    assert [r["url"] for r in out] == ["https://u", "https://u2"]
    assert out[1]["score"] == 0.0
    assert tracker.search_calls == 1
    fake_client.search.assert_awaited_once()
    assert fake_client.search.await_args.kwargs["max_results"] == 2


async def test_tavily_times_out_and_retries(mocker: MockerFixture) -> None:
    async def slow(**_: Any) -> dict[str, Any]:
        await asyncio.sleep(0.2)
        return {"results": []}

    fake_client = mocker.Mock()
    fake_client.search = slow
    mocker.patch("tavily.AsyncTavilyClient", return_value=fake_client)
    tool = TavilyWebSearch("tvly-test", timeout_seconds=0.01)
    fast = tool.search.retry_with(wait=wait_fixed(0))  # type: ignore[attr-defined]
    with pytest.raises(asyncio.TimeoutError):
        await fast(tool, "Acme")
    assert fast.statistics["attempt_number"] == 3

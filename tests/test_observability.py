from typing import Any

from src.observability import (
    UsageTracker,
    configure_logging,
    current_trace_id,
    current_usage,
    log_node,
    new_trace_id,
    start_run,
)


def test_usage_tracker_cost_uses_price_table() -> None:
    t = UsageTracker()
    t.record_llm(1_000_000, 100_000)
    t.record_search()
    t.record_search()
    cost = t.cost_usd(
        price_input_per_mtok=3.0, price_output_per_mtok=15.0, cost_per_search_usd=0.001
    )
    assert cost == 3.0 + 1.5 + 0.002
    assert t.llm_calls == 1 and t.search_calls == 2


def test_start_run_binds_tracker_and_trace_id() -> None:
    tracker = start_run("abc123")
    assert current_usage() is tracker
    assert current_trace_id() == "abc123"
    generated = new_trace_id()
    assert len(generated) == 32 and current_trace_id() == generated


async def test_log_node_records_latency_per_node() -> None:
    tracker = start_run()

    @log_node("demo")
    async def node(state: dict[str, Any]) -> dict[str, Any]:
        return {"evidence": [1, 2, 3], "text": "x" * 200}

    out = await node({"company_name": "Acme", "evidence": []})
    assert out["evidence"] == [1, 2, 3]
    assert "demo" in tracker.node_latency_ms


def test_configure_logging_does_not_raise() -> None:
    configure_logging("DEBUG")
    configure_logging("INFO")

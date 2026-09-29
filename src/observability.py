"""Tracing, token accounting, cost computation and per-node logging.

A research run is identified by a ``trace_id`` stored in a :class:`ContextVar`
so every log line emitted by any node, tool or LLM call carries it. Token
usage is accumulated in a :class:`UsageTracker`, also held in a context
variable, so the LLM client can record usage without the graph nodes having
to thread it through their return values.
"""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_usage: ContextVar[UsageTracker | None] = ContextVar("usage", default=None)

T = TypeVar("T")


@dataclass
class UsageTracker:
    """Accumulates token and tool usage for one research run.

    Attributes:
        input_tokens: Total prompt tokens sent to the LLM.
        output_tokens: Total completion tokens received from the LLM.
        llm_calls: Number of LLM requests.
        search_calls: Number of paid web-search requests (Tavily).
        node_latency_ms: Wall-clock milliseconds spent per graph node.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    search_calls: int = 0
    node_latency_ms: dict[str, int] = field(default_factory=dict)

    def record_llm(self, input_tokens: int, output_tokens: int) -> None:
        """Add the usage of one LLM call.

        Args:
            input_tokens: Prompt tokens of the call.
            output_tokens: Completion tokens of the call.
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.llm_calls += 1

    def record_search(self) -> None:
        """Count one paid web-search call."""
        self.search_calls += 1

    def cost_usd(
        self,
        *,
        price_input_per_mtok: float,
        price_output_per_mtok: float,
        cost_per_search_usd: float,
    ) -> float:
        """Compute the monetary cost of the run.

        Args:
            price_input_per_mtok: USD per million input tokens.
            price_output_per_mtok: USD per million output tokens.
            cost_per_search_usd: USD per paid web search.

        Returns:
            Cost in USD rounded to six decimals.
        """
        llm = (
            self.input_tokens * price_input_per_mtok + self.output_tokens * price_output_per_mtok
        ) / 1_000_000
        return round(llm + self.search_calls * cost_per_search_usd, 6)


def new_trace_id() -> str:
    """Generate a new trace id and bind it to the current context.

    Returns:
        The 32-character hexadecimal trace id.
    """
    tid = uuid.uuid4().hex
    _trace_id.set(tid)
    return tid


def current_trace_id() -> str:
    """Return the trace id bound to the current context, creating one if absent."""
    tid = _trace_id.get()
    return tid if tid is not None else new_trace_id()


def start_run(trace_id: str | None = None) -> UsageTracker:
    """Bind a fresh :class:`UsageTracker` (and trace id) to the current context.

    Args:
        trace_id: Optional externally supplied trace id (e.g. from a request header).

    Returns:
        The tracker that subsequent LLM/tool calls will record into.
    """
    if trace_id:
        _trace_id.set(trace_id)
    else:
        new_trace_id()
    tracker = UsageTracker()
    _usage.set(tracker)
    return tracker


def current_usage() -> UsageTracker | None:
    """Return the tracker bound to the current context, if any."""
    return _usage.get()


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru with a structured, trace-aware format.

    Args:
        level: Minimum level to emit (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "trace={extra[trace_id]} | <cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}"
        ),
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
    logger.configure(patcher=_inject_trace_id)


def _inject_trace_id(record: Record) -> None:
    record["extra"].setdefault("trace_id", _trace_id.get() or "-")


def _summarise(value: Any) -> Any:
    """Reduce a state value to something short enough to log."""
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, str):
        return value if len(value) <= 80 else value[:77] + "..."
    if hasattr(value, "model_dump"):
        return type(value).__name__
    return value


def log_node(
    name: str,
) -> Callable[
    [Callable[[Any], Awaitable[dict[str, Any]]]], Callable[[Any], Awaitable[dict[str, Any]]]
]:
    """Decorate a LangGraph node to log its input state, output update and latency.

    Args:
        name: Node name used in the log lines and in ``UsageTracker.node_latency_ms``.

    Returns:
        A decorator for ``async def node(state) -> dict`` functions.
    """

    def decorator(
        fn: Callable[[Any], Awaitable[dict[str, Any]]],
    ) -> Callable[[Any], Awaitable[dict[str, Any]]]:
        @wraps(fn)
        async def wrapper(state: Any) -> dict[str, Any]:
            t0 = time.perf_counter()
            state_view = {k: _summarise(v) for k, v in dict(state).items()}
            logger.bind(node=name).debug(f"node={name} input={state_view}")
            update = await fn(state)
            elapsed = int((time.perf_counter() - t0) * 1000)
            tracker = _usage.get()
            if tracker is not None:
                tracker.node_latency_ms[name] = tracker.node_latency_ms.get(name, 0) + elapsed
            out_view = {k: _summarise(v) for k, v in update.items()}
            logger.bind(node=name).info(f"node={name} latency_ms={elapsed} output={out_view}")
            return update

        return wrapper

    return decorator

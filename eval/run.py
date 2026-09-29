"""``python -m eval.run``: evaluate baselines, the agent and the reflection ablation.

Writes ``eval/runs/<date>-<name>.json`` with every per-company row and
regenerates ``eval/RESULTS.md`` with the spec's comparison table. Without
``ANTHROPIC_API_KEY`` and ``TAVILY_API_KEY`` the run uses the deterministic
stubs; every number it produces is then labelled as such and the cells that
require the real system stay ``pendiente``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from eval.baselines import SystemRunner, agent_runner, single_search_email, template_only
from eval.ground_truth import DEFAULT_GROUND_TRUTH, GroundTruthCompany, load_ground_truth
from eval.metrics import FIELDS, FieldScores, aggregate_accuracy, distribution, score_profile
from eval.report import render_results
from src.agents.orchestrator import AgentComponents, build_components
from src.api.schemas import ResearchResponse
from src.config import Settings, get_settings
from src.observability import configure_logging

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "eval" / "runs"
RESULTS_MD = REPO_ROOT / "eval" / "RESULTS.md"

SYSTEM_LABELS: dict[str, str] = {
    "template_only": "Template only",
    "single_search_email": "Single-search + email",
    "agent_reflection_3": "Agent loop con reflection (este)",
}


@dataclass
class SystemResult:
    """Aggregated outcome of one system over the eval set.

    Attributes:
        system: System identifier.
        n: Number of companies evaluated.
        accuracy: Micro precision/recall/F1 over :data:`eval.metrics.FIELDS`.
        per_field: Count of ``tp``/``fp``/``fn`` per field.
        personalization_proxy: Distribution of the keyword personalization score.
        judge_personalization: Distribution of the rubric personalization score.
        judge_overall: Distribution of the rubric overall score.
        judge: ``llm`` or ``heuristic``.
        tool_calls: Distribution of tool calls per company.
        iterations: Distribution of plan/execute rounds per company.
        latency_ms: Distribution of latency per company.
        cost_usd: Distribution of cost per company.
        mode: ``llm`` or ``fallback``.
        rows: Per-company rows.
    """

    system: str
    n: int
    accuracy: dict[str, float]
    per_field: dict[str, dict[str, int]]
    personalization_proxy: dict[str, float]
    judge_personalization: dict[str, float]
    judge_overall: dict[str, float]
    judge: str
    tool_calls: dict[str, float]
    iterations: dict[str, float]
    latency_ms: dict[str, float]
    cost_usd: dict[str, float]
    mode: str
    rows: list[dict[str, Any]] = field(default_factory=list)


def _row(company: GroundTruthCompany, r: ResearchResponse, fs: FieldScores) -> dict[str, Any]:
    return {
        "company": company.name,
        "batch": company.batch,
        "precision": round(fs.precision, 4),
        "recall": round(fs.recall, 4),
        "fields": fs.per_field,
        "personalization_proxy": r.email.personalization_score,
        "judge_personalization": r.scores.personalization if r.scores else None,
        "judge_accuracy": r.scores.accuracy if r.scores else None,
        "judge_cta": r.scores.cta_clarity if r.scores else None,
        "judge_overall": r.scores.overall if r.scores else None,
        "tool_calls": r.tool_calls,
        "iterations": r.iterations,
        "latency_ms": r.latency_ms,
        "cost_usd": r.cost_usd,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "subject": r.email.subject,
        "hook": r.email.hook,
    }


async def evaluate_system(
    name: str,
    companies: list[GroundTruthCompany],
    runner: SystemRunner,
    *,
    concurrency: int = 4,
) -> SystemResult:
    """Run one system over the eval set and aggregate.

    Args:
        name: System identifier.
        companies: Ground-truth companies.
        runner: Async callable producing a :class:`ResearchResponse`.
        concurrency: Maximum companies processed at once.

    Returns:
        The aggregated result.
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(company: GroundTruthCompany) -> tuple[ResearchResponse, FieldScores]:
        async with sem:
            r = await runner(company)
        return r, score_profile(r.profile, company)

    outcomes = await asyncio.gather(*(one(c) for c in companies))
    rows = [_row(c, r, fs) for c, (r, fs) in zip(companies, outcomes, strict=True)]
    per_field: dict[str, dict[str, int]] = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    for _, fs in outcomes:
        for f, outcome in fs.per_field.items():
            if outcome in per_field[f]:
                per_field[f][outcome] += 1
    responses = [r for r, _ in outcomes]
    judge = (
        "llm"
        if responses and all(r.scores and r.scores.judge == "llm" for r in responses)
        else "heuristic"
    )
    mode = "llm" if responses and all(r.mode == "llm" for r in responses) else "fallback"
    return SystemResult(
        system=name,
        n=len(companies),
        accuracy=aggregate_accuracy([fs for _, fs in outcomes]),
        per_field=per_field,
        personalization_proxy=distribution([r.email.personalization_score for r in responses]),
        judge_personalization=distribution(
            [float(r.scores.personalization) for r in responses if r.scores]
        ),
        judge_overall=distribution([r.scores.overall for r in responses if r.scores]),
        judge=judge,
        tool_calls=distribution([float(r.tool_calls) for r in responses]),
        iterations=distribution([float(r.iterations) for r in responses]),
        latency_ms=distribution([float(r.latency_ms) for r in responses]),
        cost_usd=distribution([r.cost_usd for r in responses]),
        mode=mode,
        rows=rows,
    )


def build_runners(
    components: AgentComponents, settings: Settings, *, ablation: tuple[int, ...] = (0, 1, 2, 3)
) -> dict[str, SystemRunner]:
    """Assemble the systems to evaluate.

    Args:
        components: Shared component bundle.
        settings: Settings for cost computation.
        ablation: Reflection budgets for the ablation study.

    Returns:
        Ordered mapping ``system name -> runner``.
    """

    async def _template(company: GroundTruthCompany) -> ResearchResponse:
        return await template_only(company, components, settings)

    async def _single(company: GroundTruthCompany) -> ResearchResponse:
        return await single_search_email(company, components, settings)

    runners: dict[str, SystemRunner] = {
        "template_only": _template,
        "single_search_email": _single,
    }
    for k in ablation:
        runners[f"agent_reflection_{k}"] = agent_runner(
            components, settings, max_reflection_iterations=k
        )
    return runners


async def run_eval(
    *,
    companies: list[GroundTruthCompany],
    components: AgentComponents,
    settings: Settings,
    concurrency: int = 4,
) -> dict[str, Any]:
    """Evaluate every system and return the JSON-serialisable report.

    Args:
        companies: Eval set.
        components: Component bundle.
        settings: Settings.
        concurrency: Parallel companies per system.

    Returns:
        Report dictionary (what gets written to ``eval/runs/``).
    """
    results: list[SystemResult] = []
    for name, runner in build_runners(components, settings).items():
        logger.info(f"evaluating system={name} n={len(companies)}")
        results.append(await evaluate_system(name, companies, runner, concurrency=concurrency))
    llm_enabled = components.llm_enabled
    search_real = components.web.name == "tavily"
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_companies": len(companies),
        "ground_truth": str(DEFAULT_GROUND_TRUTH.relative_to(REPO_ROOT)),
        "model": settings.anthropic_model,
        "judge_model": settings.judge_model,
        "llm_enabled": llm_enabled,
        "web_search": components.web.name,
        "news_search": components.news.name,
        "website_fetcher": components.website.name,
        "mode": "llm" if llm_enabled and search_real else "fallback determinista, sin LLM",
        "fields": list(FIELDS),
        "systems": [asdict(r) for r in results],
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baselines, agent and ablation.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N companies"
    )
    parser.add_argument("--name", default="eval", help="Run name used in the output file name")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_MD)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Path of the JSON run that was written.
    """
    args = _parse_args(argv)
    configure_logging(args.log_level)
    settings = get_settings()
    companies = load_ground_truth(args.ground_truth)
    if args.limit:
        companies = companies[: args.limit]
    components = build_components(settings)
    report = asyncio.run(
        run_eval(
            companies=companies,
            components=components,
            settings=settings,
            concurrency=args.concurrency,
        )
    )
    runs_dir = Path(args.runs_dir)
    results_path = Path(args.results)
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    out = runs_dir / f"{stamp}-{args.name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    results_path.write_text(render_results(report, run_file=out.name), encoding="utf-8")
    print(f"wrote {out} and {results_path}")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])

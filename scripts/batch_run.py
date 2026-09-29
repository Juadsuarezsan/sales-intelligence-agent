"""Batch-process companies from the eval set with bounded concurrency.

Usage::

    python scripts/batch_run.py --limit 100 --concurrency 4 --out data/outputs
    python scripts/batch_run.py --limit 8 --demo demo/predictions.json

Every result is saved through the configured repository (PostgreSQL when
``DATABASE_URL`` is set, in-memory otherwise) and written as one JSON file
per company under ``--out``. With ``--demo`` a compact gallery file is also
produced for ``demo/index.html``. Without API keys the run uses the
deterministic stubs and the output is labelled ``mode="fallback"``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from eval.ground_truth import DEFAULT_GROUND_TRUTH, GroundTruthCompany, load_ground_truth
from src.agents.orchestrator import build_components, build_graph, research_company
from src.api.schemas import ResearchResponse
from src.config import get_settings
from src.observability import configure_logging
from src.storage import PostgresRepository, build_repository

REPO_ROOT = Path(__file__).resolve().parent.parent


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "company"


async def run_batch(
    companies: list[GroundTruthCompany],
    *,
    concurrency: int,
    out_dir: Path | None,
) -> list[ResearchResponse]:
    """Research every company with at most ``concurrency`` in flight.

    Args:
        companies: Targets.
        concurrency: Parallelism.
        out_dir: Directory for per-company JSON files (``None`` to skip).

    Returns:
        Results in input order.
    """
    settings = get_settings()
    components = build_components(settings)
    graph = build_graph(components)
    repository = build_repository(settings)
    if isinstance(repository, PostgresRepository):
        await repository.open()
    sem = asyncio.Semaphore(concurrency)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    async def one(company: GroundTruthCompany) -> ResearchResponse:
        async with sem:
            result = await research_company(
                graph, company_name=company.name, website=company.website or None
            )
        await repository.save(result)
        if out_dir is not None:
            (out_dir / f"{_slug(company.name)}.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
        return result

    try:
        return list(await asyncio.gather(*(one(c) for c in companies)))
    finally:
        await repository.close()


def demo_payload(
    results: list[ResearchResponse], companies: list[GroundTruthCompany]
) -> dict[str, Any]:
    """Build the compact gallery file consumed by ``demo/index.html``.

    Args:
        results: Research results.
        companies: Matching ground-truth records (for YC metadata on the card).

    Returns:
        JSON-serialisable dictionary.
    """
    settings = get_settings()
    cases = []
    for r, c in zip(results, companies, strict=True):
        cases.append(
            {
                "yc": {
                    "batch": c.batch,
                    "industry": c.industry,
                    "location": c.location,
                    "team_size": c.team_size,
                    "one_liner": c.one_liner,
                    "yc_url": c.yc_url,
                },
                "result": r.model_dump(mode="json"),
            }
        )
    return {
        "project": "sales-intelligence-agent",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": "scripts/batch_run.py",
        "mode": "llm" if settings.llm_enabled else "fallback determinista, sin LLM",
        "model": settings.anthropic_model,
        "note": (
            "Casos pre-generados por el pipeline del repositorio. Sin ANTHROPIC_API_KEY ni "
            "TAVILY_API_KEY los nodos usan su fallback determinista y las búsquedas son stubs: "
            "los perfiles y emails ilustran el formato, no la calidad del sistema."
        ),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "outputs")
    parser.add_argument("--demo", type=Path, default=None, help="Also write the gallery JSON here")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    companies = load_ground_truth(args.ground_truth)[: args.limit]
    results = asyncio.run(run_batch(companies, concurrency=args.concurrency, out_dir=args.out))
    logger.info(f"batch done n={len(results)} out={args.out}")
    if args.demo is not None:
        args.demo.parent.mkdir(parents=True, exist_ok=True)
        args.demo.write_text(
            json.dumps(demo_payload(results, companies), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.demo}")
    print(f"processed {len(results)} companies; outputs in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

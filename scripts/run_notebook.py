"""Build and execute ``notebooks/demo.ipynb`` without Jupyter.

The notebook is generated from the cell list below (so it stays in sync with
the code), then every code cell is executed in one shared namespace with
stdout captured as the cell output. It runs with the deterministic stubs, so
no API key is needed.

Usage::

    python scripts/run_notebook.py notebooks/demo.ipynb          # build + execute
    python scripts/run_notebook.py notebooks/demo.ipynb --no-exec
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# B2B Sales Intelligence Agent — demo end-to-end\n\n"
        "Este notebook ejecuta el pipeline completo (plan → execute → reflect → build → hooks → "
        "email → score) con los **stubs deterministas**: no necesita `ANTHROPIC_API_KEY` ni "
        "`TAVILY_API_KEY`. Con llaves configuradas en `.env`, las mismas celdas usan Claude y "
        "Tavily y el campo `mode` cambia de `fallback` a `llm`.",
    ),
    (
        "code",
        "import sys, json, asyncio\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd() if (Path.cwd() / 'pyproject.toml').exists() else Path.cwd().parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import os\n"
        "os.environ.setdefault('USE_REAL_HN', 'false')\n"
        "from src.config import get_settings\n"
        "from src.observability import configure_logging\n"
        "configure_logging('WARNING')\n"
        "s = get_settings()\n"
        "print('model:', s.anthropic_model, '| llm_enabled:', s.llm_enabled, "
        "'| tavily:', bool(s.tavily_api_key))",
    ),
    ("markdown", "## 1. Construir el grafo y investigar una empresa"),
    (
        "code",
        "from src.agents.orchestrator import build_components, build_graph, research_company\n"
        "components = build_components(s)\n"
        "graph = build_graph(components)\n"
        "result = asyncio.run(\n"
        "    research_company(graph, company_name='Rescale', website='rescale.com')\n"
        ")\n"
        "print('mode:', result.mode, '| iterations:', result.iterations, '| tool_calls:', "
        "result.tool_calls, '| latency_ms:', result.latency_ms, '| cost_usd:', result.cost_usd)",
    ),
    ("markdown", "## 2. Perfil estructurado (`CompanyProfile`)"),
    ("code", "print(result.profile.model_dump_json(indent=2))"),
    ("markdown", "## 3. Ganchos de personalización y email"),
    (
        "code",
        "print(result.hooks.model_dump_json(indent=2))\n"
        "print()\n"
        "print('Subject:', result.email.subject)\n"
        "print(result.email.body())",
    ),
    ("markdown", "## 4. Puntuación del juez (rúbrica en `src/agents/quality_scorer.py`)"),
    (
        "code",
        "from src.agents.quality_scorer import RUBRIC\n"
        "print(result.scores.model_dump_json(indent=2))\n"
        "print()\n"
        "print(RUBRIC[:600], '...')",
    ),
    ("markdown", "## 5. Baselines y ablation sobre una muestra del eval set"),
    (
        "code",
        "from eval.ground_truth import load_ground_truth\n"
        "from eval.run import run_eval\n"
        "companies = load_ground_truth(ROOT / 'data/eval/yc_ground_truth_200.json')[:5]\n"
        "report = asyncio.run(run_eval(companies=companies, components=components, settings=s))\n"
        "print('mode:', report['mode'])\n"
        "for sysres in report['systems']:\n"
        "    acc = sysres['accuracy']\n"
        "    print(f\"{sysres['system']:22s} P={acc['precision']:.2f} R={acc['recall']:.2f} \"\n"
        "          f\"tool_calls={sysres['tool_calls']['mean']:.1f} \"\n"
        "          f\"judge_pers={sysres['judge_personalization']['mean']:.2f} \"\n"
        "          f\"({sysres['judge']})\")",
    ),
    (
        "markdown",
        "## 6. Persistencia\n\nEl repositorio en memoria (o PostgreSQL con `DATABASE_URL`) "
        "guarda cada resultado por nombre de empresa; el API lo usa como caché.",
    ),
    (
        "code",
        "from src.storage import build_repository\n"
        "repo = build_repository(s)\n"
        "asyncio.run(repo.save(result))\n"
        "cached = asyncio.run(repo.get('rescale'))\n"
        "print(repo.name, '->', cached.company_name, cached.trace_id)",
    ),
    (
        "markdown",
        "## Nota\n\nCon stubs, los perfiles contienen datos enlatados (por ejemplo una ronda "
        '"Series B" genérica) y no describen la empresa real: ilustran el formato y la '
        "mecánica del pipeline. La tabla con Claude + Tavily se genera con `python -m eval.run` "
        "y queda en `eval/RESULTS.md`.",
    ),
]


def build_notebook() -> nbformat.NotebookNode:
    """Create the notebook object from :data:`CELLS`."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    for kind, source in CELLS:
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(source))
    return nb


def execute(nb: nbformat.NotebookNode) -> None:
    """Execute code cells in a shared namespace, capturing stdout as outputs.

    Args:
        nb: Notebook to execute in place.

    Raises:
        RuntimeError: If a cell raises.
    """
    namespace: dict[str, Any] = {"__name__": "__notebook__"}
    count = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(cell.source, f"<cell {count}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001 - re-raised with cell context
            raise RuntimeError(f"cell {count} failed: {exc!r}\n{cell.source}") from exc
        cell.execution_count = count
        text = buffer.getvalue()
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=text)] if text else []


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, nargs="?", default=REPO_ROOT / "notebooks" / "demo.ipynb"
    )
    parser.add_argument("--no-exec", action="store_true", help="Only (re)build the notebook")
    args = parser.parse_args(argv)
    nb = build_notebook()
    if not args.no_exec:
        execute(nb)
    args.path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(args.path))
    print(f"wrote {args.path} ({len(nb.cells)} cells, executed={not args.no_exec})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

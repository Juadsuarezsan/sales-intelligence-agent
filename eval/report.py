"""Render ``eval/RESULTS.md`` from a run report.

The Markdown contains the spec's comparison table with every row and column,
measured values where the run could measure them without API keys, and the
literal string ``pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY)``
where the real system is required.
"""

from __future__ import annotations

from typing import Any

from src.agents.quality_scorer import RUBRIC

PENDING = "pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY)"
PENDING_LLM = "pendiente (requiere ANTHROPIC_API_KEY)"

_LABELS = {
    "template_only": "Template only",
    "single_search_email": "Single-search + email",
    "agent_reflection_0": "Agent loop, 0 reflexiones (ablation)",
    "agent_reflection_1": "Agent loop, 1 reflexión (ablation)",
    "agent_reflection_2": "Agent loop, 2 reflexiones (ablation)",
    "agent_reflection_3": "Agent loop con reflection (este)",
}


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def _sys(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for s in report["systems"]:
        if s["system"] == name:
            return dict(s)
    return None


def _acc_cell(s: dict[str, Any], real: bool) -> str:
    acc = s["accuracy"]
    measured = f"P {_pct(acc['precision'])} / R {_pct(acc['recall'])} / F1 {_pct(acc['f1'])}"
    if real:
        return measured
    if s["system"] == "template_only":
        return measured + " (sin búsqueda; medido)"
    return measured + " (stubs; no representa al sistema)"


def _pers_cell(s: dict[str, Any], real: bool) -> str:
    if real and s["judge"] == "llm":
        return f"{s['judge_personalization']['mean']:.2f} / 5 (juez LLM)"
    proxy = s["personalization_proxy"]["mean"]
    heur = s["judge_personalization"]["mean"]
    if s["system"] == "template_only":
        return f"proxy {proxy:.2f}; rúbrica heurística {heur:.2f}/5 (medido, sin LLM)"
    return f"{PENDING_LLM}; proxy heurístico {proxy:.2f}, rúbrica heurística {heur:.2f}/5"


def _lat_cell(s: dict[str, Any], real: bool) -> str:
    lat = s["latency_ms"]
    if real:
        return f"{lat['mean'] / 1000:.1f} s (p95 {lat['p95'] / 1000:.1f} s)"
    if s["system"] == "template_only":
        return f"{lat['mean']:.0f} ms (medido; sin I/O)"
    return f"{PENDING}; stubs: {lat['mean']:.0f} ms"


def _cost_cell(s: dict[str, Any], real: bool) -> str:
    if real:
        return f"${s['cost_usd']['mean']:.4f}"
    if s["system"] == "template_only":
        return "$0.0000 (sin LLM ni búsqueda; medido)"
    return f"{PENDING}; stubs: $0.0000"


def render_results(report: dict[str, Any], *, run_file: str) -> str:
    """Build the Markdown document.

    Args:
        report: Report produced by :func:`eval.run.run_eval`.
        run_file: File name of the JSON run (for provenance).

    Returns:
        Markdown text.
    """
    real = report["mode"] == "llm"
    lines: list[str] = []
    lines.append("# Resultados de evaluación — B2B Sales Intelligence Agent\n")
    lines.append(f"- Corrida: `eval/runs/{run_file}` generada el {report['generated_at']}")
    lines.append(
        f"- Empresas evaluadas: **{report['n_companies']}** "
        f"(ground truth: `{report['ground_truth']}`)"
    )
    lines.append(f"- Modo: **{report['mode']}**")
    lines.append(
        f"- Modelo del agente: `{report['model']}` · juez: `{report['judge_model']}` · "
        f"LLM habilitado: {report['llm_enabled']} · búsqueda: `{report['web_search']}` · "
        f"noticias: `{report['news_search']}` · sitio web: `{report['website_fetcher']}`"
    )
    lines.append("- Regenerar: `python -m eval.run`\n")

    if not real:
        lines.append(
            "> **Aviso.** Esta corrida se hizo sin llaves de API: los nodos LLM usan su "
            "fallback determinista y las búsquedas devuelven resultados enlatados (stubs). "
            "Los números marcados *stubs* miden únicamente la mecánica del pipeline "
            "(tool calls, iteraciones, extracción por reglas) y **no representan la "
            "calidad del sistema**. Las celdas `pendiente` se llenan al ejecutar el mismo "
            "comando con `ANTHROPIC_API_KEY` y `TAVILY_API_KEY` configuradas.\n"
        )

    lines.append("## Tabla comparativa (spec)\n")
    lines.append(
        "| Sistema | Accuracy (campos vs YC) | Personalization | Latencia | Costo/empresa |"
    )
    lines.append("|---|---|---|---|---|")
    for name in ("template_only", "single_search_email", "agent_reflection_3"):
        s = _sys(report, name)
        if s is None:
            continue
        lines.append(
            f"| {_LABELS[name]} | {_acc_cell(s, real)} | {_pers_cell(s, real)} | "
            f"{_lat_cell(s, real)} | {_cost_cell(s, real)} |"
        )
    lines.append("")
    lines.append(
        "Accuracy = precisión/recall micro-promediadas sobre los campos "
        f"`{'`, `'.join(report['fields'])}` del `CompanyProfile` contra la metadata de YC "
        "(reglas de match en `eval/metrics.py`). Personalization = criterio 1 de la rúbrica "
        "(1-5) juzgado por LLM sobre los emails; el *proxy heurístico* es el solapamiento "
        "de palabras clave de `score_personalization`.\n"
    )

    lines.append("## Ablation: iteraciones de reflection (0 / 1 / 2 / 3)\n")
    lines.append(
        "| Reflexiones | Accuracy | Rúbrica personalization | Tool calls (media / p95) | "
        "Iteraciones (media) | Latencia (media) | Costo/empresa |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for k in (0, 1, 2, 3):
        s = _sys(report, f"agent_reflection_{k}")
        if s is None:
            continue
        acc = s["accuracy"]
        tc = s["tool_calls"]
        pers = (
            f"{s['judge_personalization']['mean']:.2f}/5 (LLM)"
            if real and s["judge"] == "llm"
            else f"{PENDING_LLM}; heurística {s['judge_personalization']['mean']:.2f}/5"
        )
        acc_txt = f"P {_pct(acc['precision'])} / R {_pct(acc['recall'])}" + (
            "" if real else " (stubs)"
        )
        lat = f"{s['latency_ms']['mean']:.0f} ms" + ("" if real else " (stubs)")
        cost = f"${s['cost_usd']['mean']:.4f}" if real else f"{PENDING}"
        lines.append(
            f"| {k} | {acc_txt} | {pers} | {tc['mean']:.1f} / {tc['p95']:.0f} | "
            f"{s['iterations']['mean']:.1f} | {lat} | {cost} |"
        )
    lines.append("")
    lines.append(
        "El punto donde agregar reflexión deja de mejorar se lee de esta tabla cuando la "
        "corrida es real: la primera fila cuya accuracy/personalization no supera a la "
        "anterior marca el codo costo-calidad. "
        + (
            ""
            if real
            else "Con stubs las filas son idénticas por construcción (el reflector "
            "heurístico declara suficiencia en la primera ronda), así que la curva queda pendiente."
        )
        + "\n"
    )

    lines.append("## Distribución de tool calls por empresa\n")
    lines.append("| Sistema | media | p50 | p95 | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for entry in report["systems"]:
        calls: dict[str, float] = entry["tool_calls"]
        lines.append(
            f"| {_LABELS.get(entry['system'], entry['system'])} | {calls['mean']:.2f} | "
            f"{calls['p50']:.0f} | {calls['p95']:.0f} | {calls['min']:.0f} | {calls['max']:.0f} |"
        )
    lines.append("")

    lines.append("## Accuracy por campo (agente, 3 reflexiones)\n")
    agent = _sys(report, "agent_reflection_3")
    if agent is not None:
        lines.append("| Campo | TP | FP | FN |")
        lines.append("|---|---|---|---|")
        for f, counts in agent["per_field"].items():
            lines.append(f"| `{f}` | {counts['tp']} | {counts['fp']} | {counts['fn']} |")
        lines.append("")

    lines.append("## Rúbrica del LLM-as-judge (`src/agents/quality_scorer.py`)\n")
    lines.append("```text")
    lines.append(RUBRIC.strip())
    lines.append("```\n")

    lines.append("## Pendientes que requieren llaves\n")
    lines.append(
        "- Information accuracy real sobre las 200 empresas (Tavily + Claude): "
        f"{PENDING}.\n"
        f"- Personalization score con juez LLM sobre 100 emails: {PENDING_LLM}.\n"
        f"- Latencia y costo reales por empresa: {PENDING}.\n"
        "- Curva costo vs calidad por número de reflexiones: requiere la corrida real.\n"
    )
    return "\n".join(lines)

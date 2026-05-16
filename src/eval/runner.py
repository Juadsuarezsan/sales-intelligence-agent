"""Eval — runs the agent on a fixture company set; measures tool calls, iterations,
quality of profile, and personalization of email."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from src.agents.orchestrator import build_graph, research_company

COMPANIES = [
    "Anthropic", "Stripe", "Vercel", "Supabase", "Pinecone",
    "Vapi", "Modal Labs", "Replicate", "Cohere", "Mistral",
]


async def run_eval() -> dict:
    graph = build_graph()
    results = []
    for c in COMPANIES:
        out = await research_company(graph, company_name=c)
        results.append({
            "company": c,
            "iterations": out.iterations,
            "tool_calls": out.tool_calls,
            "profile_one_liner": out.profile.one_liner,
            "signals_found": len(out.profile.recent_signals),
            "sources_found": len(out.profile.sources),
            "email_personalization_score": out.email.personalization_score,
            "latency_ms": out.latency_ms,
        })
    n = len(results)
    return {
        "n": n,
        "avg_tool_calls": sum(r["tool_calls"] for r in results) / n,
        "avg_iterations": sum(r["iterations"] for r in results) / n,
        "avg_signals": sum(r["signals_found"] for r in results) / n,
        "avg_personalization": sum(r["email_personalization_score"] for r in results) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / n,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run_eval())
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"Companies: {report['n']}")
        print(f"Avg tool calls:        {report['avg_tool_calls']:.1f}")
        print(f"Avg iterations:        {report['avg_iterations']:.1f}")
        print(f"Avg signals found:     {report['avg_signals']:.1f}")
        print(f"Avg personalization:   {report['avg_personalization']:.2f}")
        print(f"Avg latency:           {report['avg_latency_ms']:.0f} ms")


if __name__ == "__main__":
    main()

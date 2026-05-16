# Project 03 — B2B Sales Intelligence Agent

> Planner-executor-reflector agent that researches target companies on the public web, builds structured profiles, and writes personalized outreach. The pattern behind Apollo.io, Clay.com, Outreach.io.

[![Status](https://img.shields.io/badge/status-planned-fbbf24)]()
[![LLM](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.5-7c5cff)]()
[![Pattern](https://img.shields.io/badge/pattern-Plan%2FExec%2FReflect-22d3ee)]()

**Industrial use case:** sales-tech (Apollo, Clay, Outreach). AI Engineers who can ship this are ultra-demanded.

## What this project does

Given a list of target companies, the system loops: plan what to research → execute searches (Tavily + HN + website fetch) → reflect on quality of info → refine. Produces a structured `CompanyProfile` and a personalized outreach email with measurable personalization.

## Architecture

```
For each target company:
   │
   ▼
[Planner] Claude → research plan
   │
   ▼
[Executor loop] LangGraph
   │ Tavily search · HN search · Website fetch
   │
   ▼
[Reflector] Claude → "enough info? contradictions?"
   │ if not enough → refine queries, loop
   │
   ▼
[Profile Builder] Claude + Pydantic → CompanyProfile
   │
   ▼
[Personalization Extractor] → pain points + recent signals
   │
   ▼
[Email Writer] Claude → subject, hook, value prop, CTA
   │
   ▼
[Quality Scorer] LLM-as-judge → personalization, accuracy, CTA clarity
```

## Roadmap to v1.0.0

1. [ ] Load YC companies dataset (~5K with metadata)
2. [ ] Tavily API integration with retries + rate limiting
3. [ ] selectolax-based website fetcher (faster than BeautifulSoup)
4. [ ] LangGraph plan-execute-reflect loop with configurable max iterations
5. [ ] Pydantic schemas for `CompanyProfile`, `OutreachEmail`
6. [ ] Quality scorer using Claude Opus as LLM-as-judge with explicit rubric
7. [ ] Eval set: 200 YC companies with ground truth metadata
8. [ ] Batch processing of 100 YC companies, results persisted
9. [ ] Next.js gallery view of the 100 pre-processed companies
10. [ ] LangSmith public trace gallery with ≥30 agent loops

## Stack

| Layer | Technology |
|---|---|
| LLM | Claude Sonnet 4.5 |
| Web search | Tavily API ($0.001/search, 1K free/mo) |
| News | HackerNews Algolia (free) |
| HTML parsing | httpx + selectolax |
| Structured outputs | Pydantic v2 |
| Orchestration | LangGraph (agent loop) |
| State | PostgreSQL |
| Frontend | Next.js 14 gallery view |
| Observability | LangSmith |

## Definition of Done — project-specific

- [ ] Batch processing of 100 YC companies completed with persistent outputs
- [ ] Demo gallery shows all 100 pre-processed companies
- [ ] Comparison against 2 baselines: template-only + single-search+email, with numbers
- [ ] LangSmith trace gallery with ≥30 agent loops linked
- [ ] Cost vs quality analysis documented: at what point does more reflection stop improving

Plus the 12 universal DoD blocks.

## License

MIT.

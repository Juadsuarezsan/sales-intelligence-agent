"""FastAPI app for B2B Sales Intelligence Agent."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv()

from src.agents.orchestrator import build_graph, research_company
from src.api.schemas import ResearchRequest, ResearchResponse
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_graph()
    yield


app = FastAPI(
    title="B2B Sales Intelligence Agent",
    version="0.5.0",
    description="Plan-execute-reflect agent over Tavily + HackerNews + Claude.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.5.0",
        "stage": "substantive",
        "tavily_enabled": "yes" if s.tavily_api_key else "no (stub)",
        "llm_enabled": "yes" if s.anthropic_api_key else "no (heuristic)",
    }


@app.post("/api/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    try:
        return await research_company(app.state.graph, company_name=req.company_name, context=req.seed_context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("research failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/eval/run")
async def run_eval_endpoint() -> dict:
    from src.eval.runner import run_eval
    return await run_eval()

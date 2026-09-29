"""Research result persistence.

The API and the batch runner talk to :class:`ResearchRepository`, a small
protocol with two real implementations:

* :class:`InMemoryRepository`: process-local dictionary with optional TTL,
  used by tests, the notebook and any deployment without ``DATABASE_URL``.
* :class:`PostgresRepository`: ``psycopg_pool.AsyncConnectionPool`` writing
  one JSONB row per company (upsert on the normalised company name).
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from loguru import logger

from src.api.schemas import ResearchResponse
from src.config import Settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS company_research (
    company_key   TEXT PRIMARY KEY,
    company_name  TEXT NOT NULL,
    trace_id      TEXT NOT NULL,
    mode          TEXT NOT NULL,
    cost_usd      DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_ms    INTEGER NOT NULL DEFAULT 0,
    payload       JSONB NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS company_research_updated_idx ON company_research (updated_at DESC);
"""

UPSERT_SQL = """
INSERT INTO company_research
    (company_key, company_name, trace_id, mode, cost_usd, latency_ms, payload, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, now())
ON CONFLICT (company_key) DO UPDATE SET
    company_name = EXCLUDED.company_name,
    trace_id     = EXCLUDED.trace_id,
    mode         = EXCLUDED.mode,
    cost_usd     = EXCLUDED.cost_usd,
    latency_ms   = EXCLUDED.latency_ms,
    payload      = EXCLUDED.payload,
    updated_at   = now();
"""

SELECT_ONE_SQL = "SELECT payload FROM company_research WHERE company_key = %s;"
SELECT_RECENT_SQL = "SELECT payload FROM company_research ORDER BY updated_at DESC LIMIT %s;"


def company_key(company_name: str) -> str:
    """Normalise a company name into the repository key.

    Args:
        company_name: Raw name.

    Returns:
        Lower-cased, whitespace-collapsed key.
    """
    return " ".join(company_name.lower().split())


class ResearchRepository(Protocol):
    """Storage for research results, keyed by company."""

    name: str

    async def save(self, result: ResearchResponse) -> None:
        """Insert or replace the result for ``result.company_name``."""
        ...

    async def get(self, company_name: str) -> ResearchResponse | None:
        """Return the stored result, or ``None``."""
        ...

    async def list_recent(self, limit: int = 100) -> list[ResearchResponse]:
        """Return the most recently updated results."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...


class InMemoryRepository:
    """Dictionary-backed repository with an optional TTL.

    Args:
        ttl_seconds: Entries older than this are ignored on read; ``0`` disables expiry.
    """

    name = "memory"

    def __init__(self, ttl_seconds: int = 0) -> None:
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, tuple[float, ResearchResponse]] = {}

    async def save(self, result: ResearchResponse) -> None:
        """Store ``result`` under its normalised company name."""
        self._items[company_key(result.company_name)] = (time.monotonic(), result)

    async def get(self, company_name: str) -> ResearchResponse | None:
        """Return a non-expired stored result, or ``None``."""
        entry = self._items.get(company_key(company_name))
        if entry is None:
            return None
        stored_at, result = entry
        if self.ttl_seconds and time.monotonic() - stored_at > self.ttl_seconds:
            del self._items[company_key(company_name)]
            return None
        return result

    async def list_recent(self, limit: int = 100) -> list[ResearchResponse]:
        """Return up to ``limit`` results, most recent first."""
        ordered = sorted(self._items.values(), key=lambda e: e[0], reverse=True)
        return [r for _, r in ordered[:limit]]

    async def close(self) -> None:
        """No-op."""
        return None

    def __len__(self) -> int:
        return len(self._items)


class PostgresRepository:
    """PostgreSQL repository backed by ``psycopg_pool.AsyncConnectionPool``.

    Args:
        conninfo: PostgreSQL DSN.
        min_size: Minimum pooled connections.
        max_size: Maximum pooled connections.
        pool: Pre-built pool (tests inject a fake); ``conninfo`` is ignored then.
    """

    name = "postgres"

    def __init__(
        self,
        conninfo: str = "",
        *,
        min_size: int = 1,
        max_size: int = 8,
        pool: Any | None = None,
    ) -> None:
        if pool is None:
            from psycopg_pool import AsyncConnectionPool

            pool = AsyncConnectionPool(
                conninfo, min_size=min_size, max_size=max_size, open=False, timeout=10.0
            )
        self._pool = pool

    async def open(self) -> None:
        """Open the pool and create the schema if needed."""
        await self._pool.open()
        async with self._pool.connection() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("postgres repository ready")

    async def save(self, result: ResearchResponse) -> None:
        """Upsert ``result`` as a JSONB payload."""
        async with self._pool.connection() as conn:
            await conn.execute(
                UPSERT_SQL,
                (
                    company_key(result.company_name),
                    result.company_name,
                    result.trace_id,
                    result.mode,
                    result.cost_usd,
                    result.latency_ms,
                    result.model_dump_json(),
                ),
            )

    async def get(self, company_name: str) -> ResearchResponse | None:
        """Fetch one result by company name."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(SELECT_ONE_SQL, (company_key(company_name),))
            row = await cur.fetchone()
        if row is None:
            return None
        return _to_response(row[0])

    async def list_recent(self, limit: int = 100) -> list[ResearchResponse]:
        """Fetch the ``limit`` most recently updated results."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(SELECT_RECENT_SQL, (limit,))
            rows = await cur.fetchall()
        return [_to_response(r[0]) for r in rows]

    async def close(self) -> None:
        """Close the pool."""
        await self._pool.close()


def _to_response(payload: Any) -> ResearchResponse:
    if isinstance(payload, str | bytes):
        payload = json.loads(payload)
    return ResearchResponse.model_validate(payload)


def build_repository(settings: Settings) -> ResearchRepository:
    """Choose the repository implementation from settings.

    Args:
        settings: Application settings.

    Returns:
        A :class:`PostgresRepository` when ``DATABASE_URL`` is set (call
        :meth:`PostgresRepository.open` before use), otherwise an
        :class:`InMemoryRepository` with the configured TTL.
    """
    if settings.database_url:
        return PostgresRepository(
            settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
        )
    return InMemoryRepository(ttl_seconds=settings.result_cache_ttl_seconds)

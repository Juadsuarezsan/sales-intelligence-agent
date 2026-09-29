from contextlib import asynccontextmanager
from typing import Any

import pytest

from src.api.schemas import CompanyProfile, OutreachEmail, ResearchResponse
from src.config import Settings
from src.storage import InMemoryRepository, PostgresRepository, build_repository
from src.storage.repository import SCHEMA_SQL, UPSERT_SQL, company_key


def _result(name: str, cost: float = 0.01) -> ResearchResponse:
    return ResearchResponse(
        trace_id="t",
        company_name=name,
        profile=CompanyProfile(name=name, one_liner="x"),
        email=OutreachEmail(subject="s", greeting="g", hook="h", value_prop="v", cta="c"),
        cost_usd=cost,
        latency_ms=10,
        mode="fallback",
    )


def test_company_key_normalises() -> None:
    assert company_key("  Acme   Robotics ") == "acme robotics"


async def test_in_memory_roundtrip_and_ordering() -> None:
    repo = InMemoryRepository()
    await repo.save(_result("A"))
    await repo.save(_result("B"))
    assert (await repo.get("a")) is not None
    assert [r.company_name for r in await repo.list_recent(1)] == ["B"]
    assert len(repo) == 2
    await repo.close()


async def test_in_memory_ttl_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = InMemoryRepository(ttl_seconds=10)
    await repo.save(_result("A"))
    clock = {"t": 1000.0}
    monkeypatch.setattr("src.storage.repository.time.monotonic", lambda: clock["t"])
    await repo.save(_result("B"))
    clock["t"] += 11
    assert await repo.get("B") is None
    assert len(repo) == 1


class FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows

    async def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class FakeConnection:
    def __init__(self, store: dict[str, str], executed: list[tuple[str, Any]]) -> None:
        self.store = store
        self.executed = executed

    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> FakeCursor:
        self.executed.append((sql, params))
        if sql == UPSERT_SQL and params is not None:
            self.store[params[0]] = params[6]
            return FakeCursor([])
        if "WHERE company_key" in sql and params is not None:
            payload = self.store.get(params[0])
            return FakeCursor([(payload,)] if payload else [])
        if "ORDER BY updated_at" in sql:
            return FakeCursor([(p,) for p in list(self.store.values())[::-1]])
        return FakeCursor([])


class FakePool:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.executed: list[tuple[str, Any]] = []
        self.opened = False
        self.closed = False

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.closed = True

    @asynccontextmanager
    async def connection(self) -> Any:
        yield FakeConnection(self.store, self.executed)


async def test_postgres_repository_upserts_and_reads_jsonb() -> None:
    pool = FakePool()
    repo = PostgresRepository(pool=pool)
    await repo.open()
    assert pool.opened and pool.executed[0][0] == SCHEMA_SQL
    await repo.save(_result("Acme"))
    assert pool.executed[-1][0] == UPSERT_SQL
    assert pool.executed[-1][1][0] == "acme"
    got = await repo.get("ACME")
    assert got is not None and got.company_name == "Acme" and got.cost_usd == 0.01
    assert await repo.get("missing") is None
    await repo.save(_result("Beta"))
    assert [r.company_name for r in await repo.list_recent(5)] == ["Beta", "Acme"]
    await repo.close()
    assert pool.closed


def test_build_repository_picks_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    assert build_repository(Settings()).name == "memory"
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    repo = build_repository(Settings())
    assert isinstance(repo, PostgresRepository) and repo.name == "postgres"

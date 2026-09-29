"""Persistence layer: repository protocol with in-memory and PostgreSQL backends."""

from src.storage.repository import (
    InMemoryRepository,
    PostgresRepository,
    ResearchRepository,
    build_repository,
)

__all__ = ["InMemoryRepository", "PostgresRepository", "ResearchRepository", "build_repository"]

from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    database_url: str = Field(
        default="postgresql://sales:sales_dev@localhost:5432/sales", alias="DATABASE_URL"
    )
    max_tool_calls_per_company: int = Field(default=8, alias="MAX_TOOL_CALLS_PER_COMPANY")
    max_reflection_iterations: int = Field(default=3, alias="MAX_REFLECTION_ITERATIONS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

"""Environment-driven configuration for the sales intelligence agent.

Every knob the system exposes lives here and is read from environment
variables (or a local ``.env`` file). Nothing else in ``src/`` reads
``os.environ`` directly, so the full configuration surface is visible in
one place and documented in ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Dated model ID the application is pinned to. Never use an undated alias.
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


class Settings(BaseSettings):
    """Application settings.

    Attributes:
        anthropic_api_key: Anthropic API key. ``None`` selects the deterministic
            offline fallback for every LLM node.
        anthropic_model: Dated Claude model ID used by the agent nodes.
        judge_model: Dated Claude model ID used by the LLM-as-judge scorer.
        llm_timeout_seconds: Per-request timeout for Anthropic calls.
        llm_max_attempts: Retry attempts (tenacity) for Anthropic calls.
        price_input_per_mtok: USD per million input tokens, used for ``cost_usd``.
        price_output_per_mtok: USD per million output tokens, used for ``cost_usd``.
        tavily_api_key: Tavily key. ``None`` selects the deterministic stub search.
        tavily_timeout_seconds: Timeout for each Tavily search.
        tavily_cost_per_search_usd: Cost attributed to each Tavily call.
        use_real_hn: Whether to call the HackerNews Algolia API or the stub.
        hn_timeout_seconds: Timeout for HackerNews calls.
        website_fetch_enabled: Whether the executor fetches the company website.
        website_timeout_seconds: Timeout for each website page fetch.
        max_tool_calls_per_company: Hard cap of tool calls per research run.
        max_reflection_iterations: Maximum plan/execute/reflect rounds.
        database_url: PostgreSQL DSN. ``None`` selects the in-memory repository.
        db_pool_min_size: Minimum connections in the psycopg pool.
        db_pool_max_size: Maximum connections in the psycopg pool.
        result_cache_ttl_seconds: TTL of the in-memory result cache.
        cors_origins: Allowed CORS origins (comma separated in the environment).
        rate_limit: slowapi rate limit expression for ``POST /api/research``.
        langchain_tracing_v2: Enables LangSmith tracing of the LangGraph run.
        langchain_api_key: LangSmith API key (only read when tracing is enabled).
        langchain_project: LangSmith project name.
        log_level: loguru log level.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default=DEFAULT_MODEL, alias="ANTHROPIC_MODEL")
    judge_model: str = Field(default=DEFAULT_MODEL, alias="JUDGE_MODEL")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS", gt=0)
    llm_max_attempts: int = Field(default=3, alias="LLM_MAX_ATTEMPTS", ge=1, le=10)
    price_input_per_mtok: float = Field(default=3.00, alias="PRICE_INPUT_PER_MTOK", ge=0)
    price_output_per_mtok: float = Field(default=15.00, alias="PRICE_OUTPUT_PER_MTOK", ge=0)

    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    tavily_timeout_seconds: float = Field(default=10.0, alias="TAVILY_TIMEOUT_SECONDS", gt=0)
    tavily_cost_per_search_usd: float = Field(default=0.001, alias="TAVILY_COST_PER_SEARCH_USD")

    use_real_hn: bool = Field(default=True, alias="USE_REAL_HN")
    hn_timeout_seconds: float = Field(default=8.0, alias="HN_TIMEOUT_SECONDS", gt=0)

    website_fetch_enabled: bool = Field(default=True, alias="WEBSITE_FETCH_ENABLED")
    website_timeout_seconds: float = Field(default=8.0, alias="WEBSITE_TIMEOUT_SECONDS", gt=0)

    max_tool_calls_per_company: int = Field(default=8, alias="MAX_TOOL_CALLS_PER_COMPANY", ge=1)
    max_reflection_iterations: int = Field(default=3, alias="MAX_REFLECTION_ITERATIONS", ge=0)

    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    db_pool_min_size: int = Field(default=1, alias="DB_POOL_MIN_SIZE", ge=1)
    db_pool_max_size: int = Field(default=8, alias="DB_POOL_MAX_SIZE", ge=1)
    result_cache_ttl_seconds: int = Field(default=3600, alias="RESULT_CACHE_TTL_SECONDS", ge=0)

    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "https://juadsuarezsan.github.io"],
        alias="CORS_ORIGINS",
    )
    rate_limit: str = Field(default="20/minute", alias="RATE_LIMIT")

    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="sales-intelligence-agent", alias="LANGCHAIN_PROJECT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string as well as a JSON list.

        Args:
            value: Raw value coming from the environment.

        Returns:
            A list of non-empty, stripped origins.
        """
        if isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

    @field_validator("anthropic_api_key", "tavily_api_key", "database_url", "langchain_api_key")
    @classmethod
    def _empty_is_none(cls, value: str | None) -> str | None:
        """Treat empty strings from ``.env`` as unset.

        Args:
            value: Raw string or ``None``.

        Returns:
            ``None`` for blank values, the stripped string otherwise.
        """
        if value is None or not value.strip():
            return None
        return value.strip()

    @property
    def llm_enabled(self) -> bool:
        """Whether real LLM calls are possible."""
        return self.anthropic_api_key is not None

    @property
    def tracing_enabled(self) -> bool:
        """Whether LangSmith tracing is fully configured."""
        return self.langchain_tracing_v2 and self.langchain_api_key is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance.

    Tests that mutate environment variables call ``get_settings.cache_clear()``.

    Returns:
        The cached settings object.
    """
    return Settings()

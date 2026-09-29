import pytest

from src.config import DEFAULT_MODEL, Settings, get_settings


def test_defaults_pin_dated_model_and_offline_fallbacks() -> None:
    s = get_settings()
    assert s.anthropic_model == DEFAULT_MODEL == "claude-sonnet-4-5-20250929"
    assert s.anthropic_api_key is None
    assert s.llm_enabled is False
    assert s.tracing_enabled is False
    assert s.database_url is None


def test_cors_origins_parse_from_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example ,*")
    assert Settings().cors_origin_list == ["https://a.example", "https://b.example"]


def test_blank_keys_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    s = Settings()
    assert s.anthropic_api_key is None
    assert s.tavily_api_key is None


def test_tracing_requires_flag_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    assert Settings().tracing_enabled is False
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-test")
    assert Settings().tracing_enabled is True

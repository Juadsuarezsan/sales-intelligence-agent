import httpx
import pytest
import respx
from anthropic import BadRequestError, InternalServerError

from src.config import Settings
from src.llm import AnthropicLLM, LLMOutputError, build_llm, parse_json_object
from src.observability import start_run

MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def _message(text: str, *, in_tok: int = 12, out_tok: int = 5) -> dict[str, object]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


@pytest.mark.parametrize(
    "text",
    [
        '{"queries": ["a"]}',
        '```json\n{"queries": ["a"]}\n```',
        'Sure, here it is:\n{"queries": ["a"]}\nHope this helps.',
    ],
)
def test_parse_json_object_handles_fences_and_prose(text: str) -> None:
    assert parse_json_object(text) == {"queries": ["a"]}


@pytest.mark.parametrize("text", ["no json here", "{not: valid}", "[1, 2, 3]", ""])
def test_parse_json_object_rejects_garbage(text: str) -> None:
    with pytest.raises(LLMOutputError):
        parse_json_object(text)


@respx.mock
async def test_anthropic_client_returns_text_and_records_usage() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=httpx.Response(200, json=_message("hi")))
    tracker = start_run()
    llm = AnthropicLLM("sk-test", "claude-sonnet-4-5-20250929", timeout_seconds=5)
    result = await llm.complete(system="s", user="u", max_tokens=50)
    assert result.text == "hi"
    assert (result.input_tokens, result.output_tokens) == (12, 5)
    assert (tracker.input_tokens, tracker.output_tokens, tracker.llm_calls) == (12, 5, 1)
    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == "sk-test"
    body = sent.read()
    assert b'"max_tokens":50' in body or b'"max_tokens": 50' in body


@respx.mock
async def test_anthropic_client_retries_transient_errors() -> None:
    route = respx.post(MESSAGES_URL).mock(
        side_effect=[
            httpx.Response(500, json={"error": {"type": "api_error", "message": "boom"}}),
            httpx.Response(200, json=_message("ok")),
        ]
    )
    llm = AnthropicLLM("sk-test", "m", timeout_seconds=5, backoff_seconds=(0, 0))
    result = await llm.complete(system="s", user="u")
    assert result.text == "ok"
    assert route.call_count == 2


@respx.mock
async def test_anthropic_client_gives_up_after_max_attempts() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(500, json={"error": {"type": "api_error", "message": "x"}})
    )
    llm = AnthropicLLM("sk-test", "m", timeout_seconds=5, max_attempts=2, backoff_seconds=(0, 0))
    with pytest.raises(InternalServerError):
        await llm.complete(system="s", user="u")


@respx.mock
async def test_anthropic_client_does_not_retry_bad_requests() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            400, json={"error": {"type": "invalid_request_error", "message": "bad"}}
        )
    )
    llm = AnthropicLLM("sk-test", "m", timeout_seconds=5, backoff_seconds=(0, 0))
    with pytest.raises(BadRequestError):
        await llm.complete(system="s", user="u")
    assert route.call_count == 1


def test_build_llm_returns_none_without_key() -> None:
    assert build_llm(Settings()) is None


def test_build_llm_uses_pinned_model_and_judge_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model-id")
    s = Settings()
    worker = build_llm(s)
    judge = build_llm(s, model=s.judge_model)
    assert isinstance(worker, AnthropicLLM) and worker.model == "claude-sonnet-4-5-20250929"
    assert isinstance(judge, AnthropicLLM) and judge.model == "judge-model-id"

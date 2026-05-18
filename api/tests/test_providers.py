from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.core.providers import complete


class _Schema(BaseModel):
    answer: str
    score: float


def _fake_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> MagicMock:
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


@pytest.mark.asyncio
async def test_complete_returns_content_cost_and_tokens() -> None:
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response("hello world")),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.001),
    ):
        result = await complete(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": "say hi"}],
        )

    assert result.content == "hello world"
    assert result.cost_usd == 0.001
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.cache_hit is False


@pytest.mark.asyncio
async def test_complete_cache_hit_skips_provider_call() -> None:
    fake = _fake_response("cached!")
    with (
        patch("app.core.providers.litellm.acompletion", AsyncMock(return_value=fake)) as mock_call,
        patch("app.core.providers.litellm.completion_cost", return_value=0.005),
    ):
        r1 = await complete(model="openai/gpt-5", messages=[{"role": "user", "content": "hi"}])
        r2 = await complete(model="openai/gpt-5", messages=[{"role": "user", "content": "hi"}])

    assert r1.cache_hit is False
    assert r2.cache_hit is True
    assert r2.content == "cached!"
    assert r2.cost_usd == 0.0  # cache hit incurs no new cost
    assert mock_call.call_count == 1  # provider only called once


@pytest.mark.asyncio
async def test_complete_parses_structured_output() -> None:
    payload = '{"answer": "yes", "score": 0.95}'
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(payload)),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.001),
    ):
        result = await complete(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": "ship?"}],
            response_format=_Schema,
        )

    assert result.parsed is not None
    assert isinstance(result.parsed, _Schema)
    assert result.parsed.answer == "yes"
    assert result.parsed.score == 0.95


@pytest.mark.asyncio
async def test_complete_cache_keys_include_schema() -> None:
    """Same prompt, different response_format → cache miss on second call."""
    fake = _fake_response('{"answer": "x", "score": 1.0}')
    with (
        patch("app.core.providers.litellm.acompletion", AsyncMock(return_value=fake)) as mock_call,
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        msgs = [{"role": "user", "content": "same"}]
        await complete(model="openai/gpt-5", messages=msgs)  # no schema
        await complete(model="openai/gpt-5", messages=msgs, response_format=_Schema)

    assert mock_call.call_count == 2


@pytest.mark.asyncio
async def test_complete_cache_keys_include_model() -> None:
    fake = _fake_response("x")
    with (
        patch("app.core.providers.litellm.acompletion", AsyncMock(return_value=fake)) as mock_call,
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        msgs = [{"role": "user", "content": "same"}]
        await complete(model="openai/gpt-5", messages=msgs)
        await complete(model="anthropic/claude-sonnet-4-6", messages=msgs)

    assert mock_call.call_count == 2


@pytest.mark.asyncio
async def test_complete_invalid_structured_output_returns_unparsed() -> None:
    """Bad JSON shouldn't crash — surface raw content with parsed=None."""
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response("not json")),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await complete(
            model="openai/gpt-5",
            messages=[{"role": "user", "content": "x"}],
            response_format=_Schema,
        )

    assert result.content == "not json"
    assert result.parsed is None

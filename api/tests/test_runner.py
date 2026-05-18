from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.runner import RunItemInput, run


def _fake_response(content: str) -> MagicMock:
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    resp = MagicMock(choices=[choice])
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    return resp


@pytest.mark.asyncio
async def test_run_renders_template_per_item() -> None:
    captured: list[str] = []

    async def fake_acompletion(**kwargs: object) -> MagicMock:
        msgs = kwargs.get("messages", [])  # type: ignore[union-attr]
        assert isinstance(msgs, list)
        captured.append(msgs[0]["content"])  # type: ignore[index]
        return _fake_response("ok")

    with (
        patch("app.core.providers.litellm.acompletion", side_effect=fake_acompletion),
        patch("app.core.providers.litellm.completion_cost", return_value=0.001),
    ):
        result = await run(
            prompt_template="classify: {{text}}",
            items=[
                RunItemInput(item_id="a", input_vars={"text": "hello"}),
                RunItemInput(item_id="b", input_vars={"text": "world"}),
            ],
            target_model="openai/gpt-5",
            max_concurrency=4,
        )

    assert sorted(captured) == ["classify: hello", "classify: world"]
    assert len(result.items) == 2
    assert all(r.actual_output == "ok" for r in result.items)
    assert result.total_cost_usd == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_run_preserves_input_order() -> None:
    """Even with concurrent dispatch, items are returned in input order."""
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response("x")),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await run(
            prompt_template="{{x}}",
            items=[RunItemInput(item_id=str(i), input_vars={"x": str(i)}) for i in range(20)],
            target_model="openai/gpt-5",
        )

    assert [r.item_id for r in result.items] == [str(i) for i in range(20)]


@pytest.mark.asyncio
async def test_run_captures_individual_errors_without_failing_run() -> None:
    call_count = {"n": 0}

    async def fake_acompletion(**_: object) -> MagicMock:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("api glitch")
        return _fake_response("ok")

    with (
        patch("app.core.providers.litellm.acompletion", side_effect=fake_acompletion),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await run(
            prompt_template="{{x}}",
            items=[
                RunItemInput(item_id="a", input_vars={"x": "1"}),
                RunItemInput(item_id="b", input_vars={"x": "2"}),
                RunItemInput(item_id="c", input_vars={"x": "3"}),
            ],
            target_model="openai/gpt-5",
            max_concurrency=1,  # serial to make the failure order deterministic
        )

    assert result.n_errors == 1
    failing = [r for r in result.items if r.error is not None]
    assert len(failing) == 1
    assert failing[0].item_id == "b"
    assert "api glitch" in (failing[0].error or "")


@pytest.mark.asyncio
async def test_run_invokes_progress_callback() -> None:
    seen: list[str] = []

    async def progress(item_result):  # type: ignore[no-untyped-def]
        seen.append(item_result.item_id)

    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response("ok")),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        await run(
            prompt_template="{{x}}",
            items=[RunItemInput(item_id=str(i), input_vars={"x": str(i)}) for i in range(3)],
            target_model="openai/gpt-5",
            progress_callback=progress,
        )

    assert sorted(seen) == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_run_empty_items_returns_empty_result() -> None:
    with patch("app.core.providers.litellm.acompletion") as mock_call:
        result = await run(prompt_template="{{x}}", items=[], target_model="openai/gpt-5")
    assert result.items == []
    assert result.total_cost_usd == 0.0
    assert mock_call.call_count == 0


@pytest.mark.asyncio
async def test_run_records_latency() -> None:
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response("ok")),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await run(
            prompt_template="{{x}}",
            items=[RunItemInput(item_id="a", input_vars={"x": "1"})],
            target_model="openai/gpt-5",
        )
    assert result.items[0].latency_ms >= 0

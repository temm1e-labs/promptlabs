import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.evalgen import EvalGenOutput, GeneratedEvalItem, generate
from app.schemas.common import PromptVariable, RubricCriterion


def _fake_response(content: str) -> MagicMock:
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    resp = MagicMock(choices=[choice])
    resp.usage.prompt_tokens = 200
    resp.usage.completion_tokens = 1000
    return resp


def _payload(items: list[dict], rubric: list[dict] | None = None) -> str:
    return json.dumps(
        {
            "rubric": rubric
            or [
                {
                    "name": "correctness",
                    "definition": "Output is correct for the input.",
                    "weight": 1.0,
                    "objective": "accuracy",
                }
            ],
            "items": items,
        }
    )


@pytest.mark.asyncio
async def test_generate_basic_split_and_rubric() -> None:
    items = [
        {
            "label": f"case {i}",
            "input_vars": {"email": f"text {i}"},
            "expected_output": None,
            "tags": ["common"],
        }
        for i in range(10)
    ]
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(_payload(items))),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.003),
    ):
        result = await generate(
            intent="classify emails",
            prompt_v0="Classify {{email}}",
            variables=[PromptVariable(name="email", description="email text", example_value="hi")],
            objectives=["accuracy"],
            known_issues=None,
            eval_size=10,
            train_ratio=0.7,
            model="openai/gpt-5",
        )

    assert len(result.train_items) == 7
    assert len(result.holdout_items) == 3
    assert isinstance(result.rubric[0], RubricCriterion)
    assert result.rubric[0].objective == "accuracy"
    assert result.cost_usd == 0.003


@pytest.mark.asyncio
async def test_generate_filters_items_with_wrong_variable_keys() -> None:
    items = [
        {"label": "good", "input_vars": {"email": "x"}, "expected_output": None, "tags": []},
        # missing var
        {"label": "missing", "input_vars": {}, "expected_output": None, "tags": []},
        # extra var
        {
            "label": "extra",
            "input_vars": {"email": "x", "ghost": "y"},
            "expected_output": None,
            "tags": [],
        },
        {"label": "good2", "input_vars": {"email": "y"}, "expected_output": None, "tags": []},
    ]
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(_payload(items))),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await generate(
            intent="x",
            prompt_v0="see {{email}}",
            variables=[PromptVariable(name="email", description="x", example_value="y")],
            objectives=["accuracy"],
            known_issues=None,
            eval_size=4,
            train_ratio=0.7,
            model="openai/gpt-5",
        )

    all_kept = result.train_items + result.holdout_items
    labels = [item.label for item in all_kept]
    assert labels == ["good", "good2"]


@pytest.mark.asyncio
async def test_generate_handles_malformed_response_gracefully() -> None:
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response("not json")),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await generate(
            intent="x",
            prompt_v0="see {{x}}",
            variables=[PromptVariable(name="x", description="x", example_value="y")],
            objectives=["accuracy"],
            known_issues=None,
            eval_size=10,
            train_ratio=0.7,
            model="openai/gpt-5",
        )

    assert result.train_items == []
    assert result.holdout_items == []
    # Degraded rubric still has at least correctness
    assert any(c.name == "correctness" for c in result.rubric)


@pytest.mark.asyncio
async def test_generate_with_known_issues_passes_them_through() -> None:
    """Smoke test: known_issues string must reach the LLM user message."""
    captured: dict[str, list] = {}

    async def fake_acompletion(**kwargs: object) -> MagicMock:
        captured["messages"] = kwargs.get("messages", [])  # type: ignore[assignment]
        return _fake_response(_payload([]))

    with (
        patch("app.core.providers.litellm.acompletion", side_effect=fake_acompletion),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        await generate(
            intent="x",
            prompt_v0="see {{x}}",
            variables=[PromptVariable(name="x", description="x", example_value="y")],
            objectives=["robustness"],
            known_issues="ENV_TOKEN_LEAK: model sometimes echoes secrets in output",
            eval_size=10,
            train_ratio=0.7,
            model="openai/gpt-5",
        )

    user_msg = captured["messages"][1]["content"]
    assert "ENV_TOKEN_LEAK" in user_msg


@pytest.mark.asyncio
async def test_generate_isolated_item_type() -> None:
    """Sanity: parsed items are GeneratedEvalItem instances."""
    items = [
        {"label": "x", "input_vars": {"q": "hi"}, "expected_output": "out", "tags": ["common"]}
    ]
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(_payload(items))),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await generate(
            intent="x",
            prompt_v0="{{q}}",
            variables=[PromptVariable(name="q", description="q", example_value="y")],
            objectives=[],
            known_issues=None,
            eval_size=1,
            train_ratio=0.7,
            model="openai/gpt-5",
        )

    all_items = result.train_items + result.holdout_items
    assert len(all_items) == 1
    assert isinstance(all_items[0], GeneratedEvalItem)
    assert all_items[0].expected_output == "out"


def test_evalgen_output_schema_roundtrip() -> None:
    """Ensure the Pydantic schema accepts well-formed JSON."""
    data = {
        "rubric": [{"name": "x", "definition": "y", "weight": 1.0, "objective": "accuracy"}],
        "items": [{"label": "t", "input_vars": {"a": "1"}, "expected_output": None, "tags": []}],
    }
    out = EvalGenOutput.model_validate(data)
    assert out.items[0].label == "t"

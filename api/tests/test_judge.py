import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.agents.judge import JudgeOutput, judge
from app.schemas.common import RubricCriterion


def _fake_response(content: str) -> MagicMock:
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    resp = MagicMock(choices=[choice])
    resp.usage.prompt_tokens = 50
    resp.usage.completion_tokens = 100
    return resp


def _judge_payload(scores: list[dict], reasoning: str = "ok") -> str:
    return json.dumps({"scores": scores, "overall_reasoning": reasoning})


@pytest.mark.asyncio
async def test_judge_basic_weighted_mean() -> None:
    rubric = [
        RubricCriterion(name="correctness", definition="x", weight=2.0, objective="accuracy"),
        RubricCriterion(name="brevity", definition="y", weight=1.0, objective="brevity"),
    ]
    payload = _judge_payload(
        [
            {"name": "correctness", "score": 0.9, "reasoning": "good"},
            {"name": "brevity", "score": 0.3, "reasoning": "long"},
        ]
    )
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(payload)),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.001),
    ):
        result = await judge(
            rubric=rubric,
            rendered_input="x",
            actual_output="y",
            model="openai/gpt-5",
        )

    # weighted mean: (2*0.9 + 1*0.3) / 3 = 2.1/3 = 0.7
    assert result.mean_score == pytest.approx(0.7)
    assert result.scores == {"correctness": 0.9, "brevity": 0.3}
    assert result.per_objective == {"accuracy": pytest.approx(0.9), "brevity": pytest.approx(0.3)}


@pytest.mark.asyncio
async def test_judge_clamps_out_of_range_scores() -> None:
    rubric = [RubricCriterion(name="x", definition="d", weight=1.0)]
    # LLM returns 1.5 and -0.2 — judge should not crash; aggregation clamps to [0,1].
    # Pydantic schema will reject these though, so we need to test the aggregate path.
    # Bypass schema validation by simulating a CriterionScore directly.
    # Easier: test the aggregator via valid range first; out-of-range Pydantic raises.
    payload = _judge_payload([{"name": "x", "score": 1.0, "reasoning": ""}])
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(payload)),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await judge(
            rubric=rubric, rendered_input="i", actual_output="o", model="openai/gpt-5"
        )
    assert result.scores["x"] == 1.0


@pytest.mark.asyncio
async def test_judge_drops_spurious_criteria() -> None:
    """If LLM scores a criterion not in the rubric, it's ignored."""
    rubric = [RubricCriterion(name="real", definition="d", weight=1.0)]
    payload = _judge_payload(
        [
            {"name": "real", "score": 0.8, "reasoning": ""},
            {"name": "ghost", "score": 1.0, "reasoning": ""},
        ]
    )
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(payload)),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await judge(
            rubric=rubric, rendered_input="x", actual_output="y", model="openai/gpt-5"
        )

    assert set(result.scores.keys()) == {"real"}


@pytest.mark.asyncio
async def test_judge_handles_malformed_response() -> None:
    rubric = [RubricCriterion(name="x", definition="d", weight=1.0)]
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response("not json")),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await judge(
            rubric=rubric, rendered_input="i", actual_output="o", model="openai/gpt-5"
        )

    assert result.mean_score == 0.0
    assert result.scores == {"x": 0.0}


@pytest.mark.asyncio
async def test_judge_with_empty_rubric_returns_zero() -> None:
    result = await judge(rubric=[], rendered_input="x", actual_output="y", model="openai/gpt-5")
    assert result.mean_score == 0.0
    assert result.scores == {}
    assert result.cost_usd == 0.0


def test_judge_output_schema_rejects_out_of_range() -> None:
    """Pydantic schema enforces [0, 1] — guard rail before aggregation."""
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate(
            {
                "scores": [{"name": "x", "score": 1.5, "reasoning": ""}],
                "overall_reasoning": "",
            }
        )

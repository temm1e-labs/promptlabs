import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.evalgen import (
    EvalGenOutput,
    GeneratedEvalItem,
    SINGLE_CALL_MAX,
    _dedup_items,
    _normalize_categories,
    _normalize_for_dedup,
    EvalCategory,
    generate,
)
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


# ─── batched path tests ────────────────────────────────────────────────


def _taxonomy_payload(n_categories: int = 5, target_per_cat: int = 10) -> str:
    """Build a TaxonomyOutput-shaped JSON response."""
    return json.dumps(
        {
            "rubric": [
                {
                    "name": "correctness",
                    "definition": "Output is correct.",
                    "weight": 1.0,
                    "objective": "accuracy",
                }
            ],
            "categories": [
                {
                    "name": f"cat-{i}",
                    "description": f"Category {i} description.",
                    "target_tag": "common" if i < 3 else "edge",
                    "target_count": target_per_cat,
                }
                for i in range(n_categories)
            ],
        }
    )


def _batch_payload(items: list[dict]) -> str:
    return json.dumps({"items": items})


def test_normalize_for_dedup_collapses_whitespace_and_case() -> None:
    a = GeneratedEvalItem(label="A", input_vars={"q": "Hello   World"}, tags=[])
    b = GeneratedEvalItem(label="B", input_vars={"q": "hello world"}, tags=[])
    c = GeneratedEvalItem(label="C", input_vars={"q": "Different"}, tags=[])
    assert _normalize_for_dedup(a) == _normalize_for_dedup(b)
    assert _normalize_for_dedup(a) != _normalize_for_dedup(c)


def test_dedup_items_drops_duplicates_keeps_first() -> None:
    items = [
        GeneratedEvalItem(label="first", input_vars={"q": "hello"}, tags=[]),
        GeneratedEvalItem(label="dup", input_vars={"q": "HELLO"}, tags=[]),
        GeneratedEvalItem(label="unique", input_vars={"q": "different"}, tags=[]),
    ]
    out = _dedup_items(items)
    assert [it.label for it in out] == ["first", "unique"]


def test_normalize_categories_rescales_to_target() -> None:
    cats = [
        EvalCategory(
            name=f"c{i}",
            description="d",
            target_tag="common",
            target_count=2,  # sums to 10
        )
        for i in range(5)
    ]
    out = _normalize_categories(cats, eval_size=40)
    total = sum(c.target_count for c in out)
    # Rescaled to ~115% of eval_size so dedup losses are absorbed
    assert total >= 40
    assert total <= 60  # generous upper bound for rounding


@pytest.mark.asyncio
async def test_generate_batched_uses_taxonomy_and_parallel_batches() -> None:
    """eval_size > SINGLE_CALL_MAX triggers the batched path with taxonomy + batches."""
    assert SINGLE_CALL_MAX < 30  # sanity — test relies on 30 being above the threshold

    # Sequenced fake responses: taxonomy first, then one CategoryBatchOutput per category
    n_categories = 5
    taxonomy_response = _fake_response(_taxonomy_payload(n_categories=n_categories))
    batch_responses = [
        _fake_response(
            _batch_payload(
                [
                    {
                        "label": f"cat{c}-item{i}",
                        "input_vars": {"email": f"text c{c} i{i}"},
                        "expected_output": None,
                        "tags": [],
                    }
                    for i in range(8)
                ]
            )
        )
        for c in range(n_categories)
    ]
    responses = [taxonomy_response, *batch_responses]
    call_log: list[str] = []

    async def fake_acompletion(**kwargs: object) -> MagicMock:
        system_msg = kwargs["messages"][0]["content"]  # type: ignore[index]
        if "taxonomy" in system_msg.lower() or "stratification" in system_msg.lower():
            call_log.append("taxonomy")
        else:
            call_log.append("batch")
        return responses.pop(0)

    with (
        patch("app.core.providers.litellm.acompletion", side_effect=fake_acompletion),
        patch("app.core.providers.litellm.completion_cost", return_value=0.001),
    ):
        result = await generate(
            intent="classify emails",
            prompt_v0="Classify {{email}}",
            variables=[PromptVariable(name="email", description="email", example_value="x")],
            objectives=["accuracy"],
            known_issues=None,
            eval_size=30,
            train_ratio=0.7,
            model="openai/gpt-5",
        )

    # 1 taxonomy call + n_categories batch calls
    assert call_log[0] == "taxonomy"
    assert call_log.count("batch") == n_categories
    # Should have ~30 items after dedup + trim (5 categories × 8 items = 40, trimmed to 30)
    total = len(result.train_items) + len(result.holdout_items)
    assert total == 30
    # Cost = (1 + n_categories) * 0.001
    assert result.cost_usd == pytest.approx((1 + n_categories) * 0.001)


@pytest.mark.asyncio
async def test_generate_batched_dedup_across_batches() -> None:
    """Identical items emitted by two batches collapse to one."""
    n_categories = 3
    taxonomy_response = _fake_response(_taxonomy_payload(n_categories=n_categories))

    # Each batch emits the SAME item — dedup should leave only one
    duplicate_item = {
        "label": "the-only-one",
        "input_vars": {"email": "same text everywhere"},
        "expected_output": None,
        "tags": [],
    }
    batch_responses = [
        _fake_response(_batch_payload([duplicate_item])) for _ in range(n_categories)
    ]

    # Top-up call: returns a few new distinct items so we have enough to split
    topup_response = _fake_response(
        _batch_payload(
            [
                {
                    "label": f"topup-{i}",
                    "input_vars": {"email": f"topup text {i}"},
                    "expected_output": None,
                    "tags": [],
                }
                for i in range(20)
            ]
        )
    )
    responses = [taxonomy_response, *batch_responses, topup_response]

    async def fake_acompletion(**kwargs: object) -> MagicMock:
        return responses.pop(0)

    with (
        patch("app.core.providers.litellm.acompletion", side_effect=fake_acompletion),
        patch("app.core.providers.litellm.completion_cost", return_value=0.001),
    ):
        result = await generate(
            intent="classify",
            prompt_v0="{{email}}",
            variables=[PromptVariable(name="email", description="x", example_value="y")],
            objectives=["accuracy"],
            known_issues=None,
            eval_size=20,
            train_ratio=0.7,
            model="openai/gpt-5",
        )

    all_items = result.train_items + result.holdout_items
    labels = [item.label for item in all_items]
    # Exactly one "duplicate-item" survived the dedup pass
    assert labels.count("the-only-one") == 1


@pytest.mark.asyncio
async def test_generate_batched_emits_progress_events() -> None:
    """on_progress is invoked at taxonomy + each batch milestone."""
    n_categories = 3
    taxonomy_response = _fake_response(_taxonomy_payload(n_categories=n_categories))
    batch_responses = [
        _fake_response(
            _batch_payload(
                [
                    {
                        "label": f"c{c}-i{i}",
                        "input_vars": {"q": f"text {c} {i}"},
                        "expected_output": None,
                        "tags": [],
                    }
                    for i in range(10)
                ]
            )
        )
        for c in range(n_categories)
    ]
    responses = [taxonomy_response, *batch_responses]

    async def fake_acompletion(**_kwargs: object) -> MagicMock:
        return responses.pop(0)

    events: list[tuple[str, dict]] = []

    async def on_progress(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    with (
        patch("app.core.providers.litellm.acompletion", side_effect=fake_acompletion),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        await generate(
            intent="x",
            prompt_v0="{{q}}",
            variables=[PromptVariable(name="q", description="x", example_value="y")],
            objectives=["accuracy"],
            known_issues=None,
            eval_size=20,
            train_ratio=0.7,
            model="openai/gpt-5",
            on_progress=on_progress,
        )

    types = [t for t, _ in events]
    assert "evalgen.taxonomy_completed" in types
    assert types.count("evalgen.batch_completed") == n_categories
    # First batch event carries the category name
    batch_events = [d for t, d in events if t == "evalgen.batch_completed"]
    assert {ev["category"] for ev in batch_events} == {"cat-0", "cat-1", "cat-2"}


@pytest.mark.asyncio
async def test_generate_batched_falls_back_when_taxonomy_fails() -> None:
    """If taxonomy parse fails, fall back to single-call path (resilient degradation)."""
    bad_taxonomy = _fake_response("not json")
    # Single-call fallback uses EvalGenOutput schema
    fallback_items = _fake_response(
        _payload(
            [
                {
                    "label": f"item-{i}",
                    "input_vars": {"q": f"fallback text {i}"},
                    "expected_output": None,
                    "tags": [],
                }
                for i in range(20)
            ]
        )
    )
    responses = [bad_taxonomy, fallback_items]

    async def fake_acompletion(**_kwargs: object) -> MagicMock:
        return responses.pop(0)

    with (
        patch("app.core.providers.litellm.acompletion", side_effect=fake_acompletion),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await generate(
            intent="x",
            prompt_v0="{{q}}",
            variables=[PromptVariable(name="q", description="x", example_value="y")],
            objectives=["accuracy"],
            known_issues=None,
            eval_size=20,
            train_ratio=0.7,
            model="openai/gpt-5",
        )

    total = len(result.train_items) + len(result.holdout_items)
    assert total == 20  # fallback succeeded


@pytest.mark.asyncio
async def test_generate_single_path_used_below_threshold() -> None:
    """eval_size <= SINGLE_CALL_MAX should make exactly one LLM call (no taxonomy)."""
    call_count = 0

    async def fake_acompletion(**_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return _fake_response(
            _payload(
                [
                    {
                        "label": f"x{i}",
                        "input_vars": {"q": f"t{i}"},
                        "expected_output": None,
                        "tags": [],
                    }
                    for i in range(SINGLE_CALL_MAX)
                ]
            )
        )

    with (
        patch("app.core.providers.litellm.acompletion", side_effect=fake_acompletion),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        await generate(
            intent="x",
            prompt_v0="{{q}}",
            variables=[PromptVariable(name="q", description="x", example_value="y")],
            objectives=["accuracy"],
            known_issues=None,
            eval_size=SINGLE_CALL_MAX,
            train_ratio=0.7,
            model="openai/gpt-5",
        )

    assert call_count == 1

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.optimizer import (
    Edit,
    OptimizerOutput,
    _enforce_budgets,
    _validate_variables_preserved,
    apply_diff,
    optimize,
)

# ─── apply_diff unit tests ──────────────────────────────────────────────────


class TestApplyDiff:
    def test_replace_unique_anchor(self) -> None:
        edits = [Edit(op="replace", anchor="hello", new_text="hi", reason="r")]
        new, fail = apply_diff("say hello there", edits)
        assert new == "say hi there"
        assert fail == []

    def test_replace_missing_anchor_skips_edit(self) -> None:
        edits = [Edit(op="replace", anchor="missing", new_text="x", reason="r")]
        new, fail = apply_diff("hello", edits)
        assert new == "hello"
        assert len(fail) == 1
        assert "anchor not found" in fail[0]

    def test_replace_ambiguous_anchor_skips(self) -> None:
        edits = [Edit(op="replace", anchor="x", new_text="y", reason="r")]
        new, fail = apply_diff("x x x", edits)
        assert new == "x x x"
        assert "ambiguous" in fail[0]

    def test_delete(self) -> None:
        edits = [Edit(op="delete", anchor=" please", reason="r")]
        new, _ = apply_diff("respond please nicely", edits)
        assert new == "respond nicely"

    def test_insert_before(self) -> None:
        edits = [Edit(op="insert_before", anchor="reply:", new_text="Be concise. ", reason="r")]
        new, _ = apply_diff("reply:", edits)
        assert new == "Be concise. reply:"

    def test_insert_after(self) -> None:
        edits = [Edit(op="insert_after", anchor="reply:", new_text=" stay on topic.", reason="r")]
        new, _ = apply_diff("reply:", edits)
        assert new == "reply: stay on topic."

    def test_append(self) -> None:
        edits = [Edit(op="append", new_text="\n\nDo not hallucinate.", reason="r")]
        new, _ = apply_diff("prompt", edits)
        assert new == "prompt\n\nDo not hallucinate."

    def test_multiple_edits_applied_in_order(self) -> None:
        edits = [
            Edit(op="replace", anchor="A", new_text="X", reason=""),
            Edit(op="append", new_text="Z", reason=""),
        ]
        new, fail = apply_diff("A", edits)
        assert new == "XZ"
        assert fail == []

    def test_one_failing_edit_doesnt_block_others(self) -> None:
        edits = [
            Edit(op="replace", anchor="MISSING", new_text="x", reason=""),
            Edit(op="append", new_text=" done", reason=""),
        ]
        new, fail = apply_diff("hello", edits)
        assert new == "hello done"
        assert len(fail) == 1


# ─── budget enforcement ──────────────────────────────────────────────────


class TestBudgets:
    def test_max_edits_truncates(self) -> None:
        edits = [Edit(op="append", new_text="x", reason="") for _ in range(10)]
        kept, drops = _enforce_budgets(
            edits, max_edits=3, max_chars_changed_ratio=1.0, prompt_len=100
        )
        assert len(kept) == 3
        assert any("max_edits=3" in d for d in drops)

    def test_chars_changed_ratio_caps(self) -> None:
        big_edit = Edit(op="append", new_text="x" * 500, reason="")
        kept, drops = _enforce_budgets(
            [big_edit], max_edits=10, max_chars_changed_ratio=0.1, prompt_len=1000
        )
        assert kept == []
        assert any("budget" in d for d in drops)


# ─── variable preservation ──────────────────────────────────────────────────


class TestVariablePreservation:
    def test_no_change_ok(self) -> None:
        ok, added, _removed = _validate_variables_preserved(
            "use {{x}}", "use {{x}} more thoroughly"
        )
        assert ok
        assert added == set()

    def test_added_var_flagged(self) -> None:
        ok, added, _ = _validate_variables_preserved("hi {{x}}", "hi {{x}} and {{y}}")
        assert not ok
        assert added == {"y"}

    def test_removed_var_ok(self) -> None:
        """Removing an unused var is allowed (eval items still work)."""
        ok, _added, removed = _validate_variables_preserved("{{a}} {{b}}", "just {{a}}")
        assert ok
        assert removed == {"b"}


# ─── end-to-end optimize() ──────────────────────────────────────────────────


def _fake_response(content: str) -> MagicMock:
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    resp = MagicMock(choices=[choice])
    resp.usage.prompt_tokens = 200
    resp.usage.completion_tokens = 300
    return resp


def _opt_payload(edits: list[dict], summary: str = "patched") -> str:
    return json.dumps({"edits": edits, "summary": summary})


@pytest.mark.asyncio
async def test_optimize_applies_edits_and_returns_new_prompt() -> None:
    edits = [
        {
            "op": "append",
            "new_text": "\n\nBe concise.",
            "reason": "fix verbosity",
            "targets_criterion": "brevity",
        }
    ]
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(_opt_payload(edits))),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.002),
    ):
        result = await optimize(
            current_prompt="Reply to {{q}}.",
            iteration=1,
            rubric=[{"name": "brevity", "definition": "short", "weight": 1.0}],
            failure_samples=[],
            objectives=["brevity"],
            known_issues=None,
            model="openai/gpt-5",
        )

    assert result.new_prompt == "Reply to {{q}}.\n\nBe concise."
    assert result.edits_applied == 1
    assert result.edits_skipped == 0


@pytest.mark.asyncio
async def test_optimize_reverts_when_new_variables_introduced() -> None:
    """Safety net: optimizer can't sneak in a new {{var}}."""
    edits = [
        {
            "op": "replace",
            "anchor": "Reply",
            "new_text": "Reply about {{new_var}}",
            "reason": "x",
        }
    ]
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response(_opt_payload(edits))),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await optimize(
            current_prompt="Reply to {{q}}.",
            iteration=1,
            rubric=[],
            failure_samples=[],
            objectives=["accuracy"],
            known_issues=None,
            model="openai/gpt-5",
        )

    assert result.new_prompt == "Reply to {{q}}."  # reverted
    assert any("introduced new variables" in r for r in result.skip_reasons)


@pytest.mark.asyncio
async def test_optimize_handles_malformed_response_no_op() -> None:
    with (
        patch(
            "app.core.providers.litellm.acompletion",
            AsyncMock(return_value=_fake_response("not json")),
        ),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        result = await optimize(
            current_prompt="prompt",
            iteration=1,
            rubric=[],
            failure_samples=[],
            objectives=[],
            known_issues=None,
            model="openai/gpt-5",
        )

    assert result.new_prompt == "prompt"
    assert result.edits_applied == 0


@pytest.mark.asyncio
async def test_optimize_passes_known_issues_on_iteration_1() -> None:
    captured: list = []

    async def fake(**kwargs: object) -> MagicMock:
        captured.append(kwargs.get("messages"))
        return _fake_response(_opt_payload([]))

    with (
        patch("app.core.providers.litellm.acompletion", side_effect=fake),
        patch("app.core.providers.litellm.completion_cost", return_value=0.0),
    ):
        await optimize(
            current_prompt="x",
            iteration=1,
            rubric=[],
            failure_samples=[],
            objectives=[],
            known_issues="LEAK_VAR",
            model="openai/gpt-5",
        )
        await optimize(
            current_prompt="x",
            iteration=2,
            rubric=[],
            failure_samples=[],
            objectives=[],
            known_issues="LEAK_VAR",
            model="openai/gpt-5",
        )

    iter1_msg = captured[0][1]["content"]
    iter2_msg = captured[1][1]["content"]
    assert "LEAK_VAR" in iter1_msg
    assert "LEAK_VAR" not in iter2_msg  # only fed in on iter-1


def test_optimizer_output_schema_roundtrip() -> None:
    out = OptimizerOutput.model_validate(
        {"edits": [{"op": "append", "new_text": "x", "reason": "r"}], "summary": "s"}
    )
    assert out.edits[0].op == "append"

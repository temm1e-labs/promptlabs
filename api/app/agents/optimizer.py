"""Optimizer agent — produces a surgical diff against the current prompt.

Design constraints (the differentiator vs. PromptWizard / DSPy / TextGrad):
  * Output is a STRUCTURED PATCH, not a whole-prompt regeneration. Users see WHAT changed
    and WHY, tied to which failing criterion the edit addresses.
  * Edits are string-anchored, not line-numbered (robust to whitespace/wrapping).
  * The Optimizer never invents new {{variables}}: that would orphan the eval items.
  * A hard cap on the number of edits prevents accidental whole-prompt rewrites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core import providers, template
from app.core.logging import log

EditOp = Literal["replace", "insert_before", "insert_after", "delete", "append"]


class Edit(BaseModel):
    op: EditOp = Field(description="Edit operation.")
    anchor: str | None = Field(
        default=None,
        description=(
            "The existing substring to find. Required for replace/insert_before/insert_after/"
            "delete. Must match exactly (including whitespace). Null for `append`."
        ),
    )
    new_text: str | None = Field(
        default=None,
        description=("Replacement text (for replace) or text to insert/append. Null for `delete`."),
    )
    reason: str = Field(
        description="Why this edit. Reference the failing criterion when applicable."
    )
    targets_criterion: str | None = Field(
        default=None,
        description="Name of a rubric criterion this edit primarily targets.",
    )


class OptimizerOutput(BaseModel):
    edits: list[Edit] = Field(default_factory=list)
    summary: str = Field(
        description="One paragraph describing the strategy: what failed and how the edits fix it."
    )


@dataclass
class OptimizerResult:
    new_prompt: str
    edits: list[Edit]
    summary: str
    edits_applied: int
    edits_skipped: int
    skip_reasons: list[str]
    cost_usd: float
    cache_hit: bool


class EditApplicationError(ValueError):
    """Raised internally when a single edit fails (e.g., anchor not found)."""


def _apply_one(current: str, edit: Edit) -> str:
    if edit.op == "append":
        if edit.new_text is None:
            raise EditApplicationError("append requires new_text")
        return current + edit.new_text
    if edit.anchor is None:
        raise EditApplicationError(f"{edit.op} requires anchor")
    if edit.anchor not in current:
        raise EditApplicationError("anchor not found")
    occurrences = current.count(edit.anchor)
    if occurrences > 1:
        raise EditApplicationError(f"anchor ambiguous ({occurrences} matches)")

    if edit.op == "replace":
        if edit.new_text is None:
            raise EditApplicationError("replace requires new_text")
        return current.replace(edit.anchor, edit.new_text, 1)
    if edit.op == "delete":
        return current.replace(edit.anchor, "", 1)
    if edit.op == "insert_before":
        if edit.new_text is None:
            raise EditApplicationError("insert_before requires new_text")
        return current.replace(edit.anchor, edit.new_text + edit.anchor, 1)
    if edit.op == "insert_after":
        if edit.new_text is None:
            raise EditApplicationError("insert_after requires new_text")
        return current.replace(edit.anchor, edit.anchor + edit.new_text, 1)
    raise EditApplicationError(f"unknown op: {edit.op}")


def apply_diff(prompt: str, edits: list[Edit]) -> tuple[str, list[str]]:
    """Apply edits in order. Returns (new_prompt, list of per-edit skip reasons).

    Each edit failure is recorded; remaining edits continue against the partially-edited prompt.
    """
    current = prompt
    failures: list[str] = []
    for i, edit in enumerate(edits):
        try:
            current = _apply_one(current, edit)
        except EditApplicationError as exc:
            failures.append(f"edit[{i}] {edit.op}: {exc}")
    return current, failures


def _enforce_budgets(
    edits: list[Edit],
    max_edits: int,
    max_chars_changed_ratio: float,
    prompt_len: int,
) -> tuple[list[Edit], list[str]]:
    """Drop edits beyond budget. Returns (kept, drop reasons)."""
    drops: list[str] = []
    if len(edits) > max_edits:
        kept = edits[:max_edits]
        for i in range(max_edits, len(edits)):
            drops.append(f"edit[{i}] dropped: max_edits={max_edits} exceeded")
        edits = kept

    # Bytes-changed estimate: sum of |anchor| + |new_text| per edit.
    # Floor at 100 chars so short prompts still admit meaningful edits.
    budget = max(100, int(prompt_len * max_chars_changed_ratio))
    spent = 0
    kept2: list[Edit] = []
    for i, e in enumerate(edits):
        cost = len(e.anchor or "") + len(e.new_text or "")
        if spent + cost > budget:
            drops.append(f"edit[{i}] dropped: chars-changed budget exceeded")
            continue
        spent += cost
        kept2.append(e)
    return kept2, drops


def _validate_variables_preserved(
    original: str,
    new: str,
) -> tuple[bool, set[str], set[str]]:
    """Returns (ok, added_vars, removed_vars). ok = no NEW vars introduced."""
    old_vars = set(template.extract_variables(original))
    new_vars = set(template.extract_variables(new))
    added = new_vars - old_vars
    removed = old_vars - new_vars
    return (not added), added, removed


_SYSTEM = """You are a senior prompt engineer performing a SURGICAL edit pass. The user
has an existing prompt that's been evaluated; some criteria failed. Your job is to fix it
WITHOUT rewriting the whole thing.

Hard rules:
1. Output STRUCTURED EDITS, not a rewritten prompt. Each edit specifies a precise anchor in
   the current prompt and a small replacement / insertion / deletion.
2. Edits must be MINIMAL and TARGETED. Touch only what needs to change to address a
   specific failure mode. No cleanup, no stylistic noise, no "while I'm here" changes.
3. Each edit's `reason` MUST reference the failure mode it addresses. When possible, set
   `targets_criterion` to the name of the rubric criterion that this edit improves.
4. PRESERVE every {{variable_name}} placeholder. NEVER introduce new variables — the eval
   set doesn't have values for them. You CAN remove an unused one if confident.
5. Anchors must MATCH THE PROMPT EXACTLY (including whitespace and casing) AND BE UNIQUE
   (one occurrence). If a string appears multiple times, anchor on a longer surrounding
   substring that's unique.
6. Output 1-5 edits MAX in a single pass. Fewer is better. The loop runs many passes.
7. Output strict JSON matching the schema."""


def _failure_block(failures: list[dict[str, Any]]) -> str:
    """Render failure samples for the LLM. `failures` is a list of dicts produced upstream."""
    if not failures:
        return "(no failures sampled)"
    lines = []
    for f in failures[:10]:  # cap context
        lines.append(
            f"- item={f.get('label', '?')!r}  "
            f"score={f.get('mean_score', 0):.2f}  "
            f"failing_criteria={f.get('failing_criteria', [])}  "
            f"input={str(f.get('input_vars'))[:200]}  "
            f"output={str(f.get('actual_output'))[:200]}  "
            f"why={str(f.get('reasoning'))[:200]}"
        )
    return "\n".join(lines)


def _rubric_block(rubric: list[dict[str, Any]]) -> str:
    return (
        "\n".join(
            f"- {r.get('name')}: {r.get('definition')} [weight {r.get('weight', 1.0)}, "
            f"objective: {r.get('objective')}]"
            for r in rubric
        )
        or "(empty)"
    )


async def optimize(
    *,
    current_prompt: str,
    iteration: int,
    rubric: list[dict[str, Any]],
    failure_samples: list[dict[str, Any]],
    objectives: list[str],
    known_issues: str | None,
    model: str,
    max_edits: int = 5,
    max_chars_changed_ratio: float = 0.35,
) -> OptimizerResult:
    extra_context = ""
    if iteration == 1 and known_issues:
        extra_context = (
            f"\nUser-reported issues (PRIORITIZE these on the first pass):\n{known_issues}\n"
        )

    user_msg = (
        f'Current prompt (v{iteration - 1}):\n"""\n{current_prompt}\n"""\n\n'
        f"Rubric:\n{_rubric_block(rubric)}\n\n"
        f"Objectives the loop optimizes for: {', '.join(objectives) or 'accuracy'}\n\n"
        f"Failing/low-scoring eval samples:\n{_failure_block(failure_samples)}\n"
        f"{extra_context}\n"
        "Produce a small, surgical patch. Output strict JSON."
    )

    result = await providers.complete(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format=OptimizerOutput,
        temperature=0.4,
    )

    if not isinstance(result.parsed, OptimizerOutput):
        log.warning("optimizer.parse_failed_no_op")
        return OptimizerResult(
            new_prompt=current_prompt,
            edits=[],
            summary="(optimizer parse failed — no edits applied)",
            edits_applied=0,
            edits_skipped=0,
            skip_reasons=["optimizer LLM returned malformed JSON"],
            cost_usd=result.cost_usd,
            cache_hit=result.cache_hit,
        )

    edits, drop_reasons = _enforce_budgets(
        result.parsed.edits,
        max_edits=max_edits,
        max_chars_changed_ratio=max_chars_changed_ratio,
        prompt_len=max(1, len(current_prompt)),
    )

    new_prompt, apply_failures = apply_diff(current_prompt, edits)
    skip_reasons = drop_reasons + apply_failures

    vars_ok, added_vars, _removed = _validate_variables_preserved(current_prompt, new_prompt)
    if not vars_ok:
        log.warning(
            "optimizer.introduced_new_variables_reverting",
            added=sorted(added_vars),
        )
        new_prompt = current_prompt
        skip_reasons.append(f"reverted: optimizer introduced new variables {sorted(added_vars)}")
        applied = 0
    else:
        applied = len(edits) - len(apply_failures)

    return OptimizerResult(
        new_prompt=new_prompt,
        edits=edits,
        summary=result.parsed.summary,
        edits_applied=applied,
        edits_skipped=len(skip_reasons),
        skip_reasons=skip_reasons,
        cost_usd=result.cost_usd,
        cache_hit=result.cache_hit,
    )

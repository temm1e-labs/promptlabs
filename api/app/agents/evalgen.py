"""EvalGen agent — generates the rubric and eval items for an experiment.

Inputs:
  - task intent + prompt v0 (template with {{var}} placeholders)
  - declared prompt variables
  - optimization objectives (shapes the rubric weight + criterion selection)
  - known_issues (warm-mode only — items probe these specifically)
  - eval_size (target N) + train_ratio (default 0.7)

Outputs:
  - rubric: 3-6 criteria, each tied to an objective when possible
  - items: N input_vars dicts with optional expected_output + label + tags
  - deterministic 70/30 train/holdout split (index-based on item order)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.core import providers
from app.core.logging import log
from app.schemas.common import PromptVariable, RubricCriterion


class GeneratedEvalItem(BaseModel):
    label: str = Field(description="3-8 word human-readable name for this test case.")
    input_vars: dict[str, str] = Field(
        description=(
            "Map of {{variable_name}} → value. Every key must match a declared prompt variable. "
            "Provide realistic, concrete values — not placeholders."
        )
    )
    expected_output: str | None = Field(
        default=None,
        description=(
            "Optional ground truth. Provide for tasks with objective correctness "
            "(classification, extraction, math). Omit for subjective tasks; the LLM judge "
            "will score against the rubric."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags. Use e.g. 'common', 'edge', 'adversarial', 'known_issue'.",
    )


class EvalGenOutput(BaseModel):
    rubric: list[RubricCriterion] = Field(
        description="3-6 criteria that scoring will use. Each tied to an objective when possible."
    )
    items: list[GeneratedEvalItem] = Field(
        description="Test cases. Mix common, edge, and adversarial inputs."
    )


@dataclass
class EvalGenResult:
    rubric: list[RubricCriterion]
    train_items: list[GeneratedEvalItem]
    holdout_items: list[GeneratedEvalItem]
    cost_usd: float
    cache_hit: bool


_OBJECTIVE_TO_CRITERIA_HINT = {
    "accuracy": (
        "Include a 'correctness' criterion: does the output correctly accomplish the task?"
    ),
    "robustness": (
        "Include criteria for hallucination, off-topic responses, and instruction adherence."
    ),
    "format_adherence": (
        "Include a strict 'format_compliance' criterion checking output structure."
    ),
    "brevity": "Include an 'appropriate_length' criterion.",
    "tone": "Include a 'tone_match' criterion describing the target register.",
    "cost": (
        "Cost is measured automatically; don't add a criterion for it. Do not include "
        "'token_count' or similar runtime metrics."
    ),
    "latency": (
        "Latency is measured automatically; don't add a criterion. Do not include "
        "'response_time' or similar."
    ),
}


_SYSTEM = """You are designing an evaluation harness for a prompt that is being iteratively
optimized. Your output drives both rubric scoring and test-case selection.

Hard rules:
1. EVERY test case's `input_vars` dict MUST have EXACTLY THESE KEYS (no more, no fewer)
   — they are listed under "Required input_vars keys" below. Do not invent synonyms
   ("email" vs "email_content" is a hard fail). Do not drop any required key. If the
   prompt declares N variables, every test case has exactly those N keys, with the same
   spelling and case.
2. Test cases must be DIVERSE: include common cases, edge cases (empty/long/multilingual/
   formatting traps), and adversarial cases (inputs that probe failure modes).
3. The rubric must have 3-6 criteria, each with a clear `definition` (one sentence) and a
   `weight` reflecting its importance. When a criterion maps to an optimization objective,
   set `objective` to that objective's name.
4. Rubric criteria must be JUDGEABLE FROM TEXT ALONE (no runtime metrics like cost/latency —
   those are measured separately by the orchestrator).
5. Provide `expected_output` ONLY for tasks with objective ground truth (classification,
   extraction, math, structured generation). For subjective tasks (creative, summarization,
   chat), leave it null — the LLM judge will score against the rubric.
6. Output strict JSON matching the schema."""


def _objectives_section(objectives: list[str]) -> str:
    lines = []
    for obj in objectives:
        hint = _OBJECTIVE_TO_CRITERIA_HINT.get(obj)
        if hint:
            lines.append(f"- {obj}: {hint}")
    return "\n".join(lines) or "- accuracy: include a 'correctness' criterion."


def _build_user_message(
    *,
    intent: str,
    prompt_v0: str,
    variables: list[PromptVariable],
    objectives: list[str],
    known_issues: str | None,
    eval_size: int,
) -> str:
    declared_keys = [v.name for v in variables]
    vars_block = (
        "\n".join(
            f"  - {v.name}: {v.description} (example: {v.example_value!r})" for v in variables
        )
        or "  (no variables declared)"
    )
    example_input_vars = "{" + ", ".join(f'"{k}": "..."' for k in declared_keys) + "}"
    keys_csv = ", ".join(f'"{k}"' for k in declared_keys) or "(none)"

    issues_block = (
        f"\nUser-reported failure modes (write items that probe these):\n{known_issues}\n"
        if known_issues
        else ""
    )

    return (
        f"Task intent:\n{intent}\n\n"
        f'Prompt template (v0):\n"""\n{prompt_v0}\n"""\n\n'
        f"Declared variables:\n{vars_block}\n\n"
        f"Required input_vars keys for EVERY test case (EXACT spelling, case-sensitive):\n"
        f"  {keys_csv}\n\n"
        f"Each test case's input_vars MUST have shape: {example_input_vars}\n\n"
        f"Optimization objectives → rubric guidance:\n{_objectives_section(objectives)}\n"
        f"{issues_block}\n"
        f"Generate exactly {eval_size} test cases. Mix ~60% common, ~25% edge, ~15% adversarial.\n"
        "Output strict JSON."
    )


def _remap_input_vars(
    item_vars: dict[str, str],
    declared_keys: list[str],
) -> dict[str, str] | None:
    """Reconcile LLM-generated keys with declared variable names.

    1. Exact match: return as-is.
    2. Case-insensitive match: rename.
    3. Same count + ambiguous names: positional remap (declared order ↔ item order).
       Mostly catches synonym substitution (e.g. "email" instead of "email_content"
       when the prompt has a single variable).
    4. Otherwise: None (caller drops the item).
    """
    declared = set(declared_keys)
    keys = set(item_vars.keys())
    if keys == declared:
        return item_vars

    # Case-insensitive
    lower_to_actual = {k.lower(): k for k in item_vars}
    if {k.lower() for k in declared_keys} == set(lower_to_actual.keys()):
        return {dk: item_vars[lower_to_actual[dk.lower()]] for dk in declared_keys}

    # Positional fallback for same-count mismatches (common when len == 1)
    if len(item_vars) == len(declared_keys) and len(declared_keys) >= 1:
        item_keys = list(item_vars.keys())
        return {declared_keys[i]: item_vars[item_keys[i]] for i in range(len(declared_keys))}

    return None


def _filter_items(
    items: list[GeneratedEvalItem],
    declared_vars: list[str],
) -> list[GeneratedEvalItem]:
    """Reconcile each item's input_vars with the declared prompt variables.

    Items whose keys can't be reconciled are dropped (logged for visibility).
    """
    kept: list[GeneratedEvalItem] = []
    for item in items:
        remapped = _remap_input_vars(item.input_vars, declared_vars)
        if remapped is None:
            log.debug(
                "evalgen.item_dropped",
                label=item.label,
                provided=sorted(item.input_vars.keys()),
                declared=declared_vars,
            )
            continue
        if remapped is not item.input_vars:
            log.debug(
                "evalgen.item_remapped",
                label=item.label,
                provided=sorted(item.input_vars.keys()),
                declared=declared_vars,
            )
        kept.append(item.model_copy(update={"input_vars": remapped}))
    return kept


def _split(
    items: list[GeneratedEvalItem], train_ratio: float
) -> tuple[list[GeneratedEvalItem], list[GeneratedEvalItem]]:
    """Index-based deterministic split. First ceil(N*ratio) → train, rest → holdout."""
    import math

    n_train = max(1, math.ceil(len(items) * train_ratio))
    n_train = min(n_train, len(items) - 1) if len(items) > 1 else n_train
    return items[:n_train], items[n_train:]


async def generate(
    *,
    intent: str,
    prompt_v0: str,
    variables: list[PromptVariable],
    objectives: list[str],
    known_issues: str | None,
    eval_size: int,
    train_ratio: float,
    model: str,
) -> EvalGenResult:
    user_msg = _build_user_message(
        intent=intent,
        prompt_v0=prompt_v0,
        variables=variables,
        objectives=objectives,
        known_issues=known_issues,
        eval_size=eval_size,
    )

    declared_var_names = [v.name for v in variables]
    if not declared_var_names:
        # No variables in the prompt — auto-add an `input` slot so eval items can vary
        log.info("evalgen.no_variables_synthesizing_input_slot")
        declared_var_names = ["input"]

    result = await providers.complete(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format=EvalGenOutput,
        temperature=0.7,
    )

    if isinstance(result.parsed, EvalGenOutput):
        log.info(
            "evalgen.parsed",
            n_rubric=len(result.parsed.rubric),
            n_items=len(result.parsed.items),
            declared_keys=declared_var_names,
        )
        rubric = result.parsed.rubric
        items = _filter_items(result.parsed.items, declared_var_names)
        if len(items) < len(result.parsed.items):
            log.warning(
                "evalgen.items_filtered",
                before=len(result.parsed.items),
                after=len(items),
                first_item_keys=(
                    list(result.parsed.items[0].input_vars.keys())
                    if result.parsed.items
                    else []
                ),
            )
    else:
        log.warning(
            "evalgen.parse_failed_degraded_fallback",
            content_preview=result.content[:500],
        )
        rubric = [
            RubricCriterion(
                name="correctness",
                definition="The output correctly accomplishes the task as described in the intent.",
                weight=1.0,
                objective="accuracy",
            )
        ]
        items = []

    train, holdout = _split(items, train_ratio)
    return EvalGenResult(
        rubric=rubric,
        train_items=train,
        holdout_items=holdout,
        cost_usd=result.cost_usd,
        cache_hit=result.cache_hit,
    )


def to_db_items(generated: list[GeneratedEvalItem]) -> list[dict[str, Any]]:
    """Helper for the orchestrator: shape items for SQLAlchemy bulk insert."""
    return [
        {
            "input_vars": item.input_vars,
            "expected_output": item.expected_output,
            "label": item.label,
            "item_metadata": {"tags": item.tags},
        }
        for item in generated
    ]

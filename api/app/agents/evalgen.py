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
    input_text: str | None = Field(
        default=None,
        description=(
            "PRIMARY field for the test input. For single-variable prompts (the common case), "
            "set this to the realistic value that should be substituted into the prompt's "
            "{{variable}} placeholder. For multi-variable prompts, leave null and use input_vars."
        ),
    )
    input_vars: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "FOR MULTI-VARIABLE PROMPTS ONLY: map of {{variable_name}} → value with every "
            "declared variable populated. Keys must match exactly (case-sensitive). For "
            "single-variable prompts, prefer `input_text` instead."
        ),
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
1. The TEST INPUT for each case MUST be populated. Use one of:
   (a) `input_text` (PREFERRED for single-variable prompts) — a single string with the
       realistic value that should be substituted into the prompt's {{variable}} placeholder.
   (b) `input_vars` (ONLY for prompts with 2+ variables) — full {key: value} dict with
       every declared variable.
   NEVER leave both empty. NEVER put the test data only in `label` or `expected_output`.
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

    if len(declared_keys) <= 1:
        which_field = (
            f"Use `input_text` (a SINGLE STRING) — it will be substituted into "
            f"{{{{{declared_keys[0]}}}}} if declared." if declared_keys else
            "Use `input_text` (a SINGLE STRING) — there are no declared variables."
        )
        example_item = (
            "{\n"
            '  "label": "Common emergency case",\n'
            '  "input_text": "<realistic test input — the actual string to test against>",\n'
            '  "input_vars": {},\n'
            '  "expected_output": "<ground truth, or null for subjective tasks>",\n'
            '  "tags": ["common"]\n'
            "}"
        )
    else:
        keys_csv = ", ".join(f'"{k}"' for k in declared_keys)
        which_field = (
            f"Use `input_vars` — the prompt has {len(declared_keys)} variables: {keys_csv}. "
            f"Every test case's input_vars MUST contain ALL of these keys."
        )
        example_input_vars = (
            "{" + ", ".join(f'"{k}": "...value..."' for k in declared_keys) + "}"
        )
        example_item = (
            "{\n"
            '  "label": "Realistic case name",\n'
            '  "input_text": null,\n'
            f'  "input_vars": {example_input_vars},\n'
            '  "expected_output": "<ground truth, or null>",\n'
            '  "tags": ["common"]\n'
            "}"
        )

    issues_block = (
        f"\nUser-reported failure modes (write items that probe these):\n{known_issues}\n"
        if known_issues
        else ""
    )

    return (
        f"Task intent:\n{intent}\n\n"
        f'Prompt template (v0):\n"""\n{prompt_v0}\n"""\n\n'
        f"Declared variables:\n{vars_block}\n\n"
        f"How to populate the test input for each case:\n  {which_field}\n\n"
        f"Concrete example for ONE item:\n{example_item}\n\n"
        f"Optimization objectives → rubric guidance:\n{_objectives_section(objectives)}\n"
        f"{issues_block}\n"
        f"Generate exactly {eval_size} test cases. Mix ~60% common, ~25% edge, ~15% adversarial.\n"
        "Output strict JSON."
    )


def _reconcile_item(
    item: GeneratedEvalItem,
    declared_keys: list[str],
) -> dict[str, str] | None:
    """Return the reconciled input_vars dict, or None if irrecoverable.

    Strategy (most permissive first):
      1. input_vars exact-match declared keys → use as-is.
      2. input_vars case-insensitive match → rename.
      3. input_vars same-count, different keys → positional remap.
      4. input_vars empty but input_text present → map input_text to first declared var.
      5. input_vars empty AND input_text empty → drop.
    """
    declared = set(declared_keys)
    item_vars = item.input_vars or {}

    # (1) exact match
    if item_vars and set(item_vars.keys()) == declared:
        return dict(item_vars)

    # (2) case-insensitive
    if item_vars:
        lower_to_actual = {k.lower(): k for k in item_vars}
        if {k.lower() for k in declared_keys} == set(lower_to_actual.keys()):
            return {dk: item_vars[lower_to_actual[dk.lower()]] for dk in declared_keys}

    # (3) positional
    if item_vars and len(item_vars) == len(declared_keys) and len(declared_keys) >= 1:
        item_keys = list(item_vars.keys())
        return {declared_keys[i]: item_vars[item_keys[i]] for i in range(len(declared_keys))}

    # (4) fall back to flat input_text → first declared var
    if item.input_text and len(declared_keys) >= 1:
        result = {declared_keys[0]: item.input_text}
        # Optionally fill remaining vars from input_vars if some keys overlap
        for k in declared_keys[1:]:
            if k in item_vars:
                result[k] = item_vars[k]
        return result

    return None


def _filter_items(
    items: list[GeneratedEvalItem],
    declared_vars: list[str],
) -> list[GeneratedEvalItem]:
    """Reconcile each item's input data with the declared prompt variables.

    Items whose inputs can't be reconciled are dropped (logged for visibility).
    """
    kept: list[GeneratedEvalItem] = []
    for item in items:
        reconciled = _reconcile_item(item, declared_vars)
        if reconciled is None:
            log.debug(
                "evalgen.item_dropped",
                label=item.label,
                provided_vars=sorted((item.input_vars or {}).keys()),
                has_input_text=bool(item.input_text),
                declared=declared_vars,
            )
            continue
        kept.append(item.model_copy(update={"input_vars": reconciled}))
    return kept


def _split(
    items: list[GeneratedEvalItem],
    train_ratio: float,
    seed: int = 0,
) -> tuple[list[GeneratedEvalItem], list[GeneratedEvalItem]]:
    """Shuffled, deterministic train/holdout split.

    EvalGen tends to emit items in the order requested by its system prompt
    ("~60% common, ~25% edge, ~15% adversarial"). Index-based slicing put
    common items in train and adversarial in holdout, making train
    systematically easier and invalidating every gap comparison. Shuffling
    with a deterministic seed removes that bias while keeping the split
    reproducible per experiment.
    """
    import math
    import random

    n = len(items)
    n_train = max(1, math.ceil(n * train_ratio))
    n_train = min(n_train, n - 1) if n > 1 else n_train

    idx = list(range(n))
    random.Random(seed).shuffle(idx)  # noqa: S311 — non-cryptographic shuffle is intentional
    train_set = set(idx[:n_train])
    train = [items[i] for i in range(n) if i in train_set]
    holdout = [items[i] for i in range(n) if i not in train_set]
    return train, holdout


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
    split_seed: int = 0,
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

    train, holdout = _split(items, train_ratio, seed=split_seed)
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

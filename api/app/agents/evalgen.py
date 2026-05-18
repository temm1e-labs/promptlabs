"""EvalGen agent — generates the rubric and eval items for an experiment.

Two execution paths, chosen by ``eval_size``:

  Path A — Single call (``eval_size <= SINGLE_CALL_MAX``):
    One LLM call returns rubric + items. Fast and cheap for small eval sets;
    a single LLM is reliably good at producing 10-15 diverse cases in one shot.

  Path B — Batched (``eval_size > SINGLE_CALL_MAX``, default 15):
    1. Taxonomy call: derive rubric + a stratification axis of 5-8 categories.
    2. Parallel per-category batches: one LLM call per category, fired
       concurrently via ``asyncio.gather``. Each batch only has to produce
       ``ceil(eval_size / n_categories)`` items — far inside the "reliable
       structured-output" envelope of a single call.
    3. Text-hash dedup: normalize input text and drop verbatim duplicates.
    4. Optional top-up: if the dedup'd pool is short, one more call fills
       the gap with explicit avoid-list of existing labels.

  Why batched: a single call asked for 40-50+ items tends to truncate the
  tail JSON, repeat themes, or produce shallow variations. Batches of ~8
  cases stay inside the structured-output envelope, and stratification by
  taxonomy makes duplication across batches structurally unlikely.

Outputs (both paths):
  - rubric: 3-6 criteria, each tied to an objective when possible
  - items: N input_vars dicts with optional expected_output + label + tags
  - shuffled, deterministic train/holdout split (seeded per experiment)
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.core import providers
from app.core.logging import log
from app.schemas.common import PromptVariable, RubricCriterion

SINGLE_CALL_MAX = 15  # eval_size at or below this stays on the single-call path
ITEMS_PER_BATCH = 8  # target items per per-category batch on the batched path
MIN_CATEGORIES = 5
MAX_CATEGORIES = 8

ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


# ─── schemas ─────────────────────────────────────────────────────────────


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
    """Single-call path schema: rubric + items in one shot."""

    rubric: list[RubricCriterion] = Field(
        description="3-6 criteria that scoring will use. Each tied to an objective when possible."
    )
    items: list[GeneratedEvalItem] = Field(
        description="Test cases. Mix common, edge, and adversarial inputs."
    )


class EvalCategory(BaseModel):
    name: str = Field(
        description=(
            "Short kebab-case identifier for this category (e.g. 'billing-clear', "
            "'edge-multilingual', 'adversarial-prompt-injection')."
        )
    )
    description: str = Field(
        description=(
            "One or two sentences describing what makes inputs in this category distinct "
            "and what aspect of the prompt they probe."
        )
    )
    target_tag: str = Field(
        description=(
            "One of: 'common' (happy path), 'edge' (unusual but legitimate), "
            "'adversarial' (attempts to break the prompt)."
        )
    )
    target_count: int = Field(
        ge=1,
        description="How many test cases should be drawn from this category.",
    )


class TaxonomyOutput(BaseModel):
    """Batched-path step 1: rubric + stratification axis."""

    rubric: list[RubricCriterion] = Field(
        description="3-6 criteria that scoring will use. Each tied to an objective when possible."
    )
    categories: list[EvalCategory] = Field(
        description=(
            f"{MIN_CATEGORIES}-{MAX_CATEGORIES} categories that partition the input space. "
            "Together they should cover common, edge, and adversarial inputs in proportion "
            "~60% common / ~25% edge / ~15% adversarial."
        ),
    )


class CategoryBatchOutput(BaseModel):
    """Batched-path step 2: items for one category."""

    items: list[GeneratedEvalItem]


class TopupOutput(BaseModel):
    """Batched-path step 4: extra items to fill gap after dedup."""

    items: list[GeneratedEvalItem]


@dataclass
class EvalGenResult:
    rubric: list[RubricCriterion]
    train_items: list[GeneratedEvalItem]
    holdout_items: list[GeneratedEvalItem]
    cost_usd: float
    cache_hit: bool


# ─── prompt construction ─────────────────────────────────────────────────


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


_SYSTEM_SINGLE = """You are designing an evaluation harness for a prompt that is being iteratively
optimized. Your output drives both rubric scoring and test-case selection.

Hard rules:
1. The TEST INPUT for each case MUST be populated. Use one of:
   (a) `input_text` (PREFERRED for single-variable prompts) — a single string with the
       realistic value that should be substituted into the prompt's {{variable}} placeholder.
   (b) `input_vars` (ONLY for prompts with 2+ variables) — full {key: value} dict with
       every declared variable.
   NEVER leave both empty. NEVER put the test data only in `label` or `expected_output`.
   `label` is a 3-8 word summary (e.g. "Dog Food Price Check"), NOT the actual input.

   WRONG (do NOT do this):
       {"label": "Dog Food Price Check", "input_vars": {}, "input_text": null}
   RIGHT:
       {"label": "Dog Food Price Check",
        "input_text": "How much is the 20lb bag of Acme dog food?"}
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


_SYSTEM_TAXONOMY = """You are designing the stratification axis for an evaluation harness.
Given a prompt and its intent, you produce TWO things:

1. A rubric of 3-6 criteria that scoring will use (with weights and definitions).
2. A taxonomy of input categories that partition the test-case space.

The categories will each be expanded by a separate LLM call. Your job is to define them
clearly enough that those downstream calls produce diverse, non-overlapping items.

Hard rules:
1. Categories must be DISJOINT — an input should belong to one category, not multiple.
2. Categories must be CONCRETE — "edge cases" is too vague; "non-English inputs" or
   "single-token inputs" is concrete.
3. Use proportions ~60% common / ~25% edge / ~15% adversarial (set `target_count` per
   category so they sum to approximately the requested total).
4. Rubric criteria must be JUDGEABLE FROM TEXT ALONE (no runtime metrics).
5. When a criterion maps to a known optimization objective, set its `objective` field.
6. Output strict JSON."""


_SYSTEM_BATCH = """You are generating test cases for ONE category of an evaluation harness.
A separate process has already chosen the rubric and the category boundaries. Your job is
to produce diverse, realistic items WITHIN this single category.

Hard rules:
1. Every item's TEST INPUT must be populated via `input_text` (single-variable prompts)
   or `input_vars` (multi-variable prompts). NEVER leave both empty. NEVER put the test
   data only in `label` or `expected_output` — `label` is a short summary, NOT the input.

   WRONG (do NOT do this):
       {"label": "Dog Food Price Check", "input_vars": {}, "input_text": null}
   RIGHT:
       {"label": "Dog Food Price Check",
        "input_vars": {"store_context": "Pet Mart, est 2010",
                       "customer_query": "How much is the 20lb bag of Acme dog food?"}}

2. All items in your output must clearly belong to the assigned category — do NOT bleed
   into adjacent categories.
3. Items must be DIVERSE within the category. Don't produce N rewordings of the same idea.
4. Provide `expected_output` ONLY for tasks with objective ground truth. Otherwise null.
5. Tag each item with the category's tag plus optional sub-tags.
6. Output strict JSON."""


def _objectives_section(objectives: list[str]) -> str:
    lines = []
    for obj in objectives:
        hint = _OBJECTIVE_TO_CRITERIA_HINT.get(obj)
        if hint:
            lines.append(f"- {obj}: {hint}")
    return "\n".join(lines) or "- accuracy: include a 'correctness' criterion."


def _which_field_block(variables: list[PromptVariable]) -> tuple[str, str]:
    """Return (which_field_instruction, example_item_json) given declared vars.

    For multi-variable prompts the instruction is heavily directive about the
    exact keys to use, and labels each variable as "user input" or "context"
    based on a name heuristic. SOTA models would have followed the looser
    instruction; weaker preview models need the extra spoonfeeding.
    """
    declared_keys = [v.name for v in variables]

    if len(declared_keys) <= 1:
        which_field = (
            f"Use `input_text` (a SINGLE STRING) — it will be substituted into "
            f"{{{{{declared_keys[0]}}}}} if declared."
            if declared_keys
            else "Use `input_text` (a SINGLE STRING) — there are no declared variables."
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
        input_idx = _pick_input_var(variables)
        keys_bulleted = "\n".join(f'      • "{k}"' for k in declared_keys)
        role_lines = []
        for i, v in enumerate(variables):
            role = "USER INPUT (vary this per item)" if i == input_idx else (
                "CONTEXT (use example_value or a similar realistic value — usually constant)"
            )
            role_lines.append(f"      • {v.name} → {role}")
        roles_block = "\n".join(role_lines)
        which_field = (
            f"Use `input_vars` (a JSON object). The prompt has {len(declared_keys)} variables.\n\n"
            f"    CRITICAL — input_vars must contain EXACTLY these keys, no more, no less:\n"
            f"{keys_bulleted}\n\n"
            f"    Do NOT invent new keys (no 'query', 'user_query', 'inquiry', etc).\n"
            f"    Do NOT omit any of the keys above.\n"
            f"    Each variable's role:\n"
            f"{roles_block}"
        )
        example_input_vars_lines = "\n".join(
            f'    "{v.name}": "<realistic value for {v.name}>"' for v in variables
        )
        example_item = (
            "{\n"
            '  "label": "Realistic case name",\n'
            '  "input_text": null,\n'
            '  "input_vars": {\n'
            f"{example_input_vars_lines}\n"
            "  },\n"
            '  "expected_output": "<ground truth, or null>",\n'
            '  "tags": ["common"]\n'
            "}"
        )
    return which_field, example_item


def _build_user_single(
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
    which_field, example_item = _which_field_block(variables)
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


def _build_user_taxonomy(
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
    issues_block = (
        f"\nUser-reported failure modes (carve out at least one category that probes these):\n"
        f"{known_issues}\n"
        if known_issues
        else ""
    )

    return (
        f"Task intent:\n{intent}\n\n"
        f'Prompt template (v0):\n"""\n{prompt_v0}\n"""\n\n'
        f"Declared variables:\n{vars_block}\n\n"
        f"Optimization objectives → rubric guidance:\n{_objectives_section(objectives)}\n"
        f"{issues_block}\n"
        f"Design {MIN_CATEGORIES}-{MAX_CATEGORIES} input categories that, between them, will "
        f"yield approximately {eval_size} test cases total. Sum of target_count fields should "
        f"be approximately {eval_size}.\n\n"
        "Output strict JSON with both `rubric` and `categories`."
    )


def _build_user_batch(
    *,
    intent: str,
    prompt_v0: str,
    variables: list[PromptVariable],
    category: EvalCategory,
    rubric_summary: str,
) -> str:
    declared_keys = [v.name for v in variables]
    vars_block = (
        "\n".join(
            f"  - {v.name}: {v.description} (example: {v.example_value!r})" for v in variables
        )
        or "  (no variables declared)"
    )
    which_field, example_item = _which_field_block(variables)

    return (
        f"Task intent:\n{intent}\n\n"
        f'Prompt template (v0):\n"""\n{prompt_v0}\n"""\n\n'
        f"Declared variables:\n{vars_block}\n\n"
        f"How to populate the test input for each case:\n  {which_field}\n\n"
        f"Concrete example for ONE item:\n{example_item}\n\n"
        f"Rubric criteria the prompt will be scored against:\n{rubric_summary}\n\n"
        f"Category you must generate:\n"
        f"  name: {category.name}\n"
        f"  description: {category.description}\n"
        f"  target tag: {category.target_tag}\n\n"
        f"Generate exactly {category.target_count} items that belong to THIS category. "
        f"Do not produce items that would fit better in a different category. Be diverse "
        f"within the category.\nOutput strict JSON."
    )


def _build_user_topup(
    *,
    intent: str,
    prompt_v0: str,
    variables: list[PromptVariable],
    n_needed: int,
    existing_labels: list[str],
    rubric_summary: str,
) -> str:
    declared_keys = [v.name for v in variables]
    vars_block = (
        "\n".join(
            f"  - {v.name}: {v.description} (example: {v.example_value!r})" for v in variables
        )
        or "  (no variables declared)"
    )
    which_field, example_item = _which_field_block(variables)
    avoid_block = "\n".join(f"  - {label}" for label in existing_labels[:60])

    return (
        f"Task intent:\n{intent}\n\n"
        f'Prompt template (v0):\n"""\n{prompt_v0}\n"""\n\n'
        f"Declared variables:\n{vars_block}\n\n"
        f"How to populate the test input for each case:\n  {which_field}\n\n"
        f"Concrete example for ONE item:\n{example_item}\n\n"
        f"Rubric criteria the prompt will be scored against:\n{rubric_summary}\n\n"
        f"Test cases ALREADY GENERATED (do NOT produce near-duplicates):\n{avoid_block}\n\n"
        f"Generate exactly {n_needed} additional test cases that are distinct in BOTH input "
        f"text and the failure mode they probe. Prefer edge and adversarial cases.\n"
        "Output strict JSON."
    )


# ─── reconciliation + dedup ──────────────────────────────────────────────


# Keywords that suggest a variable is the user-facing input (vs context).
# When EvalGen produces an item with `input_text` only but the prompt has
# multiple declared variables, we pick the variable whose name contains one
# of these keywords as the target for the input_text. Falls back to the
# LAST declared variable, since prompt templates conventionally place the
# user's turn at the end.
_INPUT_LIKE_KEYWORDS = frozenset(
    {
        "query", "question", "input", "user_input", "user_message", "message",
        "prompt", "text", "request", "inquiry", "ask", "user", "utterance",
        "turn", "msg",
    }
)


def _name_tokens(name: str) -> set[str]:
    """Split a variable name into normalized tokens for keyword matching.

    'customer_query' → {'customer', 'query'}
    'userInputText' → {'user', 'input', 'text'}    'context' → {'context'}

    Used by _pick_input_var so substring false positives don't fire
    (e.g., 'text' is a substring of 'context' but they aren't the same token).
    """
    # Split on underscores AND camelCase boundaries
    import re
    parts: list[str] = []
    for piece in name.split("_"):
        # Split camelCase (insert space before uppercase, then lowercase)
        camel_split = re.sub(r"(?<!^)(?=[A-Z])", "_", piece).split("_")
        parts.extend(p.lower() for p in camel_split if p)
    return set(parts)


def _pick_input_var(variables: list[PromptVariable]) -> int:
    """Return the index of the variable most likely to hold the user's input.

    Heuristic:
      1. First variable with a TOKEN matching a known input-like keyword.
         Token-based (not substring) so 'text' in 'context' doesn't fire.
      2. Otherwise, the LAST variable (conventional placement of user turn).
    """
    if not variables:
        return 0
    for i, v in enumerate(variables):
        if _name_tokens(v.name) & _INPUT_LIKE_KEYWORDS:
            return i
    return len(variables) - 1


def _example_default(v: PromptVariable) -> str:
    """Sensible default value for a context variable when EvalGen omitted it."""
    return v.example_value or ""


def _reconcile_item(
    item: GeneratedEvalItem,
    variables: list[PromptVariable],
) -> dict[str, str] | None:
    """Return the reconciled input_vars dict, or None if irrecoverable.

    Strategy (most permissive first):
      1. input_vars exact-match declared keys → use as-is.
      2. input_vars case-insensitive match → rename.
      3. input_vars same-count, different keys → positional remap.
      4. input_vars is a non-empty subset of declared → fill missing keys
         with their `example_value`. This handles the common multi-variable
         case where the prompt has context vars (e.g. {{store_info}}) plus
         one user-facing var (e.g. {{customer_query}}), and EvalGen only
         varies the user-facing one per test case.
      5. input_text present + multi-var prompt → map input_text to the
         most "input-like" declared var (by name keyword), fill the rest
         from example_value. Single-var prompt falls through to mapping
         input_text to the lone var.
      6. Nothing usable → drop.
    """
    declared_keys = [v.name for v in variables]
    if not declared_keys:
        return None
    declared = set(declared_keys)
    item_vars = item.input_vars or {}

    # (1) exact match
    if item_vars and set(item_vars.keys()) == declared:
        return dict(item_vars)

    # (2) case-insensitive match
    if item_vars:
        lower_to_actual = {k.lower(): k for k in item_vars}
        if {k.lower() for k in declared_keys} == set(lower_to_actual.keys()):
            return {dk: item_vars[lower_to_actual[dk.lower()]] for dk in declared_keys}

    # (3) positional (same count, different keys, single-var case included)
    if item_vars and len(item_vars) == len(declared_keys) and len(declared_keys) >= 1:
        item_keys = list(item_vars.keys())
        return {declared_keys[i]: item_vars[item_keys[i]] for i in range(len(declared_keys))}

    # (4) PARTIAL: item has SOME declared vars — fill missing from example_value.
    # Handles the common case of multi-var prompts where EvalGen only varies
    # the user-input var per test case and omits context vars.
    if item_vars:
        # Match keys case-insensitively to be lenient
        lower_to_actual = {k.lower(): k for k in item_vars}
        result: dict[str, str] = {}
        matched_any = False
        for v in variables:
            if v.name in item_vars:
                result[v.name] = item_vars[v.name]
                matched_any = True
            elif v.name.lower() in lower_to_actual:
                result[v.name] = item_vars[lower_to_actual[v.name.lower()]]
                matched_any = True
            else:
                result[v.name] = _example_default(v)
        if matched_any:
            return result

    # (5) input_text present — map to most input-like var, fill the rest
    if item.input_text:
        idx = _pick_input_var(variables)
        result = {}
        for i, v in enumerate(variables):
            if i == idx:
                result[v.name] = item.input_text
            elif v.name in item_vars:
                result[v.name] = item_vars[v.name]
            else:
                result[v.name] = _example_default(v)
        return result

    # (6) FINAL SALVAGE: item has input_vars with unrelated keys but non-empty
    # string content. Treat the longest non-empty string value as the user's
    # input (most likely the test case the model meant to generate), map it
    # to the most-input-like declared var, fill the rest from example_value.
    # This is the catch-all for when EvalGen invents its own key names that
    # don't match declared variables — common with smaller/preview models.
    if item_vars:
        candidates = [s for s in item_vars.values() if isinstance(s, str) and s.strip()]
        if candidates:
            longest = max(candidates, key=len)
            idx = _pick_input_var(variables)
            result = {}
            for i, v in enumerate(variables):
                result[v.name] = longest if i == idx else _example_default(v)
            return result

    # (7) LABEL/EXPECTED-OUTPUT SALVAGE: some weak models put the entire test
    # concept in `label` (e.g. "Dog Food Price Check") with empty input_vars
    # AND empty input_text. The label IS the user-meaningful content. Use it
    # as the input — degraded but lets the experiment proceed instead of
    # failing with 0 usable items.
    fallback_text = (item.label or "").strip() or (item.expected_output or "").strip()
    if fallback_text:
        idx = _pick_input_var(variables)
        result = {}
        for i, v in enumerate(variables):
            result[v.name] = fallback_text if i == idx else _example_default(v)
        return result

    return None


def _filter_items(
    items: list[GeneratedEvalItem],
    variables: list[PromptVariable],
) -> list[GeneratedEvalItem]:
    """Reconcile each item's input data with the declared prompt variables.

    Items whose inputs can't be reconciled are dropped. When the drop rate is
    high, a WARNING is emitted with a sample of the dropped items' shape so
    operators can see why reconciliation failed (e.g., EvalGen used different
    variable names than declared).
    """
    declared_var_names = [v.name for v in variables]
    kept: list[GeneratedEvalItem] = []
    dropped_samples: list[dict[str, Any]] = []
    for item in items:
        reconciled = _reconcile_item(item, variables)
        if reconciled is None:
            if len(dropped_samples) < 3:
                dropped_samples.append(
                    {
                        "label": item.label,
                        "provided_vars": sorted((item.input_vars or {}).keys()),
                        "has_input_text": bool(item.input_text),
                        "input_text_preview": (item.input_text or "")[:80],
                    }
                )
            log.debug(
                "evalgen.item_dropped",
                label=item.label,
                provided_vars=sorted((item.input_vars or {}).keys()),
                has_input_text=bool(item.input_text),
                declared=declared_var_names,
            )
            continue
        kept.append(item.model_copy(update={"input_vars": reconciled}))

    n_dropped = len(items) - len(kept)
    # Surface details when the drop rate is non-trivial — silent drops were
    # the hard-to-diagnose failure that made earlier loops fail with 0 items.
    if n_dropped > 0 and (n_dropped >= 3 or n_dropped == len(items)):
        log.warning(
            "evalgen.reconcile_drops",
            n_dropped=n_dropped,
            n_total=len(items),
            declared=declared_var_names,
            sample=dropped_samples,
        )
    return kept


_WS_RUN = re.compile(r"\s+")


def _normalize_for_dedup(item: GeneratedEvalItem) -> str:
    """Build a canonical string for duplicate detection.

    Hashes the lowercased, whitespace-collapsed concatenation of input_vars
    (in sorted-key order). Two items collide iff their normalized inputs are
    identical — catches verbatim duplicates and trivial whitespace variants
    without needing embedding infrastructure.
    """
    parts: list[str] = []
    for k in sorted(item.input_vars.keys()):
        v = item.input_vars[k] or ""
        parts.append(f"{k}={v}")
    blob = "||".join(parts).lower()
    return _WS_RUN.sub(" ", blob).strip()


def _dedup_items(items: list[GeneratedEvalItem]) -> list[GeneratedEvalItem]:
    """Drop items whose normalized form matches an earlier item's."""
    seen: set[str] = set()
    kept: list[GeneratedEvalItem] = []
    for item in items:
        key = _normalize_for_dedup(item)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


# ─── split ───────────────────────────────────────────────────────────────


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
    import random

    n = len(items)
    if n == 0:
        return [], []
    n_train = max(1, math.ceil(n * train_ratio))
    n_train = min(n_train, n - 1) if n > 1 else n_train

    idx = list(range(n))
    random.Random(seed).shuffle(idx)  # noqa: S311 — non-cryptographic shuffle is intentional
    train_set = set(idx[:n_train])
    train = [items[i] for i in range(n) if i in train_set]
    holdout = [items[i] for i in range(n) if i not in train_set]
    return train, holdout


# ─── single-call path ────────────────────────────────────────────────────


async def _generate_single(
    *,
    intent: str,
    prompt_v0: str,
    variables: list[PromptVariable],
    objectives: list[str],
    known_issues: str | None,
    eval_size: int,
    train_ratio: float,
    model: str,
    split_seed: int,
    declared_var_names: list[str],
) -> EvalGenResult:
    user_msg = _build_user_single(
        intent=intent,
        prompt_v0=prompt_v0,
        variables=variables,
        objectives=objectives,
        known_issues=known_issues,
        eval_size=eval_size,
    )
    result = await providers.complete(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_SINGLE},
            {"role": "user", "content": user_msg},
        ],
        response_format=EvalGenOutput,
        temperature=0.7,
    )
    if isinstance(result.parsed, EvalGenOutput):
        rubric = result.parsed.rubric
        items_raw = result.parsed.items
        items = _filter_items(items_raw, variables)
        items = _dedup_items(items)
        if len(items) < len(items_raw):
            log.warning(
                "evalgen.items_filtered",
                before=len(items_raw),
                after=len(items),
            )
    else:
        log.warning(
            "evalgen.parse_failed_degraded_fallback",
            content_preview=result.content[:500],
        )
        rubric = [
            RubricCriterion(
                name="correctness",
                definition=(
                    "The output correctly accomplishes the task as described in the intent."
                ),
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


# ─── batched path ────────────────────────────────────────────────────────


def _rubric_summary(rubric: list[RubricCriterion]) -> str:
    return "\n".join(f"  - {c.name} (weight {c.weight}): {c.definition}" for c in rubric)


def _normalize_categories(
    categories: list[EvalCategory],
    eval_size: int,
) -> list[EvalCategory]:
    """Clamp category count and rebalance target_counts to ~eval_size total."""
    if not categories:
        return []
    if len(categories) > MAX_CATEGORIES:
        categories = categories[:MAX_CATEGORIES]

    total = sum(c.target_count for c in categories)
    if total <= 0:
        per_cat = max(1, math.ceil(eval_size / len(categories)))
        return [c.model_copy(update={"target_count": per_cat}) for c in categories]

    # Rescale so the sum is approximately eval_size, with a generous buffer
    # (over-generate ~15% to absorb dedup losses).
    target_total = max(eval_size, int(math.ceil(eval_size * 1.15)))
    scale = target_total / total
    rescaled: list[EvalCategory] = []
    for c in categories:
        new_count = max(1, int(round(c.target_count * scale)))
        rescaled.append(c.model_copy(update={"target_count": new_count}))
    return rescaled


async def _generate_taxonomy(
    *,
    intent: str,
    prompt_v0: str,
    variables: list[PromptVariable],
    objectives: list[str],
    known_issues: str | None,
    eval_size: int,
    model: str,
) -> tuple[TaxonomyOutput | None, float, bool]:
    user_msg = _build_user_taxonomy(
        intent=intent,
        prompt_v0=prompt_v0,
        variables=variables,
        objectives=objectives,
        known_issues=known_issues,
        eval_size=eval_size,
    )
    result = await providers.complete(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_TAXONOMY},
            {"role": "user", "content": user_msg},
        ],
        response_format=TaxonomyOutput,
        temperature=0.5,
    )
    parsed = result.parsed if isinstance(result.parsed, TaxonomyOutput) else None
    return parsed, result.cost_usd, result.cache_hit


async def _generate_category_batch(
    *,
    intent: str,
    prompt_v0: str,
    variables: list[PromptVariable],
    category: EvalCategory,
    rubric_summary: str,
    model: str,
) -> tuple[list[GeneratedEvalItem], float, bool]:
    user_msg = _build_user_batch(
        intent=intent,
        prompt_v0=prompt_v0,
        variables=variables,
        category=category,
        rubric_summary=rubric_summary,
    )
    try:
        result = await providers.complete(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_BATCH},
                {"role": "user", "content": user_msg},
            ],
            response_format=CategoryBatchOutput,
            temperature=0.7,
        )
    except Exception as exc:  # one failed batch should not poison the whole run
        log.warning(
            "evalgen.batch_failed",
            category=category.name,
            error=str(exc)[:200],
        )
        return [], 0.0, False
    if isinstance(result.parsed, CategoryBatchOutput):
        # Apply the category's tag so downstream stratification stays visible
        items = [
            item.model_copy(
                update={
                    "tags": list(dict.fromkeys([*item.tags, category.target_tag, category.name]))
                }
            )
            for item in result.parsed.items
        ]
        return items, result.cost_usd, result.cache_hit
    log.warning(
        "evalgen.batch_parse_failed",
        category=category.name,
        content_preview=result.content[:200],
    )
    return [], result.cost_usd, result.cache_hit


async def _generate_topup(
    *,
    intent: str,
    prompt_v0: str,
    variables: list[PromptVariable],
    n_needed: int,
    existing_labels: list[str],
    rubric_summary: str,
    model: str,
) -> tuple[list[GeneratedEvalItem], float, bool]:
    user_msg = _build_user_topup(
        intent=intent,
        prompt_v0=prompt_v0,
        variables=variables,
        n_needed=n_needed,
        existing_labels=existing_labels,
        rubric_summary=rubric_summary,
    )
    try:
        result = await providers.complete(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_BATCH},
                {"role": "user", "content": user_msg},
            ],
            response_format=TopupOutput,
            temperature=0.8,
        )
    except Exception as exc:
        log.warning("evalgen.topup_failed", error=str(exc)[:200])
        return [], 0.0, False
    if isinstance(result.parsed, TopupOutput):
        return list(result.parsed.items), result.cost_usd, result.cache_hit
    return [], result.cost_usd, result.cache_hit


async def _emit(
    on_progress: ProgressCallback | None,
    event_type: str,
    **data: Any,
) -> None:
    if on_progress is None:
        return
    try:
        await on_progress(event_type, data)
    except Exception as exc:
        log.warning("evalgen.progress_callback_failed", error=str(exc)[:120])


async def _generate_batched(
    *,
    intent: str,
    prompt_v0: str,
    variables: list[PromptVariable],
    objectives: list[str],
    known_issues: str | None,
    eval_size: int,
    train_ratio: float,
    model: str,
    split_seed: int,
    declared_var_names: list[str],
    on_progress: ProgressCallback | None,
) -> EvalGenResult:
    # 1. Taxonomy
    taxonomy, taxo_cost, taxo_cache = await _generate_taxonomy(
        intent=intent,
        prompt_v0=prompt_v0,
        variables=variables,
        objectives=objectives,
        known_issues=known_issues,
        eval_size=eval_size,
        model=model,
    )
    total_cost = taxo_cost
    any_cache_hit = taxo_cache

    if taxonomy is None or not taxonomy.categories:
        log.warning(
            "evalgen.taxonomy_failed_falling_back_to_single_call",
            eval_size=eval_size,
        )
        # Fall back to single-call path for resilience — small extra cost vs failing
        return await _generate_single(
            intent=intent,
            prompt_v0=prompt_v0,
            variables=variables,
            objectives=objectives,
            known_issues=known_issues,
            eval_size=eval_size,
            train_ratio=train_ratio,
            model=model,
            split_seed=split_seed,
            declared_var_names=declared_var_names,
        )

    rubric = taxonomy.rubric
    categories = _normalize_categories(taxonomy.categories, eval_size)
    rubric_summary = _rubric_summary(rubric)

    await _emit(
        on_progress,
        "evalgen.taxonomy_completed",
        n_categories=len(categories),
        categories=[{"name": c.name, "target_count": c.target_count} for c in categories],
        cost_usd=taxo_cost,
    )

    # 2. Parallel per-category batches
    batch_results = await asyncio.gather(
        *[
            _generate_category_batch(
                intent=intent,
                prompt_v0=prompt_v0,
                variables=variables,
                category=cat,
                rubric_summary=rubric_summary,
                model=model,
            )
            for cat in categories
        ]
    )

    all_items: list[GeneratedEvalItem] = []
    for cat, (items, cost, cache_hit) in zip(categories, batch_results, strict=True):
        total_cost += cost
        any_cache_hit = any_cache_hit or cache_hit
        all_items.extend(items)
        await _emit(
            on_progress,
            "evalgen.batch_completed",
            category=cat.name,
            target_count=cat.target_count,
            actual_count=len(items),
            cost_usd=cost,
        )

    # 3. Reconcile + dedup
    filtered = _filter_items(all_items, variables)
    deduped = _dedup_items(filtered)
    log.info(
        "evalgen.batched_pool",
        raw=len(all_items),
        after_filter=len(filtered),
        after_dedup=len(deduped),
        target=eval_size,
        declared_vars=declared_var_names,
    )

    # 4. Top-up if short
    if len(deduped) < eval_size:
        gap = eval_size - len(deduped)
        topup_items, topup_cost, topup_cache = await _generate_topup(
            intent=intent,
            prompt_v0=prompt_v0,
            variables=variables,
            n_needed=gap,
            existing_labels=[item.label for item in deduped],
            rubric_summary=rubric_summary,
            model=model,
        )
        total_cost += topup_cost
        any_cache_hit = any_cache_hit or topup_cache
        topup_filtered = _filter_items(topup_items, variables)
        before = len(deduped)
        combined = _dedup_items([*deduped, *topup_filtered])
        deduped = combined
        await _emit(
            on_progress,
            "evalgen.topup_completed",
            requested=gap,
            added=len(deduped) - before,
            cost_usd=topup_cost,
        )

    # 5. Trim to target + split
    final_items = deduped[:eval_size]
    train, holdout = _split(final_items, train_ratio, seed=split_seed)
    return EvalGenResult(
        rubric=rubric,
        train_items=train,
        holdout_items=holdout,
        cost_usd=total_cost,
        cache_hit=any_cache_hit,
    )


# ─── public entry point ──────────────────────────────────────────────────


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
    on_progress: ProgressCallback | None = None,
) -> EvalGenResult:
    """Generate the eval set.

    Dispatches to the single-call path for small eval sets and the batched
    taxonomy+parallel path for larger ones. ``on_progress`` (if provided)
    is invoked with ``(event_type, data_dict)`` at each batched-path
    milestone so the orchestrator can stream SSE events to the UI.
    """
    declared_var_names = [v.name for v in variables]
    if not declared_var_names:
        log.info("evalgen.no_variables_synthesizing_input_slot")
        declared_var_names = ["input"]

    if eval_size <= SINGLE_CALL_MAX:
        log.info("evalgen.path_single", eval_size=eval_size)
        return await _generate_single(
            intent=intent,
            prompt_v0=prompt_v0,
            variables=variables,
            objectives=objectives,
            known_issues=known_issues,
            eval_size=eval_size,
            train_ratio=train_ratio,
            model=model,
            split_seed=split_seed,
            declared_var_names=declared_var_names,
        )

    log.info("evalgen.path_batched", eval_size=eval_size)
    return await _generate_batched(
        intent=intent,
        prompt_v0=prompt_v0,
        variables=variables,
        objectives=objectives,
        known_issues=known_issues,
        eval_size=eval_size,
        train_ratio=train_ratio,
        model=model,
        split_seed=split_seed,
        declared_var_names=declared_var_names,
        on_progress=on_progress,
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

"""Judge agent — scores an actual output against the rubric using LLM-as-judge.

Output is structured: per-criterion scores in [0, 1] with reasoning, plus an overall
explanation. The Judge does not change criterion definitions — that's EvalGen's job.

Aggregation:
  - mean_score: weight-normalized average across criteria the judge actually scored
  - per_objective_score: same average, grouped by RubricCriterion.objective
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.core import providers
from app.core.logging import log
from app.schemas.common import RubricCriterion


class CriterionScore(BaseModel):
    name: str = Field(description="The criterion this score is for. Must match a rubric name.")
    score: float = Field(ge=0.0, le=1.0, description="Score in [0, 1]. Higher is better.")
    reasoning: str = Field(description="One short sentence explaining the score.")


class JudgeOutput(BaseModel):
    scores: list[CriterionScore]
    overall_reasoning: str = Field(
        description="One paragraph synthesizing the per-criterion judgments."
    )


@dataclass
class JudgeResult:
    scores: dict[str, float]  # criterion_name → score
    per_objective: dict[str, float]  # objective → weighted-mean score across its criteria
    mean_score: float
    reasoning: str
    cost_usd: float
    cache_hit: bool


_SYSTEM = """You are a rigorous evaluator. Score the model's output against each criterion of
the rubric independently.

Hard rules:
1. Score each criterion in [0.0, 1.0]:
   - 0.0 = fully fails the criterion
   - 0.5 = partially meets it
   - 1.0 = fully meets it
2. Provide a one-sentence `reasoning` for every criterion you score. Be specific.
3. Only score criteria that appear in the rubric. Do NOT invent new criteria.
4. If an `expected_output` is provided, use it as ground truth for accuracy-style criteria.
   For subjective criteria, judge against the criterion's definition.
5. Be CONSERVATIVE. Don't give 1.0 unless the output unambiguously satisfies the criterion.
6. Output strict JSON matching the schema."""


def _rubric_block(rubric: list[RubricCriterion]) -> str:
    lines = []
    for c in rubric:
        obj = f" [objective: {c.objective}]" if c.objective else ""
        lines.append(f"- {c.name} (weight {c.weight}){obj}: {c.definition}")
    return "\n".join(lines)


def _build_user_message(
    *,
    rubric: list[RubricCriterion],
    rendered_input: str,
    expected_output: str | None,
    actual_output: str,
) -> str:
    expected_block = (
        f"\nExpected output (ground truth):\n{expected_output}\n" if expected_output else ""
    )
    return (
        f"Rubric:\n{_rubric_block(rubric)}\n\n"
        f"Model input (rendered prompt as sent):\n{rendered_input}\n"
        f"{expected_block}"
        f"\nModel output (what we are scoring):\n{actual_output}\n\n"
        "Score each criterion now. Output strict JSON."
    )


def _aggregate(
    raw_scores: list[CriterionScore],
    rubric: list[RubricCriterion],
) -> tuple[dict[str, float], dict[str, float], float]:
    by_name = {c.name: c for c in rubric}
    scores: dict[str, float] = {}
    for s in raw_scores:
        if s.name not in by_name:
            continue  # spurious criterion — drop
        scores[s.name] = max(0.0, min(1.0, s.score))

    # Weighted mean across criteria actually scored
    total_weight = sum(by_name[name].weight for name in scores)
    if total_weight <= 0:
        mean = 0.0
    else:
        mean = sum(by_name[name].weight * v for name, v in scores.items()) / total_weight

    # Per-objective weighted means
    by_objective: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for name, v in scores.items():
        crit = by_name[name]
        if crit.objective:
            by_objective[crit.objective].append((crit.weight, v))
    per_obj: dict[str, float] = {}
    for obj, items in by_objective.items():
        tw = sum(w for w, _ in items)
        per_obj[obj] = (sum(w * v for w, v in items) / tw) if tw > 0 else 0.0

    return scores, per_obj, mean


async def judge(
    *,
    rubric: list[RubricCriterion],
    rendered_input: str,
    actual_output: str,
    expected_output: str | None = None,
    model: str,
    temperature: float = 0.0,
) -> JudgeResult:
    if not rubric:
        return JudgeResult(
            scores={},
            per_objective={},
            mean_score=0.0,
            reasoning="(no rubric)",
            cost_usd=0.0,
            cache_hit=False,
        )

    result = await providers.complete(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _build_user_message(
                    rubric=rubric,
                    rendered_input=rendered_input,
                    expected_output=expected_output,
                    actual_output=actual_output,
                ),
            },
        ],
        response_format=JudgeOutput,
        temperature=temperature,
    )

    if isinstance(result.parsed, JudgeOutput):
        scores, per_obj, mean = _aggregate(result.parsed.scores, rubric)
        reasoning = result.parsed.overall_reasoning
    else:
        log.warning("judge.parse_failed_zero_scoring")
        scores = {c.name: 0.0 for c in rubric}
        per_obj = {}
        mean = 0.0
        reasoning = "(judge LLM returned malformed JSON; defaulted to 0.0)"

    return JudgeResult(
        scores=scores,
        per_objective=per_obj,
        mean_score=mean,
        reasoning=reasoning,
        cost_usd=result.cost_usd,
        cache_hit=result.cache_hit,
    )

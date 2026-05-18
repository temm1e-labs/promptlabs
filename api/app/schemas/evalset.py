from typing import Any

from pydantic import Field

from app.models.evalset import Split
from app.schemas.common import RubricCriterion, TimestampedOut


class EvalItemOut(TimestampedOut):
    eval_set_id: str
    split: Split
    input_vars: dict[str, Any]
    expected_output: str | None
    label: str | None
    item_metadata: dict[str, Any]


class EvalSetOut(TimestampedOut):
    experiment_id: str
    rubric_criteria: list[RubricCriterion]
    items: list[EvalItemOut] = Field(default_factory=list)

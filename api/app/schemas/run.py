from typing import Any

from pydantic import Field

from app.models.evalset import Split
from app.models.run import RunStatus
from app.schemas.common import TimestampedOut


class RunResultOut(TimestampedOut):
    run_id: str
    eval_item_id: str
    actual_output: str
    scores: dict[str, float]
    judge_reasoning: str | None
    mean_score: float
    latency_ms: int
    cost_usd: float
    extras: dict[str, Any]


class RunOut(TimestampedOut):
    experiment_id: str
    prompt_version_id: str
    target_model: str
    split: Split
    iteration: int
    status: RunStatus
    cost_usd: float
    mean_score: float | None
    error: str | None
    results: list[RunResultOut] = Field(default_factory=list)

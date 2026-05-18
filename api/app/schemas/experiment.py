from typing import Any, Literal

from pydantic import Field, model_validator

from app.models.experiment import ExperimentStatus, OptimizationObjective
from app.schemas.common import AgentConfig, ORMModel, TimestampedOut


class ExperimentCreate(ORMModel):
    name: str = Field(min_length=1, max_length=200)
    mode: Literal["cold", "warm"]
    intent: str = Field(min_length=1)
    requirements: str | None = None
    existing_prompt: str | None = None  # required when mode == "warm"
    known_issues: str | None = None  # warm-mode only: user-described failure modes
    optimization_objectives: list[OptimizationObjective] = Field(
        default_factory=lambda: [OptimizationObjective.ACCURACY],
        min_length=1,
        description="What the loop optimizes for. Drives rubric generation and edit strategy.",
    )
    target_models: list[str] = Field(min_length=1)
    agent_config: AgentConfig = Field(default_factory=AgentConfig)
    budget_usd: float = Field(default=5.0, gt=0)
    max_iterations: int = Field(default=10, ge=1, le=50)
    eval_size: int = Field(default=30, ge=4, le=200)
    train_ratio: float = Field(default=0.7, gt=0, lt=1)

    @model_validator(mode="after")
    def _warm_needs_prompt(self) -> "ExperimentCreate":
        if self.mode == "warm" and not (self.existing_prompt and self.existing_prompt.strip()):
            raise ValueError("warm mode requires existing_prompt")
        if self.mode == "cold" and self.known_issues:
            raise ValueError("known_issues only applies to warm mode")
        return self


class ExperimentOut(TimestampedOut):
    project_id: str
    name: str
    intent: str
    requirements: str | None
    known_issues: str | None
    optimization_objectives: list[OptimizationObjective]
    target_models: list[str]
    agent_config: dict[str, Any]
    budget_usd: float
    cost_usd: float
    max_iterations: int
    eval_size: int
    train_ratio: float
    current_iteration: int
    accepted_iteration: int | None
    status: ExperimentStatus
    failure_reason: str | None


class ExperimentSummary(ORMModel):
    """Compact form for list views."""

    id: str
    name: str
    intent: str
    status: ExperimentStatus
    current_iteration: int
    cost_usd: float
    budget_usd: float
    target_models: list[str]
    optimization_objectives: list[OptimizationObjective]
    best_score: float | None = None

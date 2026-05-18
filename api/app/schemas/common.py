from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class TimestampedOut(ORMModel):
    id: str
    created_at: datetime
    updated_at: datetime


class AgentConfig(BaseModel):
    """Per-role model identifiers (LiteLLM format). All optional — fall back to default_model."""

    writer_model: str | None = None
    evalgen_model: str | None = None
    judge_model: str | None = None
    optimizer_model: str | None = None
    diversity_judge_model: str | None = None  # opt-in second judge

    extra: dict[str, Any] = Field(default_factory=dict)


class RubricCriterion(BaseModel):
    name: str
    definition: str
    weight: float = 1.0
    objective: str | None = None  # links the criterion to an OptimizationObjective


class PromptVariable(BaseModel):
    """A {{var}} placeholder declared in a prompt template."""

    name: str
    description: str
    example_value: str

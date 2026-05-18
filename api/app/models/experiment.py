from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.evalset import EvalSet
    from app.models.project import Project
    from app.models.prompt import PromptVersion
    from app.models.run import Run


class ExperimentStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    CONVERGED = "converged"
    OVERFIT = "overfit"
    EXHAUSTED = "exhausted"
    FAILED = "failed"
    ACCEPTED = "accepted"


class OptimizationObjective(enum.StrEnum):
    """What the loop is optimizing for. Multi-select on the experiment.

    ACCURACY        — correctness against the rubric (default for every experiment)
    COST            — minimize tokens / dollars
    LATENCY         — minimize time-to-first-token + total response time
    ROBUSTNESS      — absence of bugs / unintended behaviors (hallucination, off-topic,
                      prompt-injection susceptibility, instruction violations)
    FORMAT_ADHERENCE — strict structural conformance (JSON, schema, layout)
    BREVITY         — concise outputs
    TONE            — match a target tone profile
    """

    ACCURACY = "accuracy"
    COST = "cost"
    LATENCY = "latency"
    ROBUSTNESS = "robustness"
    FORMAT_ADHERENCE = "format_adherence"
    BREVITY = "brevity"
    TONE = "tone"


class Experiment(IdMixin, TimestampMixin, Base):
    __tablename__ = "experiments"

    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), kw_only=True
    )
    name: Mapped[str] = mapped_column(String(200), kw_only=True)
    intent: Mapped[str] = mapped_column(Text, kw_only=True)
    requirements: Mapped[str | None] = mapped_column(Text, default=None, kw_only=True)

    # warm-mode only: user-described failure modes the system should fix
    known_issues: Mapped[str | None] = mapped_column(Text, default=None, kw_only=True)

    # list[OptimizationObjective] — what the loop is optimizing for
    optimization_objectives: Mapped[list[str]] = mapped_column(
        JSON, default_factory=lambda: ["accuracy"], kw_only=True
    )

    # multi-provider primitive — list of LiteLLM model strings
    target_models: Mapped[list[str]] = mapped_column(JSON, kw_only=True)

    # {writer_model, evalgen_model, judge_model, optimizer_model}
    agent_config: Mapped[dict[str, Any]] = mapped_column(JSON, kw_only=True)

    budget_usd: Mapped[float] = mapped_column(Float, default=5.0, kw_only=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, kw_only=True)
    max_iterations: Mapped[int] = mapped_column(Integer, default=10, kw_only=True)
    eval_size: Mapped[int] = mapped_column(Integer, default=30, kw_only=True)
    train_ratio: Mapped[float] = mapped_column(Float, default=0.7, kw_only=True)

    current_iteration: Mapped[int] = mapped_column(Integer, default=0, kw_only=True)
    accepted_iteration: Mapped[int | None] = mapped_column(Integer, default=None, kw_only=True)

    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus, native_enum=False, length=20),
        default=ExperimentStatus.PENDING,
        kw_only=True,
    )
    failure_reason: Mapped[str | None] = mapped_column(Text, default=None, kw_only=True)

    project: Mapped[Project] = relationship(back_populates="experiments", init=False)
    prompt_versions: Mapped[list[PromptVersion]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        default_factory=list,
        kw_only=True,
    )
    eval_sets: Mapped[list[EvalSet]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        default_factory=list,
        kw_only=True,
    )
    runs: Mapped[list[Run]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        default_factory=list,
        kw_only=True,
    )

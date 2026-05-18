from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin
from app.models.evalset import Split

if TYPE_CHECKING:
    from app.models.experiment import Experiment


class RunStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Run(IdMixin, TimestampMixin, Base):
    __tablename__ = "runs"

    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="CASCADE"), kw_only=True
    )
    prompt_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("prompt_versions.id", ondelete="CASCADE"), kw_only=True
    )
    target_model: Mapped[str] = mapped_column(String(200), kw_only=True)
    split: Mapped[Split] = mapped_column(Enum(Split, native_enum=False, length=10), kw_only=True)
    iteration: Mapped[int] = mapped_column(Integer, kw_only=True)

    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, length=20),
        default=RunStatus.PENDING,
        kw_only=True,
    )
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, kw_only=True)
    mean_score: Mapped[float | None] = mapped_column(Float, default=None, kw_only=True)
    error: Mapped[str | None] = mapped_column(Text, default=None, kw_only=True)

    experiment: Mapped[Experiment] = relationship(back_populates="runs", init=False)
    results: Mapped[list[RunResult]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        default_factory=list,
        kw_only=True,
    )


class RunResult(IdMixin, TimestampMixin, Base):
    __tablename__ = "run_results"

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("runs.id", ondelete="CASCADE"), kw_only=True
    )
    eval_item_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("eval_items.id", ondelete="CASCADE"), kw_only=True
    )
    actual_output: Mapped[str] = mapped_column(Text, kw_only=True)
    # {criterion_name: 0..1, ...}
    scores: Mapped[dict[str, float]] = mapped_column(JSON, kw_only=True)
    judge_reasoning: Mapped[str | None] = mapped_column(Text, default=None, kw_only=True)
    mean_score: Mapped[float] = mapped_column(Float, kw_only=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, kw_only=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, kw_only=True)
    extras: Mapped[dict[str, Any]] = mapped_column(JSON, default_factory=dict, kw_only=True)

    run: Mapped[Run] = relationship(back_populates="results", init=False)

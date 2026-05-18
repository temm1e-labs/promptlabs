from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.experiment import Experiment


class Split(enum.StrEnum):
    TRAIN = "train"
    HOLDOUT = "holdout"


class EvalSet(IdMixin, TimestampMixin, Base):
    __tablename__ = "eval_sets"

    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="CASCADE"), kw_only=True
    )

    # [{name, definition, weight?}, ...]
    rubric_criteria: Mapped[list[dict[str, Any]]] = mapped_column(JSON, kw_only=True)

    experiment: Mapped[Experiment] = relationship(back_populates="eval_sets", init=False)
    items: Mapped[list[EvalItem]] = relationship(
        back_populates="eval_set",
        cascade="all, delete-orphan",
        default_factory=list,
        kw_only=True,
    )


class EvalItem(IdMixin, TimestampMixin, Base):
    __tablename__ = "eval_items"

    eval_set_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("eval_sets.id", ondelete="CASCADE"), kw_only=True
    )
    split: Mapped[Split] = mapped_column(Enum(Split, native_enum=False, length=10), kw_only=True)
    # variable-name → value map. The prompt template has {{var}} placeholders;
    # at runtime Runner renders against this dict before calling the model.
    input_vars: Mapped[dict[str, Any]] = mapped_column(JSON, kw_only=True)
    expected_output: Mapped[str | None] = mapped_column(Text, default=None, kw_only=True)
    # human-readable summary of the test case (used in failure displays)
    label: Mapped[str | None] = mapped_column(Text, default=None, kw_only=True)
    item_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default_factory=dict, kw_only=True)

    eval_set: Mapped[EvalSet] = relationship(back_populates="items", init=False)

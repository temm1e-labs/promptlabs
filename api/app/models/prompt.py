from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.experiment import Experiment


class PromptSource(enum.StrEnum):
    COLD = "cold"
    WARM = "warm"
    OPTIMIZER = "optimizer"


class PromptVersion(IdMixin, TimestampMixin, Base):
    __tablename__ = "prompt_versions"

    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id", ondelete="CASCADE"), kw_only=True
    )
    iteration: Mapped[int] = mapped_column(Integer, kw_only=True)
    content: Mapped[str] = mapped_column(Text, kw_only=True)
    rationale: Mapped[str | None] = mapped_column(Text, default=None, kw_only=True)

    parent_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("prompt_versions.id", ondelete="SET NULL"),
        default=None,
        kw_only=True,
    )
    source: Mapped[PromptSource] = mapped_column(
        Enum(PromptSource, native_enum=False, length=20), kw_only=True
    )
    diff: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None, kw_only=True)

    experiment: Mapped[Experiment] = relationship(back_populates="prompt_versions", init=False)

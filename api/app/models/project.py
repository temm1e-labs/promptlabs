from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.experiment import Experiment


class Project(IdMixin, TimestampMixin, Base):
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(200), kw_only=True)
    description: Mapped[str | None] = mapped_column(Text, default=None, kw_only=True)

    experiments: Mapped[list[Experiment]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        default_factory=list,
        kw_only=True,
    )

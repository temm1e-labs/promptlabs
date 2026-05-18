import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column


def _uuid7() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class IdMixin(MappedAsDataclass):
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default_factory=_uuid7, kw_only=True
    )


class TimestampMixin(MappedAsDataclass):
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default_factory=_now,
        server_default=func.now(),
        kw_only=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default_factory=_now,
        server_default=func.now(),
        onupdate=_now,
        kw_only=True,
    )

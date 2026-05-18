from pydantic import Field

from app.schemas.common import ORMModel, TimestampedOut


class ProjectCreate(ORMModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None


class ProjectUpdate(ORMModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None


class ProjectOut(TimestampedOut):
    name: str
    description: str | None
    experiment_count: int = 0

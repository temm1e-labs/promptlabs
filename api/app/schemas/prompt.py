from typing import Any

from app.models.prompt import PromptSource
from app.schemas.common import TimestampedOut


class PromptVersionOut(TimestampedOut):
    experiment_id: str
    iteration: int
    content: str
    rationale: str | None
    parent_id: str | None
    source: PromptSource
    diff: dict[str, Any] | None

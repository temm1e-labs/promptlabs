from fastapi import APIRouter

from app.core.model_catalog import DEFAULT_CATALOG, grouped_by_provider
from app.schemas.common import ORMModel

router = APIRouter(prefix="/models", tags=["models"])


class ModelOut(ORMModel):
    id: str
    label: str
    provider: str
    family: str | None = None
    is_default: bool = False


class ModelGroupOut(ORMModel):
    provider: str
    models: list[ModelOut]


@router.get("", response_model=list[ModelOut])
async def list_models() -> list[ModelOut]:
    return [
        ModelOut(
            id=m.id, label=m.label, provider=m.provider, family=m.family, is_default=m.is_default
        )
        for m in DEFAULT_CATALOG
    ]


@router.get("/grouped", response_model=list[ModelGroupOut])
async def list_models_grouped() -> list[ModelGroupOut]:
    return [
        ModelGroupOut(
            provider=provider,
            models=[
                ModelOut(
                    id=m.id,
                    label=m.label,
                    provider=m.provider,
                    family=m.family,
                    is_default=m.is_default,
                )
                for m in models
            ],
        )
        for provider, models in grouped_by_provider().items()
    ]

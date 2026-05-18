import os
from typing import Any

from fastapi import APIRouter

from app.core import cache
from app.core.config import settings

router = APIRouter(prefix="/settings", tags=["settings"])


_PROVIDER_ENV_KEYS = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "aws_bedrock": ["AWS_ACCESS_KEY_ID"],
    "azure": ["AZURE_API_KEY"],
}


@router.get("/status")
async def status() -> dict[str, Any]:
    return {
        "default_model": settings.default_model,
        "providers": {
            name: any(bool(os.getenv(k)) for k in keys)
            for name, keys in _PROVIDER_ENV_KEYS.items()
        },
        "cache_ttl_days": settings.cache_ttl_s / 86400,
        "max_concurrent_requests": settings.max_concurrent_requests,
        "request_timeout_s": settings.request_timeout_s,
        "defaults": {
            "eval_size": settings.default_eval_size,
            "train_ratio": settings.default_train_ratio,
            "max_iterations": settings.default_max_iterations,
            "budget_usd": settings.default_budget_usd,
        },
    }


@router.post("/cache/clear")
async def clear_cache() -> dict[str, int]:
    n = await cache.clear()
    return {"cleared": n}

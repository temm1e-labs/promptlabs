"""Provider layer — the single chokepoint for every LLM call.

Responsibilities:
  - Multi-provider via LiteLLM (140+ providers, OpenAI-compatible interface).
  - Structured output via Pydantic → JSON schema → LiteLLM response_format.
  - Cost tracking via litellm.completion_cost (fallback 0.0).
  - Retry with exponential backoff for transient errors.
  - Content-addressed response cache.
  - Bounded concurrency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core import cache
from app.core.config import settings
from app.core.logging import log

litellm.drop_params = True  # gracefully drop params unsupported by a given provider
litellm.suppress_debug_info = True


@dataclass
class CompletionResult:
    content: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    model: str
    parsed: BaseModel | None = None


_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)


# Errors worth retrying — transient infra issues, not bad requests
_RETRYABLE: tuple[type[BaseException], ...] = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    InternalServerError,
    ServiceUnavailableError,
)


def _schema_hash(response_format: type[BaseModel] | None) -> str | None:
    if response_format is None:
        return None
    schema = response_format.model_json_schema()
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _build_response_format(response_format: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_format.__name__,
            "schema": response_format.model_json_schema(),
            "strict": True,
        },
    }


def _model_max_output_tokens(model: str) -> int | None:
    """Return the model's native max-output tokens, or None if unknown.

    Standard: PromptLabs NEVER imposes an arbitrary output cap. We pass the model's full
    max-output where required by the provider API (e.g., Anthropic) and let it default
    everywhere else.
    """
    try:
        info = litellm.get_model_info(model)
        max_out = info.get("max_output_tokens") or info.get("max_tokens")
        if max_out:
            return int(max_out)
    except Exception as exc:
        log.debug("provider.unknown_model_max_output", model=model, error=str(exc)[:120])
    return None


async def _call_litellm(
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | None,
    temperature: float,
) -> tuple[str, float, int, int]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": settings.request_timeout_s,
    }
    # Pass max_tokens ONLY when the model's native max is known, so providers that
    # require it (Anthropic Messages API) accept the request without artificial caps.
    native_max = _model_max_output_tokens(model)
    if native_max is not None:
        kwargs["max_tokens"] = native_max
    if response_format is not None:
        kwargs["response_format"] = _build_response_format(response_format)

    response = await litellm.acompletion(**kwargs)
    content = response.choices[0].message.content or ""

    try:
        cost_usd = float(litellm.completion_cost(completion_response=response))
    except Exception:
        cost_usd = 0.0

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return content, cost_usd, input_tokens, output_tokens


async def complete(
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | None = None,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> CompletionResult:
    """Single chokepoint for LLM calls.

    If ``response_format`` is given, the response is validated against the
    Pydantic schema and surfaced as ``result.parsed``.
    """
    schema_h = _schema_hash(response_format)
    key = cache.cache_key(model, messages, schema_h)

    if use_cache:
        hit = await cache.get(key)
        if hit is not None:
            log.debug("provider.cache_hit", model=model, key=key[:12])
            parsed: BaseModel | None = None
            if response_format is not None:
                try:
                    parsed = response_format.model_validate_json(hit.response_text)
                except ValidationError:
                    parsed = None
            return CompletionResult(
                content=hit.response_text,
                cost_usd=0.0,  # cache hit incurs no new cost
                input_tokens=hit.input_tokens,
                output_tokens=hit.output_tokens,
                cache_hit=True,
                model=model,
                parsed=parsed,
            )

    async with _semaphore:
        content = ""
        cost_usd = 0.0
        input_tokens = 0
        output_tokens = 0
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                content, cost_usd, input_tokens, output_tokens = await _call_litellm(
                    model, messages, response_format, temperature
                )

    parsed = None
    if response_format is not None:
        try:
            parsed = response_format.model_validate_json(content)
        except ValidationError as e:
            log.warning(
                "provider.parse_failed",
                model=model,
                error=str(e)[:200],
                content_preview=content[:200],
            )

    if use_cache:
        await cache.put(
            key=key,
            model=model,
            response_text=content,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return CompletionResult(
        content=content,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
        model=model,
        parsed=parsed,
    )

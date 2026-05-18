"""Runner agent — executes a prompt template across eval items on a target model.

Render the template with each item's input_vars, dispatch the LLM call with bounded
concurrency, record latency + cost per item. No scoring happens here — that's Judge.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.core import providers, template
from app.core.logging import log


@dataclass
class RunItemInput:
    """A single test case Runner will execute."""

    item_id: str
    input_vars: dict[str, object]


@dataclass
class RunItemResult:
    item_id: str
    rendered_prompt: str
    actual_output: str
    latency_ms: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    error: str | None = None


@dataclass
class RunnerResult:
    items: list[RunItemResult]
    total_cost_usd: float
    n_cache_hits: int
    n_errors: int


ProgressCallback = Callable[[RunItemResult], Awaitable[None]]


async def _run_one(
    *,
    semaphore: asyncio.Semaphore,
    prompt_template: str,
    item: RunItemInput,
    target_model: str,
    temperature: float,
    progress: ProgressCallback | None,
) -> RunItemResult:
    async with semaphore:
        rendered = template.render(prompt_template, item.input_vars)
        start = time.perf_counter()
        try:
            result = await providers.complete(
                model=target_model,
                messages=[{"role": "user", "content": rendered}],
                temperature=temperature,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            item_result = RunItemResult(
                item_id=item.item_id,
                rendered_prompt=rendered,
                actual_output=result.content,
                latency_ms=latency_ms,
                cost_usd=result.cost_usd,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_hit=result.cache_hit,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log.warning(
                "runner.item_failed",
                item_id=item.item_id,
                target_model=target_model,
                error=str(exc)[:200],
            )
            item_result = RunItemResult(
                item_id=item.item_id,
                rendered_prompt=rendered,
                actual_output="",
                latency_ms=latency_ms,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                cache_hit=False,
                error=str(exc),
            )

    if progress is not None:
        await progress(item_result)
    return item_result


async def run(
    *,
    prompt_template: str,
    items: list[RunItemInput],
    target_model: str,
    max_concurrency: int = 8,
    temperature: float = 0.7,
    progress_callback: ProgressCallback | None = None,
) -> RunnerResult:
    """Execute `prompt_template` against each item on `target_model`.

    Returns per-item results in input order, plus aggregates.
    Individual item errors are captured (not raised) so a single failure doesn't poison the run.
    """
    if not items:
        return RunnerResult(items=[], total_cost_usd=0.0, n_cache_hits=0, n_errors=0)

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _run_one(
            semaphore=semaphore,
            prompt_template=prompt_template,
            item=item,
            target_model=target_model,
            temperature=temperature,
            progress=progress_callback,
        )
        for item in items
    ]
    results = await asyncio.gather(*tasks)

    total_cost = sum(r.cost_usd for r in results)
    n_hits = sum(1 for r in results if r.cache_hit)
    n_errors = sum(1 for r in results if r.error is not None)
    return RunnerResult(
        items=results,
        total_cost_usd=total_cost,
        n_cache_hits=n_hits,
        n_errors=n_errors,
    )

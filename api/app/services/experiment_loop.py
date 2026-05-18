"""Experiment loop orchestrator — the closed loop.

Phases:
  1. Writer (cold | warm) → v0 PromptVersion
  2. EvalGen → EvalSet (rubric + items, split into train/holdout)
  3. Iterate up to max_iterations:
     a. For each target_model: Runner on TRAIN → Judge → persist Run + RunResults
     b. Aggregate; if budget exhausted → stop
     c. Optimizer → diff → new PromptVersion
     d. For each target_model: Runner on HOLDOUT (with the new version) → Judge → persist
     e. Convergence checks (train plateau OR holdout declining)
  4. Mark final status
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents import evalgen as evalgen_agent
from app.agents import judge as judge_agent
from app.agents import optimizer as optimizer_agent
from app.agents import runner as runner_agent
from app.agents import writer as writer_agent
from app.agents.writer import WriterResult
from app.core.config import settings
from app.core.db import SessionLocal
from app.core.logging import log
from app.core.sse import Event, bus
from app.models.evalset import EvalItem, EvalSet, Split
from app.models.experiment import Experiment, ExperimentStatus
from app.models.prompt import PromptSource, PromptVersion
from app.models.run import Run, RunResult, RunStatus
from app.schemas.common import PromptVariable

# ─── helpers ──────────────────────────────────────────────────────────────


async def _emit(experiment_id: str, type_: str, **data: Any) -> None:
    await bus.publish(experiment_id, Event(type=type_, data=data))


def _agent_model(experiment: Experiment, role: str) -> str:
    cfg = experiment.agent_config or {}
    return cfg.get(f"{role}_model") or settings.default_model


async def _accumulate_cost(db: AsyncSession, experiment: Experiment, delta: float) -> bool:
    """Add delta to experiment.cost_usd, persist, return True if STILL within budget."""
    experiment.cost_usd = float(experiment.cost_usd) + float(delta)
    await db.flush()
    return experiment.cost_usd < float(experiment.budget_usd)


async def _check_cancelled(db: AsyncSession, experiment_id: str) -> bool:
    """Re-query the experiment's status — true if user requested a cancel."""
    fresh = await db.execute(
        select(Experiment.status).where(Experiment.id == experiment_id)
    )
    current_status = fresh.scalar_one_or_none()
    return current_status == ExperimentStatus.CANCELLED


def _declare_user_input_if_no_variables(
    writer_result: WriterResult,
) -> tuple[WriterResult, bool]:
    """When Writer's prompt has no {{var}} placeholders, declare a virtual `user_input`
    variable. This DOES NOT modify the prompt text — the prompt is preserved verbatim.

    The Runner detects placeholder-less prompts at execution time and sends them as a
    [system: prompt, user: user_input] chat structure. This cleanly supports full
    agent system prompts (the common warm-mode case) without polluting the prompt itself.

    Returns (writer_result, was_declared).
    """
    if writer_result.output.variables or "{{" in writer_result.output.prompt:
        return writer_result, False

    new_output = writer_result.output.model_copy(
        update={
            "variables": [
                PromptVariable(
                    name="user_input",
                    description="The user message / query to test the prompt against.",
                    example_value="Hello, I need help with my issue.",
                )
            ],
            "rationale": (
                (writer_result.output.rationale or "")
                + "\n\n[PromptLabs: prompt has no {{var}} placeholders → treating as a "
                "system message; EvalGen will generate `user_input` values as test cases.]"
            ).strip(),
        }
    )
    return (
        WriterResult(
            output=new_output, cost_usd=writer_result.cost_usd, cache_hit=writer_result.cache_hit
        ),
        True,
    )


# ─── prompt + evalset persistence ─────────────────────────────────────────


async def _persist_prompt(
    db: AsyncSession,
    experiment: Experiment,
    iteration: int,
    content: str,
    rationale: str,
    source: PromptSource,
    parent_id: str | None,
    diff: dict[str, Any] | None,
) -> PromptVersion:
    version = PromptVersion(
        experiment_id=experiment.id,
        iteration=iteration,
        content=content,
        rationale=rationale,
        source=source,
        parent_id=parent_id,
        diff=diff,
    )
    db.add(version)
    await db.flush()
    return version


async def _persist_evalset(
    db: AsyncSession,
    experiment: Experiment,
    rubric: list[dict[str, Any]],
    train: list[evalgen_agent.GeneratedEvalItem],
    holdout: list[evalgen_agent.GeneratedEvalItem],
) -> EvalSet:
    eval_set = EvalSet(experiment_id=experiment.id, rubric_criteria=rubric)
    db.add(eval_set)
    await db.flush()
    rows: list[EvalItem] = []
    for it in train:
        rows.append(
            EvalItem(
                eval_set_id=eval_set.id,
                split=Split.TRAIN,
                input_vars=it.input_vars,
                expected_output=it.expected_output,
                label=it.label,
                item_metadata={"tags": it.tags},
            )
        )
    for it in holdout:
        rows.append(
            EvalItem(
                eval_set_id=eval_set.id,
                split=Split.HOLDOUT,
                input_vars=it.input_vars,
                expected_output=it.expected_output,
                label=it.label,
                item_metadata={"tags": it.tags},
            )
        )
    db.add_all(rows)
    await db.flush()
    return eval_set


# ─── score + run persistence ──────────────────────────────────────────────


@dataclass
class _ScoredItem:
    eval_item: EvalItem
    run_item: runner_agent.RunItemResult
    judge: judge_agent.JudgeResult


async def _judge_one(
    *,
    eval_item: EvalItem,
    run_item: runner_agent.RunItemResult,
    rubric: list[dict[str, Any]],
    judge_model: str,
) -> _ScoredItem:
    from app.schemas.common import RubricCriterion

    rubric_objs = [RubricCriterion.model_validate(r) for r in rubric]
    judge_result = await judge_agent.judge(
        rubric=rubric_objs,
        rendered_input=run_item.rendered_prompt,
        actual_output=run_item.actual_output,
        expected_output=eval_item.expected_output,
        model=judge_model,
    )
    return _ScoredItem(eval_item=eval_item, run_item=run_item, judge=judge_result)


async def _execute_on_split(
    db: AsyncSession,
    experiment: Experiment,
    prompt_version: PromptVersion,
    target_model: str,
    items: list[EvalItem],
    iteration: int,
    split: Split,
    rubric: list[dict[str, Any]],
    judge_model: str,
) -> tuple[Run, float]:
    """Run the prompt against `items` on `target_model`, judge each, persist a Run."""
    run = Run(
        experiment_id=experiment.id,
        prompt_version_id=prompt_version.id,
        target_model=target_model,
        split=split,
        iteration=iteration,
        status=RunStatus.RUNNING,
    )
    db.add(run)
    await db.flush()
    await _emit(
        experiment.id,
        "run.started",
        run_id=run.id,
        target_model=target_model,
        split=split.value,
        iteration=iteration,
        n_items=len(items),
    )

    runner_result = await runner_agent.run(
        prompt_template=prompt_version.content,
        items=[
            runner_agent.RunItemInput(item_id=item.id, input_vars=item.input_vars)
            for item in items
        ],
        target_model=target_model,
        max_concurrency=settings.max_concurrent_requests,
    )

    await _emit(
        experiment.id,
        "judge.started",
        run_id=run.id,
        target_model=target_model,
        split=split.value,
        iteration=iteration,
        n_items=len(items),
        judge_model=judge_model,
    )

    item_by_id = {item.id: item for item in items}
    scored = await asyncio.gather(
        *[
            _judge_one(
                eval_item=item_by_id[r.item_id],
                run_item=r,
                rubric=rubric,
                judge_model=judge_model,
            )
            for r in runner_result.items
        ]
    )

    # Persist RunResults + accumulate cost
    score_total = 0.0
    n_scored = 0
    total_run_cost = runner_result.total_cost_usd
    total_judge_cost = 0.0
    for s in scored:
        db.add(
            RunResult(
                run_id=run.id,
                eval_item_id=s.eval_item.id,
                actual_output=s.run_item.actual_output,
                scores=s.judge.scores,
                judge_reasoning=s.judge.reasoning,
                mean_score=s.judge.mean_score,
                latency_ms=s.run_item.latency_ms,
                cost_usd=s.run_item.cost_usd + s.judge.cost_usd,
                extras={
                    "per_objective": s.judge.per_objective,
                    "cache_hit": s.run_item.cache_hit,
                    "runner_error": s.run_item.error,
                },
            )
        )
        score_total += s.judge.mean_score
        n_scored += 1
        total_judge_cost += s.judge.cost_usd

    mean = score_total / n_scored if n_scored > 0 else 0.0
    run.cost_usd = total_run_cost + total_judge_cost
    run.mean_score = mean
    run.status = RunStatus.COMPLETED
    await db.flush()
    await _emit(
        experiment.id,
        "run.completed",
        run_id=run.id,
        target_model=target_model,
        split=split.value,
        iteration=iteration,
        mean_score=mean,
        cost_usd=run.cost_usd,
    )
    return run, mean


# ─── convergence + best-version tracking ─────────────────────────────────


def _scaled_threshold(n_items: int, sigma: float = 0.2, z: float = 1.96) -> float:
    """Sample-size-aware threshold.

    SE of an iteration mean over N items with per-item std sigma is sigma/sqrt(N).
    A change of z * SE is the 95% CI half-width — i.e. the minimum change that
    isn't pure sampling noise.

    Default sigma=0.2 reflects empirically-observed per-item std of LLM-judge
    scores in [0, 1]. With N=15 → ~0.10, N=30 → ~0.07, N=60 → ~0.05.
    """
    return z * sigma / math.sqrt(max(1, n_items))


def _train_plateaued(train_means: list[float], n_train: int = 30) -> bool:
    """Plateau when 3-iteration window range is below the per-iteration noise floor."""
    if len(train_means) < 3:
        return False
    delta = _scaled_threshold(n_train)
    window = train_means[-3:]
    return (max(window) - min(window)) < delta


def _overfitting(
    train_means: list[float],
    holdout_means: list[float],
    n_train: int = 30,
    n_holdout: int = 15,
) -> bool:
    """Divergence-based overfit detection.

    Overfit ⇔ train rising AND holdout falling, each by more than its own
    sample-size-aware noise threshold. Co-regression (both down) and co-
    improvement (both up) explicitly do NOT trigger — those aren't overfit,
    those are different phenomena.
    """
    if len(train_means) < 3 or len(holdout_means) < 3:
        return False
    delta_train = train_means[-1] - train_means[-3]
    delta_holdout = holdout_means[-1] - holdout_means[-3]
    tau_train = _scaled_threshold(n_train)
    tau_holdout = _scaled_threshold(n_holdout)
    return delta_train >= tau_train and delta_holdout <= -tau_holdout


# ─── failure sampling for the optimizer ─────────────────────────────────


async def _failure_samples_for_optimizer(
    db: AsyncSession,
    experiment_id: str,
    iteration: int,
    k: int = 8,
) -> list[dict[str, Any]]:
    """Stratified failure sampling — balanced coverage across failing criteria.

    Pass 1: for each rubric criterion C, take the lowest-scoring train item where
            C scored < 0.5 (one exemplar per failing criterion).
    Pass 2: fill remaining slots up to k with lowest-overall failures not yet
            chosen.

    Previous behavior (top-k by mean) often returned 8 duplicates of the same
    failure mode, blinding the Optimizer to other broken criteria.
    """
    stmt = (
        select(RunResult, EvalItem)
        .join(Run, Run.id == RunResult.run_id)
        .join(EvalItem, EvalItem.id == RunResult.eval_item_id)
        .where(
            Run.experiment_id == experiment_id,
            Run.iteration == iteration,
            Run.split == Split.TRAIN,
        )
        .order_by(RunResult.mean_score.asc())
    )
    rows = (await db.execute(stmt)).all()

    # Pass 1: one exemplar per failing criterion (the lowest-scoring one)
    per_criterion: dict[str, tuple[Any, Any]] = {}
    for run_result, eval_item in rows:
        for crit_name, score in (run_result.scores or {}).items():
            if score < 0.5 and crit_name not in per_criterion:
                per_criterion[crit_name] = (run_result, eval_item)

    chosen_ids: set[str] = {rr.id for rr, _ in per_criterion.values()}
    samples: list[dict[str, Any]] = []
    for rr, ei in per_criterion.values():
        failing = [c for c, v in (rr.scores or {}).items() if v < 0.5]
        samples.append(
            {
                "label": ei.label,
                "mean_score": rr.mean_score,
                "failing_criteria": failing,
                "input_vars": ei.input_vars,
                "actual_output": rr.actual_output,
                "reasoning": rr.judge_reasoning,
            }
        )

    # Pass 2: fill remaining slots with lowest-overall failures
    for rr, ei in rows:
        if len(samples) >= k:
            break
        if rr.id in chosen_ids:
            continue
        failing = [c for c, v in (rr.scores or {}).items() if v < 0.5]
        samples.append(
            {
                "label": ei.label,
                "mean_score": rr.mean_score,
                "failing_criteria": failing,
                "input_vars": ei.input_vars,
                "actual_output": rr.actual_output,
                "reasoning": rr.judge_reasoning,
            }
        )
    return samples[:k]


# ─── the orchestrator ───────────────────────────────────────────────────


async def _reload(db: AsyncSession, experiment_id: str) -> Experiment | None:
    stmt = (
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .options(
            selectinload(Experiment.prompt_versions),
            selectinload(Experiment.eval_sets).selectinload(EvalSet.items),
        )
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _set_status(
    db: AsyncSession,
    experiment: Experiment,
    status: ExperimentStatus,
    failure_reason: str | None = None,
) -> None:
    experiment.status = status
    if failure_reason is not None:
        experiment.failure_reason = failure_reason
    await db.commit()


async def run_experiment(
    experiment_id: str,
    *,
    mode: str,
    existing_prompt: str | None,
    on_finished: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Background task entrypoint. Drives one experiment from intent → final prompt."""
    async with SessionLocal() as db:
        experiment = await _reload(db, experiment_id)
        if experiment is None:
            log.warning("loop.experiment_not_found", id=experiment_id)
            return

        try:
            await _set_status(db, experiment, ExperimentStatus.RUNNING)
            await _emit(experiment_id, "loop.started")

            # ── Phase 1: Writer ────────────────────────────────────────
            writer_model = _agent_model(experiment, "writer")
            await _emit(
                experiment_id,
                "writer.started",
                model=writer_model,
                mode=mode,
            )
            if mode == "warm":
                writer_result = await writer_agent.write_warm(
                    existing_prompt=existing_prompt or "",
                    known_issues=experiment.known_issues,
                    objectives=list(experiment.optimization_objectives),
                    model=writer_model,
                )
                source = PromptSource.WARM
            else:
                writer_result = await writer_agent.write_cold(
                    intent=experiment.intent,
                    requirements=experiment.requirements,
                    objectives=list(experiment.optimization_objectives),
                    model=writer_model,
                )
                source = PromptSource.COLD

            writer_result, was_wrapped = _declare_user_input_if_no_variables(writer_result)
            if was_wrapped:
                await _emit(
                    experiment_id,
                    "writer.auto_wrapped",
                    reason=(
                        "prompt has no {{var}} placeholders — runner will send as "
                        "system+user chat"
                    ),
                    added_variable="user_input",
                )

            ok = await _accumulate_cost(db, experiment, writer_result.cost_usd)
            v0 = await _persist_prompt(
                db,
                experiment,
                iteration=0,
                content=writer_result.output.prompt,
                rationale=writer_result.output.rationale,
                source=source,
                parent_id=None,
                diff=None,
            )
            await db.commit()
            await _emit(
                experiment_id,
                "writer.completed",
                prompt_version_id=v0.id,
                variables=[v.model_dump() for v in writer_result.output.variables],
                cost_usd=writer_result.cost_usd,
            )
            if not ok:
                await _set_status(db, experiment, ExperimentStatus.EXHAUSTED, "budget after writer")
                await _emit(experiment_id, "loop.finished", status=experiment.status.value)
                return

            # ── Phase 2: EvalGen ────────────────────────────────────────
            evalgen_model = _agent_model(experiment, "evalgen")
            # Deterministic split seed: same experiment → same split.
            split_seed = int(
                hashlib.sha256(experiment_id.encode()).hexdigest()[:8], 16
            )
            await _emit(
                experiment_id,
                "evalgen.started",
                model=evalgen_model,
                eval_size=experiment.eval_size,
                train_ratio=experiment.train_ratio,
                mode=(
                    "batched"
                    if experiment.eval_size > evalgen_agent.SINGLE_CALL_MAX
                    else "single"
                ),
            )

            async def _evalgen_progress(event_type: str, data: dict[str, Any]) -> None:
                await _emit(experiment_id, event_type, **data)

            evalgen_result = await evalgen_agent.generate(
                intent=experiment.intent,
                prompt_v0=writer_result.output.prompt,
                variables=list(writer_result.output.variables),
                objectives=list(experiment.optimization_objectives),
                known_issues=experiment.known_issues,
                eval_size=experiment.eval_size,
                train_ratio=experiment.train_ratio,
                model=evalgen_model,
                split_seed=split_seed,
                on_progress=_evalgen_progress,
            )
            ok = await _accumulate_cost(db, experiment, evalgen_result.cost_usd)
            rubric_dicts = [c.model_dump() for c in evalgen_result.rubric]
            eval_set = await _persist_evalset(
                db,
                experiment,
                rubric_dicts,
                evalgen_result.train_items,
                evalgen_result.holdout_items,
            )
            await db.commit()
            await _emit(
                experiment_id,
                "evalgen.completed",
                eval_set_id=eval_set.id,
                rubric=rubric_dicts,
                n_train=len(evalgen_result.train_items),
                n_holdout=len(evalgen_result.holdout_items),
                cost_usd=evalgen_result.cost_usd,
            )
            if not ok:
                await _set_status(
                    db, experiment, ExperimentStatus.EXHAUSTED, "budget after evalgen"
                )
                await _emit(experiment_id, "loop.finished", status=experiment.status.value)
                return

            # Query EvalItems directly by FK — avoids stale relationship collections
            # after add_all() within the same session.
            items_stmt = (
                select(EvalItem)
                .join(EvalSet, EvalSet.id == EvalItem.eval_set_id)
                .where(EvalSet.experiment_id == experiment_id)
                .order_by(EvalItem.created_at.asc())
            )
            all_items = list((await db.execute(items_stmt)).scalars().all())
            if not all_items:
                await _set_status(
                    db,
                    experiment,
                    ExperimentStatus.FAILED,
                    "evalgen produced no usable items (variable-name mismatch?)",
                )
                await _emit(
                    experiment_id,
                    "loop.failed",
                    error="evalgen produced no usable items",
                )
                return
            train_items = [i for i in all_items if i.split == Split.TRAIN]
            holdout_items = [i for i in all_items if i.split == Split.HOLDOUT]
            log.info(
                "loop.eval_items_loaded",
                n_train=len(train_items),
                n_holdout=len(holdout_items),
            )

            # ── Phase 3: Iterate ────────────────────────────────────────
            train_means: list[float] = []
            holdout_means: list[float] = []
            current_prompt_version = v0

            judge_model = _agent_model(experiment, "judge")
            optimizer_model = _agent_model(experiment, "optimizer")
            final_status: ExperimentStatus = ExperimentStatus.EXHAUSTED

            for iteration in range(1, experiment.max_iterations + 1):
                if await _check_cancelled(db, experiment_id):
                    await _emit(experiment_id, "loop.finished", status="cancelled")
                    return
                experiment.current_iteration = iteration
                await db.commit()
                await _emit(experiment_id, "iteration.started", iteration=iteration)

                # 3a. Train pass — parallel fan-out across target models
                train_results = await asyncio.gather(
                    *[
                        _execute_on_split(
                            db,
                            experiment,
                            prompt_version=current_prompt_version,
                            target_model=target_model,
                            items=train_items,
                            iteration=iteration,
                            split=Split.TRAIN,
                            rubric=rubric_dicts,
                            judge_model=judge_model,
                        )
                        for target_model in experiment.target_models
                    ]
                )
                iter_train_scores = [mean for _run, mean in train_results]
                for _run, _ in train_results:
                    ok = await _accumulate_cost(db, experiment, _run.cost_usd)
                if not ok:
                    await _set_status(
                        db, experiment, ExperimentStatus.EXHAUSTED, "budget during train pass"
                    )
                    await _emit(experiment_id, "loop.finished", status=experiment.status.value)
                    return
                train_mean_iter = statistics.fmean(iter_train_scores) if iter_train_scores else 0.0
                train_means.append(train_mean_iter)

                # 3b. Optimize
                samples = await _failure_samples_for_optimizer(db, experiment_id, iteration)
                await _emit(
                    experiment_id,
                    "optimizer.started",
                    iteration=iteration,
                    model=optimizer_model,
                    n_failure_samples=len(samples),
                )
                opt_result = await optimizer_agent.optimize(
                    current_prompt=current_prompt_version.content,
                    iteration=iteration,
                    rubric=rubric_dicts,
                    failure_samples=samples,
                    objectives=list(experiment.optimization_objectives),
                    known_issues=experiment.known_issues,
                    model=optimizer_model,
                )
                ok = await _accumulate_cost(db, experiment, opt_result.cost_usd)

                if opt_result.new_prompt == current_prompt_version.content:
                    await _emit(
                        experiment_id,
                        "optimizer.noop",
                        iteration=iteration,
                        skip_reasons=opt_result.skip_reasons,
                        cost_usd=opt_result.cost_usd,
                    )
                    final_status = ExperimentStatus.CONVERGED
                    break

                new_version = await _persist_prompt(
                    db,
                    experiment,
                    iteration=iteration,
                    content=opt_result.new_prompt,
                    rationale=opt_result.summary,
                    source=PromptSource.OPTIMIZER,
                    parent_id=current_prompt_version.id,
                    diff={
                        "edits": [e.model_dump() for e in opt_result.edits],
                        "applied": opt_result.edits_applied,
                        "skipped": opt_result.edits_skipped,
                        "skip_reasons": opt_result.skip_reasons,
                    },
                )
                await db.commit()
                await _emit(
                    experiment_id,
                    "optimizer.completed",
                    iteration=iteration,
                    new_prompt_version_id=new_version.id,
                    edits_applied=opt_result.edits_applied,
                    edits_skipped=opt_result.edits_skipped,
                    cost_usd=opt_result.cost_usd,
                )
                if not ok:
                    await _set_status(
                        db, experiment, ExperimentStatus.EXHAUSTED, "budget after optimizer"
                    )
                    await _emit(experiment_id, "loop.finished", status=experiment.status.value)
                    return

                # 3c. Holdout pass on the new version — parallel fan-out
                holdout_results = await asyncio.gather(
                    *[
                        _execute_on_split(
                            db,
                            experiment,
                            prompt_version=new_version,
                            target_model=target_model,
                            items=holdout_items,
                            iteration=iteration,
                            split=Split.HOLDOUT,
                            rubric=rubric_dicts,
                            judge_model=judge_model,
                        )
                        for target_model in experiment.target_models
                    ]
                )
                iter_holdout_scores = [mean for _run, mean in holdout_results]
                for _run, _ in holdout_results:
                    ok = await _accumulate_cost(db, experiment, _run.cost_usd)
                if not ok:
                    await _set_status(
                        db, experiment, ExperimentStatus.EXHAUSTED, "budget during holdout"
                    )
                    await _emit(experiment_id, "loop.finished", status=experiment.status.value)
                    return
                holdout_mean_iter = (
                    statistics.fmean(iter_holdout_scores) if iter_holdout_scores else 0.0
                )
                holdout_means.append(holdout_mean_iter)

                await _emit(
                    experiment_id,
                    "iteration.completed",
                    iteration=iteration,
                    train_mean=train_mean_iter,
                    holdout_mean=holdout_mean_iter,
                    train_holdout_gap=train_mean_iter - holdout_mean_iter,
                    cost_so_far=experiment.cost_usd,
                )

                current_prompt_version = new_version

                # 3d. Convergence checks (sample-size aware)
                if _overfitting(
                    train_means,
                    holdout_means,
                    n_train=len(train_items),
                    n_holdout=len(holdout_items),
                ):
                    final_status = ExperimentStatus.OVERFIT
                    break
                if _train_plateaued(train_means, n_train=len(train_items)):
                    final_status = ExperimentStatus.CONVERGED
                    break
            else:
                # max_iterations reached without break
                final_status = ExperimentStatus.EXHAUSTED

            await _set_status(db, experiment, final_status)
            await _emit(
                experiment_id,
                "loop.finished",
                status=final_status.value,
                train_means=train_means,
                holdout_means=holdout_means,
                final_iteration=experiment.current_iteration,
            )
        except Exception as exc:
            log.exception("loop.failed", id=experiment_id)
            await _set_status(db, experiment, ExperimentStatus.FAILED, str(exc))
            await _emit(experiment_id, "loop.failed", error=str(exc))
        finally:
            if on_finished is not None:
                try:
                    await on_finished()
                except Exception as exc:
                    log.warning("loop.on_finished_callback_failed", error=str(exc)[:200])

import asyncio
import json
import math
import statistics
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.db import get_session
from app.core.sse import bus
from app.models.evalset import EvalSet, Split
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project
from app.models.prompt import PromptVersion
from app.models.run import Run, RunResult
from app.schemas.experiment import ExperimentCreate, ExperimentOut, ExperimentSummary
from app.services import experiment_loop


def _stats(values: list[float]) -> dict[str, float | int | None]:
    """Per-iteration sample statistics. Returns mean, std, n, and 95% CI half-width.

    Std uses Bessel's correction (n-1 denominator). CI half-width is the
    standard error of the mean times 1.96 (~95% normal approximation).
    """
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0, "ci_half_width": None, "se": None}
    if n == 1:
        return {
            "mean": values[0],
            "std": 0.0,
            "n": 1,
            "ci_half_width": None,
            "se": None,
        }
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    se = std / math.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "se": se,
        "ci_half_width": 1.96 * se,
    }

router = APIRouter(tags=["experiments"])


def _to_out(e: Experiment) -> ExperimentOut:
    return ExperimentOut(
        id=e.id,
        project_id=e.project_id,
        name=e.name,
        intent=e.intent,
        requirements=e.requirements,
        known_issues=e.known_issues,
        optimization_objectives=list(e.optimization_objectives),
        target_models=list(e.target_models),
        agent_config=e.agent_config,
        budget_usd=e.budget_usd,
        cost_usd=e.cost_usd,
        max_iterations=e.max_iterations,
        eval_size=e.eval_size,
        train_ratio=e.train_ratio,
        current_iteration=e.current_iteration,
        accepted_iteration=e.accepted_iteration,
        status=e.status,
        failure_reason=e.failure_reason,
        created_at=e.created_at,
        updated_at=e.updated_at,
    )


def _to_summary(e: Experiment, best_score: float | None) -> ExperimentSummary:
    return ExperimentSummary(
        id=e.id,
        name=e.name,
        intent=e.intent,
        status=e.status,
        current_iteration=e.current_iteration,
        cost_usd=e.cost_usd,
        budget_usd=e.budget_usd,
        target_models=list(e.target_models),
        optimization_objectives=list(e.optimization_objectives),
        best_score=best_score,
    )


@router.post(
    "/projects/{project_id}/experiments",
    response_model=ExperimentOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_experiment(
    project_id: str,
    body: ExperimentCreate,
    db: AsyncSession = Depends(get_session),
) -> ExperimentOut:
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="project_not_found")

    experiment = Experiment(
        project_id=project_id,
        name=body.name,
        intent=body.intent,
        requirements=body.requirements,
        known_issues=body.known_issues,
        optimization_objectives=[o.value for o in body.optimization_objectives],
        target_models=list(body.target_models),
        agent_config=body.agent_config.model_dump(exclude_none=False),
        budget_usd=body.budget_usd,
        max_iterations=body.max_iterations,
        eval_size=body.eval_size,
        train_ratio=body.train_ratio,
        status=ExperimentStatus.PENDING,
    )
    db.add(experiment)
    await db.flush()
    out = _to_out(experiment)
    await db.commit()

    asyncio.create_task(  # noqa: RUF006 — background fire-and-forget
        experiment_loop.run_experiment(
            experiment.id,
            mode=body.mode,
            existing_prompt=body.existing_prompt,
        )
    )

    return out


@router.get("/projects/{project_id}/experiments", response_model=list[ExperimentSummary])
async def list_experiments_for_project(
    project_id: str,
    db: AsyncSession = Depends(get_session),
) -> list[ExperimentSummary]:
    stmt = (
        select(Experiment)
        .where(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc())
    )
    experiments = (await db.execute(stmt)).scalars().all()
    # Best score per experiment from HOLDOUT runs
    summaries = []
    for e in experiments:
        best_stmt = select(Run.mean_score).where(
            Run.experiment_id == e.id,
            Run.split == "holdout",
        )
        scores = [s for (s,) in (await db.execute(best_stmt)).all() if s is not None]
        best = max(scores) if scores else None
        summaries.append(_to_summary(e, best))
    return summaries


@router.get("/experiments", response_model=list[dict[str, Any]])
async def list_all_experiments(
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """All experiments across projects, most recent first. Includes project name."""
    from app.models.project import Project

    stmt = (
        select(Experiment, Project.name)
        .join(Project, Project.id == Experiment.project_id)
        .order_by(Experiment.created_at.desc())
        .limit(100)
    )
    rows = (await db.execute(stmt)).all()
    out: list[dict[str, Any]] = []
    for e, project_name in rows:
        best_stmt = select(Run.mean_score).where(
            Run.experiment_id == e.id,
            Run.split == "holdout",
        )
        scores = [s for (s,) in (await db.execute(best_stmt)).all() if s is not None]
        best = max(scores) if scores else None
        out.append(
            {
                "id": e.id,
                "project_id": e.project_id,
                "project_name": project_name,
                "name": e.name,
                "intent": e.intent,
                "status": e.status.value,
                "current_iteration": e.current_iteration,
                "cost_usd": e.cost_usd,
                "budget_usd": e.budget_usd,
                "target_models": list(e.target_models),
                "optimization_objectives": list(e.optimization_objectives),
                "best_score": best,
                "created_at": e.created_at.isoformat(),
            }
        )
    return out


@router.get("/experiments/{experiment_id}", response_model=ExperimentOut)
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_session),
) -> ExperimentOut:
    stmt = (
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .options(
            selectinload(Experiment.prompt_versions),
            selectinload(Experiment.eval_sets).selectinload(EvalSet.items),
            selectinload(Experiment.runs),
        )
    )
    e = (await db.execute(stmt)).scalar_one_or_none()
    if e is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="experiment_not_found")
    return _to_out(e)


@router.post("/experiments/{experiment_id}/accept", response_model=ExperimentOut)
async def accept_iteration(
    experiment_id: str,
    iteration: int,
    db: AsyncSession = Depends(get_session),
) -> ExperimentOut:
    e = await db.get(Experiment, experiment_id)
    if e is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="experiment_not_found")
    e.accepted_iteration = iteration
    e.status = ExperimentStatus.ACCEPTED
    await db.flush()
    return _to_out(e)


@router.get("/experiments/{experiment_id}/iteration-stats", response_model=dict[str, Any])
async def get_iteration_stats(
    experiment_id: str,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Per-iteration sample statistics: mean ± SE + 95% CI half-width.

    The Run.mean_score that the orchestrator persists is a single number per
    (iteration, split, target_model). This endpoint aggregates the per-item
    RunResult.mean_score back into a sampling distribution and reports the
    proper sampling stats. That's what the UI needs to draw error bars.
    """
    stmt = (
        select(RunResult, Run.iteration, Run.split, Run.prompt_version_id)
        .join(Run, Run.id == RunResult.run_id)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.iteration.asc())
    )
    rows = (await db.execute(stmt)).all()

    # Group per-item scores by (iteration, split)
    grouped: dict[tuple[int, str], list[float]] = {}
    version_by_iter: dict[int, str] = {}
    for run_result, iteration, split, prompt_version_id in rows:
        key = (iteration, split.value)
        grouped.setdefault(key, []).append(run_result.mean_score)
        version_by_iter.setdefault(iteration, prompt_version_id)

    iterations: list[dict[str, Any]] = []
    for iter_n in sorted({i for i, _ in grouped.keys()}):
        train_scores = grouped.get((iter_n, "train"), [])
        holdout_scores = grouped.get((iter_n, "holdout"), [])
        iterations.append(
            {
                "iteration": iter_n,
                "prompt_version_id": version_by_iter.get(iter_n),
                "train": _stats(train_scores),
                "holdout": _stats(holdout_scores),
            }
        )

    return {"iterations": iterations}


@router.get("/experiments/{experiment_id}/best-prompt", response_model=dict[str, Any])
async def get_best_prompt(
    experiment_id: str,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return the prompt + metadata that downstream services should use.

    Priority:
      1. The iteration the user explicitly accepted (`accepted_iteration`).
      2. The iteration with the highest mean holdout score.
      3. The most recent iteration if no holdout scores exist yet (e.g. mid-run).

    Designed for production consumption: stable JSON shape, no SQLAlchemy types leaked.
    """
    exp = await db.get(Experiment, experiment_id)
    if exp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="experiment_not_found")

    version: PromptVersion | None = None
    selection_reason: str

    if exp.accepted_iteration is not None:
        accepted_stmt = (
            select(PromptVersion)
            .where(
                PromptVersion.experiment_id == experiment_id,
                PromptVersion.iteration == exp.accepted_iteration,
            )
            .limit(1)
        )
        version = (await db.execute(accepted_stmt)).scalar_one_or_none()
        selection_reason = "accepted"

    if version is None:
        # Lower-confidence-bound selection: argmax over (mean - 1.96 * SE).
        # This prefers robust prompts over lucky-streak high-variance ones.
        # On small holdout sets the difference matters: a v with 0.92 ± 0.10
        # has LCB ~0.72; a v with 0.91 ± 0.04 has LCB ~0.83. The latter is the
        # safer production choice even though its mean is lower.
        per_item_stmt = (
            select(RunResult.mean_score, Run.iteration, Run.prompt_version_id)
            .join(Run, Run.id == RunResult.run_id)
            .where(Run.experiment_id == experiment_id, Run.split == Split.HOLDOUT)
        )
        per_item_rows = (await db.execute(per_item_stmt)).all()
        # Group per-item holdout scores by iteration
        scores_by_iter: dict[int, list[float]] = {}
        version_by_iter: dict[int, str] = {}
        for score, iteration, prompt_version_id in per_item_rows:
            if score is not None:
                scores_by_iter.setdefault(iteration, []).append(score)
                version_by_iter[iteration] = prompt_version_id

        best_iter: int | None = None
        best_lcb: float | None = None
        best_mean: float | None = None
        best_se: float | None = None
        for iteration, scores in scores_by_iter.items():
            stats = _stats(scores)
            mean = stats["mean"]
            n = stats["n"]
            std = stats["std"]
            if mean is None or n is None or std is None or n < 1:
                continue
            n_int = int(n)
            se = float(std) / math.sqrt(n_int) if n_int > 0 else 0.0
            lcb = float(mean) - 1.96 * se
            if best_lcb is None or lcb > best_lcb:
                best_lcb = lcb
                best_iter = iteration
                best_mean = float(mean)
                best_se = se

        if best_iter is not None:
            version = await db.get(PromptVersion, version_by_iter[best_iter])
            mean_str = f"{best_mean:.3f}" if best_mean is not None else "?"
            se_str = f"{best_se:.3f}" if best_se is not None else "?"
            lcb_str = f"{best_lcb:.3f}" if best_lcb is not None else "?"
            selection_reason = (
                f"best_lcb (iter {best_iter}, mean={mean_str}, "
                f"se={se_str}, lcb={lcb_str})"
            )
        else:
            # Fallback: most recent prompt version
            recent_stmt = (
                select(PromptVersion)
                .where(PromptVersion.experiment_id == experiment_id)
                .order_by(PromptVersion.iteration.desc())
                .limit(1)
            )
            version = (await db.execute(recent_stmt)).scalar_one_or_none()
            selection_reason = "most_recent (no holdout scores yet)"

    if version is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail="no_prompt_version_for_experiment"
        )

    return {
        "experiment_id": experiment_id,
        "experiment_name": exp.name,
        "iteration": version.iteration,
        "prompt": version.content,
        "source": version.source.value,
        "selection_reason": selection_reason,
        "experiment_status": exp.status.value,
        "version_id": version.id,
        "created_at": version.created_at.isoformat(),
    }


@router.post("/experiments/{experiment_id}/cancel", response_model=ExperimentOut)
async def cancel_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_session),
) -> ExperimentOut:
    """Cooperative cancel: flip status to `cancelled`. The orchestrator checks this
    at each iteration boundary and exits cleanly (in-flight LLM calls finish first)."""
    e = await db.get(Experiment, experiment_id)
    if e is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="experiment_not_found")
    terminal = {
        ExperimentStatus.CONVERGED,
        ExperimentStatus.OVERFIT,
        ExperimentStatus.EXHAUSTED,
        ExperimentStatus.FAILED,
        ExperimentStatus.ACCEPTED,
        ExperimentStatus.CANCELLED,
    }
    if e.status in terminal:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"already {e.status.value}")
    e.status = ExperimentStatus.CANCELLED
    e.failure_reason = "cancelled by user"
    await db.flush()
    return _to_out(e)


@router.delete("/experiments/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_session),
) -> None:
    e = await db.get(Experiment, experiment_id)
    if e is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="experiment_not_found")
    await db.delete(e)


@router.get("/experiments/{experiment_id}/prompt-versions", response_model=list[dict[str, Any]])
async def list_prompt_versions(
    experiment_id: str,
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    stmt = (
        select(PromptVersion)
        .where(PromptVersion.experiment_id == experiment_id)
        .order_by(PromptVersion.iteration.asc())
    )
    versions = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": v.id,
            "iteration": v.iteration,
            "content": v.content,
            "rationale": v.rationale,
            "source": v.source.value,
            "parent_id": v.parent_id,
            "diff": v.diff,
            "created_at": v.created_at.isoformat(),
        }
        for v in versions
    ]


@router.get("/experiments/{experiment_id}/runs", response_model=list[dict[str, Any]])
async def list_runs(
    experiment_id: str,
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    stmt = (
        select(Run)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.iteration.asc(), Run.split.asc(), Run.target_model.asc())
    )
    runs = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id,
            "iteration": r.iteration,
            "split": r.split.value,
            "target_model": r.target_model,
            "prompt_version_id": r.prompt_version_id,
            "status": r.status.value,
            "mean_score": r.mean_score,
            "cost_usd": r.cost_usd,
            "created_at": r.created_at.isoformat(),
        }
        for r in runs
    ]


@router.get("/experiments/{experiment_id}/stream")
async def stream_events(experiment_id: str) -> StreamingResponse:
    async def _gen() -> AsyncIterator[bytes]:
        async for event in bus.stream(experiment_id):
            payload = json.dumps(
                {
                    "type": event.type,
                    "timestamp": event.timestamp.isoformat(),
                    **event.data,
                }
            )
            yield f"event: {event.type}\ndata: {payload}\n\n".encode()

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

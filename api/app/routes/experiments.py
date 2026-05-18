import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.db import get_session
from app.core.sse import bus
from app.models.evalset import EvalSet
from app.models.experiment import Experiment, ExperimentStatus
from app.models.project import Project
from app.models.prompt import PromptVersion
from app.models.run import Run
from app.schemas.experiment import ExperimentCreate, ExperimentOut, ExperimentSummary
from app.services import experiment_loop

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

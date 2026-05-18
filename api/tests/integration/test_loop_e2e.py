"""End-to-end loop test against the real Gemini API.

Skipped by default unless GEMINI_API_KEY is set in the environment. Use:
  GEMINI_API_KEY=... uv run pytest tests/integration -v -s
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import select

from app.core.db import SessionLocal
from app.models.evalset import EvalSet
from app.models.experiment import Experiment, ExperimentStatus, OptimizationObjective
from app.models.project import Project
from app.models.prompt import PromptVersion
from app.models.run import Run, RunResult
from app.services import experiment_loop

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
LIVE_TESTS = pytest.mark.skipif(
    not GEMINI_KEY,
    reason="GEMINI_API_KEY not set — skipping live integration test",
)


@LIVE_TESTS
@pytest.mark.asyncio
async def test_full_loop_on_gemini_flash_preview() -> None:
    """Drive a tiny experiment end-to-end against gemini-3-flash-preview.

    Tiny eval set + 1 iteration + low budget — the goal is to verify the wiring,
    not to measure quality.
    """
    # Using the stable 2.5-flash model — gemini-3-*-preview is currently unreliable
    # with strict JSON-schema response_format (returns empty `items` lists).
    target_model = "gemini/gemini-2.5-flash"
    agent_model = "gemini/gemini-2.5-flash"

    async with SessionLocal() as db:
        project = Project(name="e2e-smoke", description="live integration test")
        db.add(project)
        await db.flush()

        experiment = Experiment(
            project_id=project.id,
            name="e2e-classify",
            intent=(
                "Classify a customer support email into one of: "
                "billing, technical, account, other. Return only the single word label."
            ),
            requirements="Reply with one word, lowercase.",
            optimization_objectives=[
                OptimizationObjective.ACCURACY.value,
                OptimizationObjective.FORMAT_ADHERENCE.value,
            ],
            target_models=[target_model],
            agent_config={
                "writer_model": agent_model,
                "evalgen_model": agent_model,
                "judge_model": agent_model,
                "optimizer_model": agent_model,
            },
            budget_usd=2.0,
            max_iterations=1,
            eval_size=5,
            train_ratio=0.6,
            status=ExperimentStatus.PENDING,
        )
        db.add(experiment)
        await db.commit()
        exp_id = experiment.id

    # Drive the loop
    await experiment_loop.run_experiment(exp_id, mode="cold", existing_prompt=None)

    # Inspect results
    async with SessionLocal() as db:
        exp = await db.get(Experiment, exp_id)
        assert exp is not None
        assert exp.status in {
            ExperimentStatus.CONVERGED,
            ExperimentStatus.EXHAUSTED,
            ExperimentStatus.OVERFIT,
            ExperimentStatus.FAILED,
        }, f"unexpected status: {exp.status}"

        versions = (
            await db.execute(select(PromptVersion).where(PromptVersion.experiment_id == exp_id))
        ).scalars().all()
        assert len(versions) >= 1, "Writer should have produced at least v0"

        eval_sets = (
            await db.execute(select(EvalSet).where(EvalSet.experiment_id == exp_id))
        ).scalars().all()
        assert len(eval_sets) == 1, "EvalGen should have produced one EvalSet"

        runs = (
            await db.execute(select(Run).where(Run.experiment_id == exp_id))
        ).scalars().all()
        # At minimum we expect one train run on iteration 1 if the loop progressed past evalgen
        if exp.status != ExperimentStatus.FAILED:
            assert len(runs) >= 1, "Runner should have produced at least one Run"

        results = (
            await db.execute(
                select(RunResult)
                .join(Run, Run.id == RunResult.run_id)
                .where(Run.experiment_id == exp_id)
            )
        ).scalars().all()

        # Print a summary the test will surface with -s
        print("\n─── e2e smoke summary ─────────────────────────────────────────")
        print(f"  status:           {exp.status.value}")
        print(f"  prompt versions:  {len(versions)}")
        print(f"  eval sets:        {len(eval_sets)}")
        print(f"  runs:             {len(runs)}")
        print(f"  run results:      {len(results)}")
        print(f"  cost_usd:         ${exp.cost_usd:.4f} of ${exp.budget_usd:.2f}")
        if exp.failure_reason:
            print(f"  failure_reason:   {exp.failure_reason}")
        for v in versions:
            print(
                f"  prompt v{v.iteration} ({v.source.value}, "
                f"{len(v.content)} chars):  {v.content[:120].replace(chr(10), ' ')!r}"
            )
        for r in runs:
            print(
                f"  run iter={r.iteration} split={r.split.value} "
                f"model={r.target_model.split('/')[-1]} mean={r.mean_score} cost=${r.cost_usd:.4f}"
            )
        print("───────────────────────────────────────────────────────────────")

        # Final assertion: the loop must terminate WITHOUT a permanent failure
        assert exp.status != ExperimentStatus.FAILED, (
            f"loop failed: {exp.failure_reason}"
        )

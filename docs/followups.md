# Followups

Things we know we want to build but haven't yet. Captured here so we can revisit
without re-deriving the design each time.

---

## 1. Experiment lifecycle controls (start / pause / resume / restart)

**Status**: designed, not implemented. User has confirmed the design choices
below; this is ready to be picked up.

### Current behavior

- **Start**: implicit. `POST /projects/{id}/experiments` creates the experiment
  with `status=PENDING` and immediately fires `asyncio.create_task(run_experiment(...))`.
- **Cancel**: `POST /experiments/{id}/cancel` — cooperative, checked at each
  iteration boundary via `_check_cancelled()`.
- **Delete**: `DELETE /experiments/{id}`.
- **Pause / Resume / Restart**: not implemented. `ExperimentStatus.PAUSED` exists
  in the enum but is never referenced anywhere in the loop.

### Confirmed design choices

| Action  | Decision |
|---------|----------|
| Start   | Keep auto-start on create. No explicit button. |
| Pause   | Cooperative pause: set status → `PAUSED`; loop checks at iteration boundary and exits cleanly. |
| Resume  | Same-process resume: spawn a new task that reads `current_iteration` and continues from N+1. Reuses existing eval set + prompt versions. **Does not survive API restart** — but neither does the current loop. |
| Restart | Clone — create a new experiment with the same config. New `experiment_id`, runs fresh. Preserves the original experiment for reference. |

### Implementation plan

**Backend** (`api/app/`):

1. **Add `/pause` endpoint** in `routes/experiments.py`
   - Only allowed when status ∈ {`PENDING`, `RUNNING`}
   - Sets `status = PAUSED` (no other state changes)
   - Returns `ExperimentOut`
2. **Add `_check_paused()` in `services/experiment_loop.py`**
   - Mirror of `_check_cancelled()` (line 72)
   - At iteration boundary, if PAUSED → emit `loop.paused` event and return (don't transition to terminal state)
3. **Add `/resume` endpoint**
   - Only allowed when `status == PAUSED`
   - Sets `status = RUNNING`
   - Calls a new entry point `resume_experiment(experiment_id)` via `asyncio.create_task`
4. **Add `resume_experiment()` in `experiment_loop.py`**
   - Skips Writer + EvalGen phases (already done)
   - Loads v0, train_items, holdout_items from DB
   - Re-enters the iteration loop at `current_iteration + 1`
   - **Care needed**: most state is already on the experiment row + persisted prompt versions, but `train_means` / `holdout_means` accumulators need rehydration from `iteration_stats` for proper convergence checks
5. **Add `/clone` endpoint** (for restart)
   - Reads source experiment config
   - Creates new experiment (new id, same project, same agent_config / target_models / objectives / budget / eval_size / etc.)
   - Sets `name` to `"<original_name> (re-run)"`
   - Fires `asyncio.create_task(run_experiment(...))` like the create endpoint does
   - Returns the new experiment

**Frontend** (`web/`):

1. **Hooks in `lib/api/hooks.ts`** — add `usePauseExperiment`, `useResumeExperiment`, `useCloneExperiment` mirroring the existing `useCancelExperiment` pattern.
2. **Buttons in `app/projects/[id]/experiments/[expId]/page.tsx`** — add to the header action bar (next to Stop / Delete):
   - **Pause** — visible when status ∈ {`PENDING`, `RUNNING`}
   - **Resume** — visible when status == `PAUSED`
   - **Restart** — visible in any terminal state (`CONVERGED`, `OVERFIT`, `EXHAUSTED`, `FAILED`, `ACCEPTED`, `CANCELLED`). On click → call clone endpoint, then navigate to the new experiment.
3. **Update `RUNNING_STATUSES`** (page.tsx line 41) to include `"paused"` so polling continues during paused state, and so the SSE rail knows to listen for `loop.resumed`.
4. **Status badge** — already supports PAUSED via `types.ts` line 119, just verify the color.

### Risks / gotchas

- **Resume after API restart loses everything**. `asyncio.create_task` is in-process. Documenting this clearly is fine for now; proper fix is a job queue (out of scope).
- **In-flight LLM call latency on pause**. The cooperative pause-at-boundary model means any running judge/optimizer call must complete before pause takes effect. Same caveat as cancel today.
- **SSE stream lifecycle**. The frontend SSE rail closes on `loop.finished` / `loop.failed` events. Pause needs a `loop.paused` event that does NOT close the stream (so resume can re-emit). Or close + reopen on resume.
- **Race on resume**. If the user clicks Resume twice quickly, we could spawn two background tasks. Guard with a status check in the endpoint (only resume from PAUSED state, atomically flip to RUNNING).

---

## 2. Holdout never runs when v0 already passes (early-convergence blind spot)

**Status**: observed, no fix shipped.

### Symptom

User created experiment `333f48ce-2e11-4bd5-b8e6-2d296fd06cb3`. The loop reported
CONVERGED after iteration 1 with train mean **0.9929** (n=35) and **holdout n=0**
(holdout never ran). User asked "why converged in 1 iteration."

### Root cause

In `services/experiment_loop.py:689–800` the iteration loop does:

```text
for iteration in 1..max:
    3a. train pass on current_version
    3b. optimizer
    3c. if optimizer.new_prompt == current_prompt:
            final_status = CONVERGED
            break                          ← exits before 3d
    3d. persist new_version
    3e. holdout pass on new_version
```

Holdout is only ever evaluated on prompts the optimizer **changed**. If v0
already scores so well that the optimizer can't propose surgical edits (or its
edits get reverted by the variable-preservation guard), the loop converges
**without any holdout signal**. The "converged" label means "optimizer gave up,"
not "we verified generalization."

### Why the optimizer no-ops on this kind of input

In the specific experiment we investigated:
- 32/35 train items scored 1.0
- 3/35 scored 0.92 — all single-criterion failures on emergency-classification
- Failure samples passed to the optimizer were narrow and isolated
- Optimizer's "surgical" prompt biases it toward returning empty edits when
  failures look like model calibration drift rather than prompt issues
- Result: `OptimizerResult.edits == []` → `new_prompt == current_prompt` → CONVERGED

This is correct behavior in the narrow sense (optimizer was right not to
hand-tune around 8% tail noise) but wrong product behavior (user gets no
holdout score to trust the result).

### Proposed fixes (pick later)

Three options, ordered by scope:

1. **Always run holdout on v0 once**, even when convergence triggers in
   iteration 1. Move the holdout pass before the noop-break, OR run a
   one-shot holdout pass after the break for the LAST scored version. Adds
   one full holdout pass to the budget envelope.

2. **Add a manual "Run holdout" button** for any prompt version. Lets the
   user opt-in to validation when they want it, without spending budget
   automatically. Simpler product, more user-driven.

3. **Distinguish CONVERGED from OPTIMIZER_NOOP_AT_HIGH_SCORE**. New terminal
   status that flags "model already passes, but you have no holdout signal —
   run holdout to validate." Surface a CTA in the UI.

(2) is probably the smallest first step. (1) is the cleanest long-term fix.

### Related: the experiment we investigated

For future regression-testing, the specifics of the test case:

- **Mode**: warm
- **Source**: user's VinFast CRM extraction prompt (Python `.format()` syntax,
  `{group}` / `{group_description}` / `{subject_template}` / `{json_fields}`)
- **Failure mode that wasn't caught**: model over-classifies `emergency=true`
  (dead battery in residential basement, flat tire while driving, locked car
  in mall) when the rubric defines "emergency" as life-threatening only
- **Known_issues the user reported**: "bad labels in output (key/value)
  hallucination and sometimes leaves the placeholder values as is" — none of
  the actual failures matched these; the real bug is classification
  calibration that the user didn't anticipate
- **Cost spent**: $2.22 of $10 budget. Most went to EvalGen (50 items) +
  judging 35 items

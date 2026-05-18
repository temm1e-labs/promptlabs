# PromptLabs Architecture

## System

```
┌─────────────────────────────────────────────────────────────────┐
│  Next.js 15 (web)                                               │
│    pages: projects · experiments · lab-notebook · compare       │
│    charts: trajectory · train-vs-holdout · radar · pareto       │
└───────────────┬─────────────────────────────────────────────────┘
                │  REST + SSE (TanStack Query)
┌───────────────▼─────────────────────────────────────────────────┐
│  FastAPI (api)                                                  │
│    routes → services → agents → providers → db                  │
└────┬───────────────────────────────┬────────────────────────────┘
     │                               │
┌────▼─────────────┐         ┌───────▼──────────┐
│  Five agents     │         │  Provider layer  │
│  Writer          │ ◄─────► │  LiteLLM wrapper │ ─► 140+ providers
│  EvalGen         │         │  caching         │
│  Runner          │         │  cost tracking   │
│  Judge           │         │  retries         │
│  Optimizer       │         └──────────────────┘
└────┬─────────────┘
     │
┌────▼─────────────────────────────────────────────────────────────┐
│  SQLite (SQLAlchemy + Alembic)                                   │
│  Project · PromptVersion · EvalSet · Run · RunResult · Experiment│
└──────────────────────────────────────────────────────────────────┘
```

## Closed loop

```
START
  v0 = Writer(intent | existing_prompt)
  evalset = EvalGen(intent, v0)  → train (70%) + holdout (30%)

ITERATE n = 1..max_iterations OR until convergence OR until budget exhausted
  for each target_model:
    train_results = Runner(v_{n-1}, target_model, train)
    Judge scores train_results
  v_n_diff = Optimizer(v_{n-1}, train_results, failures)
  v_n = apply_diff(v_{n-1}, v_n_diff)
  for each target_model:
    holdout_results = Runner(v_n, target_model, holdout)
    Judge scores holdout_results
  emit SSE: iteration_complete

  CONVERGENCE
    train_score plateau (Δ < 0.01 over 2 iters), OR
    holdout_score declining (overfit detected)

END
  user picks the iteration to "accept" → marked final
```

## Five agents

1. **Writer** — cold (intent → prompt v0) or warm (existing prompt → v0 passthrough)
2. **EvalGen** — task intent + prompt → eval items + rubric, split deterministically into train/holdout
3. **Runner** — executes (PromptVersion × target_model × eval_split) with bounded concurrency and a content-hash cache
4. **Judge** — LLM-as-judge with structured rubric output; built-in deterministic scorers (exact_match, regex_match, json_valid, length_within)
5. **Optimizer** — failure analysis → surgical diff against the previous prompt, tied to which criterion failed; never a whole-prompt regeneration

## Data model

```
Project
  └─ Experiment (intent, target_models[], agent_config, budget_usd, status)
       ├─ PromptVersion (iteration, content, parent_id, source)
       ├─ EvalSet (split: train|holdout, rubric_criteria[])
       │    └─ EvalItem (input, expected_output?, metadata)
       └─ Run (prompt_version_id, target_model, eval_split, cost_usd)
            └─ RunResult (eval_item_id, actual_output, scores, latency, cost)
```

## Design principles

- **Provider layer is the single chokepoint.** Every LLM call goes through `api/app/core/providers.py`. Caching, cost tracking, retries, and structured-output enforcement live there once.
- **Diff over rewrite.** The Optimizer outputs a structured patch with line ranges and per-edit reasons. Users see *what changed and why*. Capped to K lines per iteration.
- **Train/holdout is sacrosanct.** The Optimizer only ever sees train scores. The lab notebook foregrounds the gap.
- **Multi-provider is a primitive.** Agent models and target models are separate axes. Defaulting all agents to the same model keeps the UX simple; per-role overrides exist for power users.
- **Aesthetic science.** Charts are first-class: ScoreTrajectory, TrainHoldoutGap, ParetoFront, CriterionRadar, ModelLeaderboard, ScoreDistribution. Real axis labels, real error bars, dark-first.

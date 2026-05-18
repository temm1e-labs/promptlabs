# PromptLabs

**A closed-loop lab for prompt engineering.** Write a prompt, auto-generate the
eval set, run it across multiple target models, score with an LLM judge, get a
*surgical* diff-style rewrite, repeat. Train/holdout discipline so overfitting is
visible the moment it starts. Multi-provider as a primitive — every agent role
can run on a different model from a different vendor.

Built for the moment between *"I have an idea for a prompt"* and *"this prompt
is production-ready."* That gap is currently filled by hand-tuning, vibes, and
ad-hoc spreadsheets. PromptLabs replaces it with a measurable loop.

```
┌─────────────────────────────────────────────────────────────────┐
│  Next.js UI                                                     │
│    Wizard → Lab notebook (charts, diffs, live SSE)              │
└───────────────┬─────────────────────────────────────────────────┘
                │  REST + SSE
┌───────────────▼─────────────────────────────────────────────────┐
│  FastAPI                                                        │
│    routes → orchestrator → 5 agents → LiteLLM → providers       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                       SQLite / Postgres
```

## Quickstart

```bash
git clone git@github.com:temm1e-labs/promptlabs.git
cd promptlabs
cp .env.example .env          # fill in API keys for the providers you'll use
make install                  # uv (python) + pnpm (node) deps
make migrate                  # apply database migrations
make dev                      # api on :8000, web on :3000
```

Open <http://localhost:3000>, create a project, then click **New experiment**.

For deployment as a backend microservice (fetch optimized prompts over HTTP
from your own apps), see **[docs/deployment.md](docs/deployment.md)**.

---

## The problem

You can't iterate on a prompt without three things you almost never have:

1. **A baseline.** Most people start with "let me try a prompt and see what
   happens." There's no v0 you can point to.
2. **An eval set that probes failure modes.** Writing 30 diverse test cases by
   hand is a half-day of work most engineers won't do.
3. **A way to know whether your rewrite actually helped.** Without a held-out
   set, every "I think this is better" is vibes.

Existing tools assume you have these. Langfuse, LangSmith, Braintrust, Helicone
are **observability** systems — they help you understand what's happening in
production prompts you've already deployed. DSPy, PromptWizard, TextGrad are
**optimization libraries** — they expect you to bring a metric, a training set,
and a programmer. Promptfoo is a great **evaluation harness** — but you bring
the prompt and the cases.

PromptLabs assumes you have nothing but an intent or a draft prompt, and
constructs the rest.

## How the loop works

**Phase 1 — Writer.** Given an intent (cold start) or an existing prompt (warm
start), produce v0. In warm mode, the user's prompt is preserved *verbatim*;
the Writer only extracts variable declarations and notes observations.

**Phase 2 — EvalGen.** Generate the rubric (3–6 weighted criteria, each
optionally tagged with an optimization objective) and N test cases. Cases are
split deterministically 70/30 into *train* and *holdout*. **The Optimizer
never sees holdout.**

**Phase 3 — iterate** up to `max_iterations`, or until convergence, or until
budget is exhausted:

1. For each *target* model: run the current prompt against train items,
   judge each output against the rubric.
2. Aggregate train scores; if budget exhausted → stop.
3. Pass the lowest-scoring train items as failure samples to the Optimizer.
   It returns a **structured diff** — anchored replace/insert/delete edits, not
   a rewritten prompt. Apply the diff → produce v_n.
4. Run v_n against *holdout* items, judge, persist.
5. Convergence check (see math below).

**Phase 4 — accept.** The user picks the iteration to mark as the production
version. Downstream apps fetch it via `GET /experiments/{id}/best-prompt`.

---

## The five agents

All agents share one **provider chokepoint** (`app/core/providers.py`) that
wraps LiteLLM: structured-output via Pydantic schemas, content-addressed
response cache, exponential-backoff retry, bounded concurrency, cost tracking.

| Agent | Job | Output schema | Key constraint |
|---|---|---|---|
| **Writer** | `(intent \| existing_prompt) → v0` | `{prompt, variables, rationale, assumptions}` | Warm mode preserves user prompt verbatim; if no `{{var}}` present, Runner uses chat-structured execution (system+user) |
| **EvalGen** | `(intent, v0, objectives) → rubric + eval items` | `{rubric: [{name, definition, weight, objective}], items: [{label, input_text/input_vars, expected_output?, tags}]}` | Items reconciled to declared variables (case-insensitive, positional, or flat `input_text` fallback). Deterministic train/holdout split |
| **Runner** | `(prompt, items, target_model) → per-item outputs` | n/a — emits `RunResult` per item | Bounded concurrency; auto-detects templated vs. chat-structured mode based on placeholder presence |
| **Judge** | `(rubric, item, actual_output) → scores` | `{scores: [{name, score: 0..1, reasoning}], overall_reasoning}` | Drops criteria not in the rubric; clamps scores to [0,1]; weighted aggregation |
| **Optimizer** | `(prompt_v_{n-1}, train failures, objectives) → diff` | `{edits: [{op, anchor?, new_text?, reason, targets_criterion?}], summary}` | Hard cap on edits/iter; rejects diffs that introduce new `{{var}}`; rejects ambiguous anchors |

Each role is independently configurable per experiment (`agent_config` field).
You can run Writer on a cheap model and Judge on an expensive one, or anything
else. Target models — the models being *tested* — are a separate axis.

---

## The science (why each guardrail exists)

### Train / holdout split

Goodhart: *when a measure becomes a target, it ceases to be a good measure*.
If the Optimizer can see every eval item, it'll converge to memorize them —
producing prompts that look great on the eval but fail in production.

Mitigation: a hard split. The Optimizer is fed the rubric, the current prompt,
and **failure samples drawn from the train pool only**. The holdout pool is
only used to *measure*, never to *steer*.

```
Items: 50
Train: 35  ← Optimizer sees these
Holdout: 15  ← Optimizer never sees these
```

### Train-vs-holdout gap as the overfit detector

Define for each iteration:

```
train_mean[n]   = mean(judge_score(item) for item in train, using v_n)
holdout_mean[n] = mean(judge_score(item) for item in holdout, using v_n)
gap[n]          = train_mean[n] − holdout_mean[n]
```

Overfitting is the regime where `train_mean` rises while `holdout_mean` falls
or stagnates. The detector fires when **both**:

```
holdout_mean[n-3] − holdout_mean[n]  ≥  0.04       (holdout dropped ≥4pp over 3 iters)
train_mean[n]     − holdout_mean[n]  ≥  0.05       (current gap ≥5pp)
```

The dual condition matters: with small eval sets (N=10–30 per split), a single
item flipping its score moves the average by ~3–10pp purely from grading noise.
Requiring a *sustained* drop **and** a widened gap filters that out.

### Convergence

`train_mean` plateau detection: if the last 3 iterations' train means differ by
less than `δ = 0.01`, the experiment is marked **converged**. No point burning
more LLM calls if signal is flat.

### Weighted scoring

For each rubric criterion `c` with weight `w_c`, the judge returns a score
`s_c ∈ [0, 1]`. The item-level mean is:

```
mean_score(item) = Σ_c (w_c · s_c) / Σ_c (w_c)
```

restricted to criteria the judge actually scored (it's allowed to abstain).

Per-objective aggregation uses the same weighted-mean formula but restricted
to criteria tagged with that objective:

```
score(objective) = Σ_{c: c.objective = obj} (w_c · s_c) / Σ_{c: c.objective = obj} (w_c)
```

This is what lets you ask, after the run, "*how well does v4 do on
robustness versus brevity?*" instead of one global number.

### Diff over rewrite

The Optimizer outputs a list of structured edits, not a new prompt:

```python
{
  "op": "replace" | "insert_before" | "insert_after" | "delete" | "append",
  "anchor": "<existing substring, must be unique>",
  "new_text": "<replacement / insertion>",
  "reason": "<why this edit — tied to a failing criterion>",
  "targets_criterion": "<rubric criterion name>"
}
```

Three reasons this matters:

1. **Interpretability.** The user sees *what changed and why* on every
   iteration. A whole-prompt rewrite hides the reasoning.
2. **Stability.** The prompt evolves through small steps. Catastrophic
   regressions are rare; rollback to v_{n-1} is trivial.
3. **Constraint enforcement.** Code-level guards:
   - Anchor must be unique (one occurrence) — prevents accidental wide
     replacement.
   - Max 5 edits per iteration.
   - Cumulative chars changed capped at `max(100, 0.35 × prompt_len)`.
   - Diff is **rejected** if it introduces any new `{{variable}}` — the eval
     items would have no values for it, breaking every downstream call.
   - Reverts to v_{n-1} on validation failure rather than persisting a broken
     prompt.

### Budget as a hard constraint

Every LLM call adds its cost (from `litellm.completion_cost`) to
`experiment.cost_usd`. Before each phase boundary the orchestrator checks
`cost_usd < budget_usd`; if not, status flips to `EXHAUSTED` and the loop
stops cleanly. No silent overruns.

---

## The two execution modes

The Runner inspects the prompt for `{{variable}}` placeholders (regex:
`\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}`) and chooses:

**Templated mode** (placeholders present) — the canonical case. Substitute the
eval item's `input_vars` and send as a single user message. Use for:
classification, extraction, summarization, anything where the prompt is a
function of one or more inputs.

**Chat-structured mode** (no placeholders) — the prompt becomes a **system**
message, and the eval item's `user_input` (or first available variable)
becomes the **user** message. Use for: agent system prompts, callbot
instructions, anything where the whole prompt *is* the role definition and
variation comes from the user's turn.

The Writer/EvalGen pipeline auto-detects this and synthesizes a virtual
`user_input` variable for the chat case, without touching the original prompt
text.

---

## What it explicitly is NOT

- Not a production observability tool. Use Langfuse for that.
- Not a multi-tenant SaaS. Single-user, local-first by design.
- Not a replacement for human eval. The LLM judge can be wrong; the rubric
  matters. Always read the failure samples in the lab notebook.
- Not magic. If the LLM driving the agents is unreliable (e.g. some preview
  models flake at structured output), the loop degrades.

---

## Stack

**Backend** — Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.x async,
aiosqlite, Alembic, LiteLLM, tenacity, sse-starlette, structlog. Managed by
`uv`. 83 unit tests + 1 live end-to-end test against the Gemini API.

**Frontend** — Next.js 15 (App Router), React 19, TypeScript strict,
Tailwind v4, shadcn/ui primitives, Recharts, TanStack Query, Sonner. Managed
by `pnpm`. Dark-first theme.

**Deployment** — Docker images for both `api/` and `web/`. The API is a
self-contained microservice that downstream apps query via REST. See
`docs/deployment.md`.

---

## Fetching the optimized prompt from your app

```python
import httpx
r = httpx.get(
    f"{PROMPTLABS_URL}/experiments/{experiment_id}/best-prompt",
    headers={"Authorization": f"Bearer {PROMPTLABS_API_KEY}"},
)
prompt: str = r.json()["prompt"]
```

The endpoint returns the user-accepted iteration if any; otherwise the
iteration with the highest mean holdout score; otherwise the most recent
iteration. See `docs/deployment.md` for production patterns (caching,
fallback prompts, alias proposals).

---

## License

MIT.

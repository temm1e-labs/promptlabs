# PromptLabs

**A closed-loop lab for production prompt engineering.** You paste an existing prompt or describe an intent. PromptLabs generates a stratified eval set, runs the prompt across every target model you care about in parallel, scores each output with an LLM judge, then proposes a surgical diff. It loops with real train/holdout discipline until the prompt converges, overfits, or runs out of budget.

> Built for the moment between *"I have an idea for a prompt"* and *"this prompt is production-ready."* That gap is filled today by hand-tuning, vibes, and ad-hoc spreadsheets. PromptLabs replaces it with a measurable loop.

---

## The whole loop in one picture

```
       INPUT
       ─────
       intent  ──►  ┌────────────┐
            OR      │   Writer   │  drafts v0
       prompt ───►  └──────┬─────┘
                           ▼
                    ┌────────────┐    taxonomy (5–8 categories)
                    │  EvalGen   │  → parallel batch per category
                    └──────┬─────┘  → dedup → train (70%) | holdout (30%)
                           │
            ╔══════════════╪═══════════════════════════════════╗
            ║              ▼                                   ║
            ║       ┌────────────┐    fan-out in parallel      ║
            ║       │   Runner   │    across every target      ║
            ║       └──────┬─────┘    model you're comparing   ║
            ║              ▼                                   ║
            ║       ┌────────────┐    per-criterion scores,    ║
            ║       │   Judge    │    weighted by rubric,      ║
            ║       └──────┬─────┘    reasoning attached       ║
            ║              ▼                                   ║
            ║       ┌────────────┐    returns a DIFF, not a    ║
            ║       │ Optimizer  │    regenerated prompt       ║
            ║       └──────┬─────┘    → apply → v(n+1)         ║
            ║              ▼                                   ║
            ║      score v(n+1) on HOLDOUT                     ║
            ║              ▼                                   ║
            ║      converged?  overfit?  budget exhausted?     ║
            ║              │                                   ║
            ║              ├──── all NO  ──►  back to Runner   ║
            ║              │                                   ║
            ║              └──── any YES ──┐                   ║
            ╚══════════════════════════════│═══════════════════╝
                                           ▼
                                  accepted version
                              served over HTTP to prod
```

Five agents. One closed loop. Every box exists for a reason and the science section below explains each one.

---

## Quickstart

```bash
git clone git@github.com:temm1e-labs/promptlabs.git
cd promptlabs
cp .env.example .env          # fill in API keys for the providers you'll use
make install                  # uv (python) + pnpm (node) deps
make migrate                  # apply database migrations
make dev                      # api on :8000, web on :3000
```

Open <http://localhost:3000>, create a project, click **New experiment**.

For deployment as a microservice (fetch optimized prompts over HTTP from your own apps), see **[docs/deployment.md](docs/deployment.md)**.

---

## What problem this solves

You can't iterate on a prompt without three things you almost never have:

```
   ✗  no baseline           →  "let me try something and see what happens"
   ✗  no eval set           →  half a day of writing test cases by hand
   ✗  no holdout            →  every "I think this is better" is vibes
```

Existing tools assume you already have these:

| Tool category | Examples | What they do | What they assume |
|---|---|---|---|
| **Observability** | Langfuse, LangSmith, Braintrust, Helicone | Trace prompts in production | You already shipped a prompt |
| **Optimization library** | DSPy, TextGrad, PromptWizard | Optimize a prompt to a metric | You bring a metric, a training set, and a programmer |
| **Eval harness** | Promptfoo | Run eval cases against prompts | You bring the prompt and the cases |

PromptLabs assumes you have nothing but an intent or a draft prompt. It constructs the rest.

---

## Five agents, one chokepoint

Every LLM call in the system flows through **one provider layer** (`app/core/providers.py`) wrapping LiteLLM — structured output via Pydantic, content-addressed cache, exponential-backoff retry, bounded concurrency, cost tracking. So every cost number, every retry, every cached response is visible in one place.

```
                       ┌──────────────────┐
   Writer   ─────────► │                  │
   EvalGen  ─────────► │   providers.py   │ ───► LiteLLM ───► 140+ providers
   Runner   ─────────► │   (chokepoint)   │
   Judge    ─────────► │                  │ ───► cache (SHA256 keyed)
   Optimizer ────────► │                  │ ───► cost tracking
                       └──────────────────┘
```

| Agent | Job | Key constraint |
|---|---|---|
| **Writer** | `(intent \| existing_prompt) → v0` | Warm mode preserves user prompt verbatim. No `{{var}}` placeholders → Runner uses chat-structured execution |
| **EvalGen** | `(intent, v0, objectives) → rubric + eval items` | Stratified taxonomy + parallel batches + dedup. Deterministic shuffled 70/30 split |
| **Runner** | `(prompt, items, target_model) → outputs` | Bounded concurrency. Auto-detects templated vs. chat-structured mode |
| **Judge** | `(rubric, item, actual_output) → scores` | Per-criterion scoring with reasoning, weighted aggregation, clamped to `[0,1]` |
| **Optimizer** | `(prompt_v_{n-1}, train failures) → diff` | Anchored edits with hard cap. Rejects diffs that introduce new variables |

Each role is independently configurable per experiment. Run Writer on Sonnet 4.6, Judge on Haiku, target models on whatever you actually ship with. Target models (what you're *testing*) and agent models (what *drives the loop*) are separate axes.

---

## How EvalGen generates a real eval set (not 50 happy-path copies)

A single LLM call asked for 50 diverse test cases produces 40 reworded variations of the easiest one. Output quality degrades past ~15 items per call. PromptLabs splits the work:

```
        ┌────────────────────────────────────────┐
        │  Step 1 — Taxonomy (one small call)    │
        │                                        │
        │   intent  +  prompt v0   →   rubric    │
        │                          +   5-8       │
        │                              categories│
        └────────────────────┬───────────────────┘
                             │
                             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
   │  cat 1  │  │  cat 2  │  │  cat 3  │  │  cat 4  │  │  cat 5  │
   │ ~8 items│  │ ~8 items│  │ ~8 items│  │ ~8 items│  │ ~8 items│
   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
        │            │            │            │            │
        └────────────┴───── parallel ──────────┴────────────┘
                             │
                             ▼
                  Step 2 — Dedup (text hash)
                             │
                             ▼
                  Step 3 — Top-up if short
                             │
                             ▼
                shuffled deterministic split
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
            TRAIN (70%)              HOLDOUT (30%)
            Optimizer sees these     Optimizer NEVER sees these
```

The taxonomy gives you structural diversity. Parallel calls keep each batch inside the "reliable structured output" envelope. Dedup catches near-duplicates across batches. The result is an eval set that covers what your prompt will actually see, not what was easiest to imagine.

For `eval_size <= 15` PromptLabs skips the taxonomy and uses a single call. The overhead isn't worth it for small sets.

---

## How the loop runs target models in parallel

Choosing between Sonnet 4.6, GPT-5, and Gemini 3? They all run on the same eval set simultaneously, not sequentially:

```
                   train items (one set, 35 items)
                              │
                              ▼
            ┌──────────────────────────────────────┐
            │                                      │
       ┌────┴────┐                            ┌────┴────┐
       │ runner  │  runner  │  runner  │   ...│ runner  │
       │ Sonnet  │  GPT-5   │  Gemini  │      │  ...    │
       └────┬────┘  └────┬───┘  └────┬───┘   └────┬─────┘
            │            │           │            │
            └──────── all in parallel ────────────┘
                              │
                              ▼
                       judges (parallel)
                              │
                              ▼
                   per-model leaderboard
                  + cost/eval on same chart
```

For N target models this is an Nx walltime reduction on the runner and judge phases. Results in four minutes, not three afternoons.

---

## The science (why each guardrail exists)

### Train / holdout — shuffled, deterministic, enforced

```
   raw items from EvalGen:   [happy, happy, happy, edge, edge, adv, adv, ...]
                                              │
                                              ▼  shuffle by SHA256(experiment_id)
                                              │
                             [edge, happy, adv, happy, edge, happy, adv, ...]
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
                    train (70%)                            holdout (30%)
                    seen by Optimizer                      never leaks
```

The shuffle matters. Without it, EvalGen's natural tendency to emit common cases first and adversarial last puts the easy cases in train and the hard cases in holdout, making train systematically easier and invalidating every train-vs-holdout comparison. The seed is `SHA256(experiment_id)[:8]`, so the split is reproducible per experiment.

### Overfit detector — divergence-based, sample-size aware

Overfitting is *divergence*, not *holdout going down*. The detector watches both sides:

```
          train  ─────────────────────────────────────► time
                 ↗ ↗ ↗ ↗ ↗ ↗
                                                     ┌─── OVERFIT
                                                     │     train ↑
                                                     │     holdout ↓
                 ↘ ↘ ↘ ↘ ↘                           │     by > τ(N)
          holdout ─────────────────────────────────► time
```

```
   τ(N) = 1.96 · σ / √N                    (95% CI half-width; σ ≈ 0.2)

   overfit  ⇔   Δtrain[n]   ≥  +τ(N_train)    AND
                Δholdout[n] ≤  −τ(N_holdout)
```

Each side's threshold scales with that side's sample size. The rule behaves consistently from Quick (N=20) through Thorough (N=100). Co-regression (both falling) and co-improvement (both rising) explicitly do **not** trigger — those are different phenomena.

### Convergence — sample-size aware plateau

Stop the loop when the last three iterations' means are inside the per-iteration noise floor:

```
   δ(N)        = 1.96 · σ / √N
   converged   ⇔   max(train[n-2:n+1]) − min(train[n-2:n+1])  <  δ(N_train)
```

No point burning more LLM calls when the signal sits below the noise.

### Production selection — lower confidence bound, not greedy max

```
   greedy max:    pick argmax(mean)               ← favors lucky high-variance prompts
   LCB:           pick argmax(mean − 1.96·SE)     ← favors robust prompts
```

`GET /experiments/{id}/best-prompt` uses LCB. On small holdout sets (N ≤ 15) the two rules pick different winners often enough to matter, and the robust pick is the right default for production.

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

1. **Interpretability.** Every iteration shows you *what changed and why*. A whole-prompt rewrite hides reasoning.
2. **Stability.** The prompt evolves through small steps. Catastrophic regressions are rare; rollback to v_{n-1} is trivial.
3. **Constraint enforcement.** Code-level guards: anchor must be unique, max 5 edits per iteration, cumulative chars changed capped at `max(100, 0.35 × prompt_len)`, diff rejected if it introduces new `{{variable}}`.

### Failure-sample selection — stratified, not greedy

The Optimizer receives `k=8` failure samples per iteration:

```
   Pass 1:   for each rubric criterion C:
                pick the lowest-scoring train item where score(C) < 0.5
                → one exemplar per failing criterion

   Pass 2:   fill remaining slots up to k with next-lowest overall
```

Previous behavior (top-k by mean) collapsed when many items failed the same criterion — the Optimizer saw 8 duplicates of "format wrong" and missed items failing "robustness". Stratified selection gives balanced edit signal.

### Weighted scoring

For each rubric criterion `c` with weight `w_c` and judge score `s_c ∈ [0,1]`:

```
   mean_score(item)   =   Σ_c (w_c · s_c)  /  Σ_c (w_c)
                        (restricted to criteria the judge actually scored)
```

Per-objective aggregation uses the same formula restricted to criteria tagged with that objective. That's what lets you ask "how well does v4 do on robustness vs. brevity?" instead of one global number.

### Budget as a hard constraint

Every LLM call adds its cost (from `litellm.completion_cost`) to `experiment.cost_usd`. Before each phase boundary the orchestrator checks `cost_usd < budget_usd`; if not, status flips to `EXHAUSTED` and the loop stops cleanly. No silent overruns.

---

## The two execution modes

The Runner inspects the prompt for `{{variable}}` placeholders and picks one of two modes:

```
                  has {{placeholders}}?
                          │
                ┌─────────┴─────────┐
              YES                  NO
                │                   │
                ▼                   ▼
    ┌─────────────────┐   ┌─────────────────────┐
    │ Templated mode  │   │ Chat-structured     │
    │                 │   │                     │
    │ substitute vars │   │ prompt → system msg │
    │ → single user   │   │ user_input → user   │
    │   message       │   │   message           │
    └─────────────────┘   └─────────────────────┘
       classification           agent system prompts
       extraction               callbot instructions
       summarization            "the prompt is the role"
```

The Writer/EvalGen pipeline auto-detects the placeholder-less case and synthesizes a virtual `user_input` variable for evals, without modifying the original prompt text. So warm-starting from a real production system prompt works without rewriting anything.

---

## What it explicitly is NOT

- Not a production observability tool. Use Langfuse for that.
- Not a multi-tenant SaaS. Single-user, local-first by design.
- Not a replacement for human eval. The LLM judge can be wrong; always read the failure samples.
- Not magic. If the agent LLM is unreliable at structured output (some preview models flake), the loop degrades.

## Honest limitations

The closed loop is well-instrumented, but the measurements have boundaries:

1. **Internal validity only.** A 95% holdout score means "this prompt does well on an LLM-generated eval set, graded by another LLM." It does NOT mean 95% on your production traffic. Always re-measure with real data after deploying.

2. **In-distribution overfit detection only.** Train and holdout both come from a single EvalGen call. Blind spots in EvalGen (no multilingual, no injection probes) affect both splits equally.

3. **Greedy multi-target optimization.** With multiple target models the Optimizer's failure samples are aggregated; the prompt drifts toward whichever model's failures dominate. Per-target Optimizer is on the roadmap.

4. **Power limits at small N.** Quick preset (N=20, holdout=6) can detect a ~12pp improvement reliably. Smaller improvements are below the noise floor. Use Standard (N=50) or Thorough (N=100) for fine-grained changes.

5. **Partial reproducibility.** The train/holdout split is deterministic per experiment. The eval items themselves are LLM-generated and not byte-reproducible — re-running gets a similar but not identical eval set.

6. **The judge has no ground truth.** LLM-graded scores are subjective fuzzy values, not probabilities. Always read failure samples.

7. **Rubric leakage.** EvalGen produces the rubric and items in the same call, so the rubric's criteria implicitly reflect knowledge of the holdout. The Optimizer sees the rubric. Real (if subtle) information leak that we haven't yet plugged.

When all this is acknowledged, the system is what it is: **an instrumented fast loop for prompt iteration with honest error bars, sample-size-aware guardrails, and a defensible production selection rule.** Useful. Not "automatic prompt optimization with statistical guarantees."

---

## Stack

```
       ┌─────────────────────────────────────────────────────────┐
       │   Frontend                                              │
       │   Next.js 15 · React 19 · TS strict · Tailwind v4       │
       │   shadcn/ui · Recharts · TanStack Query · Sonner        │
       └─────────────────────────────┬───────────────────────────┘
                                     │  REST + SSE
       ┌─────────────────────────────▼───────────────────────────┐
       │   Backend                                               │
       │   Python 3.12 · FastAPI · Pydantic v2 · async SQLA 2.x  │
       │   LiteLLM · aiosqlite · Alembic · sse-starlette         │
       │   tenacity · structlog · uv                             │
       └─────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                             SQLite / Postgres
```

97 unit tests + 1 live end-to-end test against the Gemini API. Docker images for both `api/` and `web/`. The API is a self-contained microservice that downstream apps query via REST.

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

The endpoint returns the user-accepted iteration if one exists; otherwise the highest-LCB iteration; otherwise the most recent. See `docs/deployment.md` for production patterns: caching, fallback prompts, alias proposals, CORS allowlists, rate limiting.

---

## License

MIT.

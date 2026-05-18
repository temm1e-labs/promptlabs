# PromptLabs

A lab — not an observability tool — for the full prompt-engineering loop.

**Idea or existing prompt → auto-generated eval dataset → execution across multiple target models → LLM-judge scoring → surgical, diff-style rewrite → repeat.**

Built on [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider access (140+ providers, 2,500+ models). FastAPI backend, Next.js frontend, SQLite storage. Local-first, single-user.

## What makes it different

- **Closed loop, not a dashboard.** Most tools (Langfuse, Promptfoo, Braintrust) require you to bring prompts and evals. PromptLabs writes both, then rewrites the prompt based on signal.
- **Surgical diffs, not regeneration.** The Optimizer outputs a patch against the previous prompt, tied to which eval criterion failed. You see *what changed and why* on every iteration.
- **Multi-provider as a primitive.** Any agent role (Writer, EvalGen, Judge, Optimizer) can run on any provider; target models for A/B are a separate axis.
- **Train/holdout discipline.** Every auto-generated eval splits 70/30. The Optimizer never sees holdout. Train-vs-holdout gap is a first-class chart — overfit is visible the moment it starts.
- **Aesthetic science.** Real metrics, real charts (score trajectory, Pareto front, criterion radar, violin distributions). Not Streamlit defaults.

## Quickstart

```bash
make install      # install api + web deps
make migrate      # apply database migrations
make dev          # run api (:8000) + web (:3000) concurrently
```

Then open http://localhost:3000.

Set provider API keys in `.env` (see `.env.example`).

## Architecture

See [docs/architecture.md](docs/architecture.md).

## License

MIT

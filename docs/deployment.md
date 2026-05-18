# Deploying PromptLabs as a microservice

PromptLabs is built as a FastAPI service + a Next.js UI. You can deploy them
independently. Most production users want **just the API**, and have their own apps
call it to fetch optimized prompts.

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│  Your application   │ ──────▶ │   PromptLabs API     │ ──▶ LLM providers
│  (any language)     │  HTTPS  │   (FastAPI :8000)    │
└─────────────────────┘         └────────┬─────────────┘
                                         │
                                ┌────────▼─────────┐
                                │  SQLite or       │
                                │  Postgres        │
                                └──────────────────┘
```

## Quick deploy with Docker Compose

```bash
cp .env.example .env
# Fill in: GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY, etc.
# Set PROMPTLABS_API_KEY to a strong secret for production.

docker compose up -d
```

That gives you:
- API on `:8000`
- Web UI on `:3000`
- SQLite volume mounted at `./data`

## Run only the API (the common production case)

```bash
docker build -t promptlabs-api ./api
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY=... \
  -e ANTHROPIC_API_KEY=... \
  -e PROMPTLABS_API_KEY=$(openssl rand -hex 32) \
  -e PROMPTLABS_DB_URL=postgresql+asyncpg://user:pass@host/db \
  -v promptlabs-data:/app/data \
  promptlabs-api
```

Point it at Postgres for HA. SQLite is fine for single-instance.

## Authentication

Set `PROMPTLABS_API_KEY` to require a bearer token on every request:

```bash
curl -H "Authorization: Bearer $PROMPTLABS_API_KEY" \
     https://promptlabs.example.com/projects
```

Public paths (always open): `/healthz`, `/docs`, `/redoc`, `/openapi.json`.

If `PROMPTLABS_API_KEY` is unset, auth is disabled (local dev mode).

## CORS

By default the API allows any `http://localhost:*` origin. For production:

```bash
PROMPTLABS_CORS_ORIGINS=https://app.yourcompany.com,https://admin.yourcompany.com
```

## Fetching the optimized prompt from your application

The endpoint:

```http
GET /experiments/{experiment_id}/best-prompt
Authorization: Bearer <PROMPTLABS_API_KEY>
```

Returns:

```json
{
  "experiment_id": "9db0de78-...",
  "experiment_name": "state machine intention",
  "iteration": 4,
  "prompt": "<the prompt text — use this in your app>",
  "source": "optimizer",
  "selection_reason": "best_holdout (iter 4, score 0.992)",
  "experiment_status": "accepted",
  "version_id": "abcd1234-...",
  "created_at": "2026-05-18T14:23:30Z"
}
```

**Selection priority** (built into the endpoint):
1. The iteration the user explicitly accepted (clicked "Accept" in the UI).
2. The iteration with the highest mean holdout score.
3. The most recent iteration (fallback while a run is still in progress).

### Example: Python client

```python
import os
import httpx

PROMPTLABS = os.environ["PROMPTLABS_URL"]
TOKEN = os.environ["PROMPTLABS_API_KEY"]

def get_prompt(experiment_id: str) -> str:
    r = httpx.get(
        f"{PROMPTLABS}/experiments/{experiment_id}/best-prompt",
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=5,
    )
    r.raise_for_status()
    return r.json()["prompt"]


# Cache for 1 minute, fall back to a baked-in default on failure
@functools.lru_cache(maxsize=128)
def cached_prompt(experiment_id: str) -> str:
    try:
        return get_prompt(experiment_id)
    except Exception:
        return FALLBACK_PROMPT
```

### Example: TypeScript / Node

```ts
async function getPrompt(experimentId: string): Promise<string> {
  const res = await fetch(
    `${process.env.PROMPTLABS_URL}/experiments/${experimentId}/best-prompt`,
    { headers: { Authorization: `Bearer ${process.env.PROMPTLABS_API_KEY}` } },
  );
  if (!res.ok) throw new Error(`promptlabs ${res.status}`);
  const data = await res.json();
  return data.prompt;
}
```

## Recommended production setup

1. **Database:** Postgres (managed; e.g. RDS, Supabase, Neon). Set
   `PROMPTLABS_DB_URL=postgresql+asyncpg://...`. SQLite is fine for prototypes
   but doesn't handle concurrent writes well across replicas.
2. **API behind a reverse proxy** (nginx, Caddy, Cloud Run, Fly.io). Terminate
   TLS there.
3. **One API key per consuming app**, not a shared one. Even though the
   middleware accepts a single token, you can put a gateway in front that maps
   distinct keys to the same upstream — easier rotation.
4. **Pin the prompt version on the client side too.** Fetching `best-prompt`
   every request adds latency and creates a dependency loop if PromptLabs is
   down. Either:
   - cache aggressively (LRU + short TTL), with the last-known-good prompt baked
     in as a fallback constant; OR
   - poll on a schedule (every 5 min) and serve from local cache.
5. **Use `/experiments/{id}/accept` deliberately.** That endpoint marks an
   iteration as the production version. Until then, `best-prompt` will keep
   moving as new iterations improve the score — sometimes desirable, sometimes
   not. Mark `accepted` when you're confident.

## What's NOT yet built (open about it)

- **Prompt aliases / human-readable handles.** Currently you fetch by UUID. A
  "promote `prod` alias to experiment X" feature is on the roadmap.
- **Webhooks** when an experiment finishes or a new best version emerges.
- **Multi-tenant auth.** All API keys give the same access — fine for one team,
  not for SaaS.
- **Read-replica / scaling.** The orchestrator uses an in-process task queue
  (`asyncio.create_task`); restart-time interruption is possible. For
  high-throughput production, replace with Celery/RQ or a queue-backed worker.

These are all reasonable next steps when you actually hit them.

"""SQLite-backed response cache for the provider layer.

Independent of the app DB so cache lifecycle is decoupled from migrations.
Keys are content-addressed: (model, messages, response_format) → SHA256.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite

from app.core.config import DATA_DIR, settings

CACHE_DB_PATH = DATA_DIR / "provider_cache.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response_text TEXT NOT NULL,
    cost_usd REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);
"""


@dataclass
class CacheHit:
    response_text: str
    cost_usd: float
    input_tokens: int
    output_tokens: int


def cache_key(model: str, messages: list[dict[str, Any]], schema_hash: str | None) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "schema": schema_hash},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


async def _ensure_schema() -> None:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(CACHE_DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


async def get(key: str) -> CacheHit | None:
    await _ensure_schema()
    async with aiosqlite.connect(CACHE_DB_PATH) as db:
        cursor = await db.execute(
            "SELECT response_text, cost_usd, input_tokens, output_tokens, expires_at "
            "FROM cache WHERE key = ?",
            (key,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        response_text, cost_usd, input_tokens, output_tokens, expires_at = row
        if datetime.fromisoformat(expires_at) < datetime.now(UTC):
            return None
        return CacheHit(
            response_text=response_text,
            cost_usd=float(cost_usd),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
        )


async def put(
    key: str,
    model: str,
    response_text: str,
    cost_usd: float,
    input_tokens: int,
    output_tokens: int,
    ttl_s: int | None = None,
) -> None:
    await _ensure_schema()
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=ttl_s or settings.cache_ttl_s)
    async with aiosqlite.connect(CACHE_DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO cache (key, model, response_text, cost_usd, "
            "input_tokens, output_tokens, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                model,
                response_text,
                cost_usd,
                input_tokens,
                output_tokens,
                now.isoformat(),
                expires_at.isoformat(),
            ),
        )
        await db.commit()


async def clear() -> int:
    await _ensure_schema()
    async with aiosqlite.connect(CACHE_DB_PATH) as db:
        cursor = await db.execute("DELETE FROM cache")
        count = cursor.rowcount
        await db.commit()
        return int(count)

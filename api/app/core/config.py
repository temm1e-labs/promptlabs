from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# api/app/core/config.py → api/ is parents[2]; repo root is parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[
            str(PROJECT_ROOT / ".env"),
            str(PROJECT_ROOT / "api" / ".env"),
        ],
        env_prefix="PROMPTLABS_",
        extra="ignore",
    )

    default_model: str = Field(default="claude-sonnet-4-6")
    db_url: str = Field(default=f"sqlite+aiosqlite:///{DATA_DIR / 'promptlabs.db'}")
    cache_dir: Path = Field(default=DATA_DIR / "cache")
    log_level: str = Field(default="INFO")

    # Loop defaults
    default_eval_size: int = 30
    default_train_ratio: float = 0.7
    default_max_iterations: int = 10
    default_budget_usd: float = 5.0

    # Provider layer
    request_timeout_s: int = 120
    max_concurrent_requests: int = 8
    cache_ttl_s: int = 60 * 60 * 24 * 30  # 30 days

    # Deployment / API auth — set this in production to require a bearer token
    # on every request. Leave unset for local dev.
    api_key: str | None = None
    cors_origins: list[str] | None = None  # comma-separated; allow_origin_regex by default


settings = Settings()


def ensure_data_dirs() -> None:
    """Create data dirs idempotently. Safe to call from any entrypoint."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    if settings.db_url.startswith("sqlite"):
        path_part = settings.db_url.split("///", 1)[-1]
        Path(path_part).parent.mkdir(parents=True, exist_ok=True)


ensure_data_dirs()

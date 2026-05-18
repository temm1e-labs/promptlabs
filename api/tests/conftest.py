from pathlib import Path

import pytest

from app.core import cache


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets its own provider cache file."""
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "cache.db")

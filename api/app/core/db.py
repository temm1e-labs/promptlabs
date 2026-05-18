from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import ensure_data_dirs, settings

ensure_data_dirs()

_is_sqlite = "sqlite" in settings.db_url

engine = create_async_engine(
    settings.db_url,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)


# SQLite write-concurrency fix.
#
# The orchestrator now opens a fresh AsyncSession per parallel target-model run
# (so SQLAlchemy session safety holds), but multiple sessions hammering the same
# SQLite file caused "database is locked" errors — the cancel endpoint could
# 500 mid-experiment because it couldn't acquire the write lock.
#
# WAL (Write-Ahead Logging) lets readers and one writer proceed without blocking
# each other, and busy_timeout makes SQLite wait for the lock instead of failing
# immediately. synchronous=NORMAL is the canonical pairing with WAL — fsync per
# checkpoint instead of per commit, durability sufficient for a local lab tool.
if _is_sqlite:

    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

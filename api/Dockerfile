FROM python:3.12-slim

WORKDIR /app

# Install uv (the fast python package manager)
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /uvx /bin/

# Cache deps separately from app code
COPY pyproject.toml uv.lock* /app/
RUN uv sync --frozen --no-dev || uv sync --no-dev

COPY app /app/app
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Apply migrations then start the server. For production, replace --reload with workers.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

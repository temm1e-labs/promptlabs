.PHONY: help install install-api install-web dev dev-api dev-web migrate test test-api test-web lint lint-api lint-web typecheck typecheck-api typecheck-web build ci clean

help:
	@echo "PromptLabs — make targets"
	@echo ""
	@echo "  install       install api + web deps"
	@echo "  dev           run api + web concurrently"
	@echo "  migrate       apply db migrations"
	@echo "  test          run all tests"
	@echo "  lint          lint api + web"
	@echo "  typecheck     type-check api + web"
	@echo "  build         build web for production"
	@echo "  ci            lint + typecheck + test (what GitHub Actions runs)"
	@echo "  clean         remove caches, build artifacts"

install: install-api install-web

install-api:
	cd api && uv sync

install-web:
	cd web && pnpm install

dev:
	@mkdir -p data
	@bash -c 'trap "kill 0" EXIT; \
	  (cd api && uv run uvicorn app.main:app --reload --port 8000) & \
	  (cd web && pnpm dev) & \
	  wait'

dev-api:
	@mkdir -p data
	cd api && uv run uvicorn app.main:app --reload --port 8000

dev-web:
	cd web && pnpm dev

migrate:
	@mkdir -p data
	cd api && uv run alembic upgrade head

migration:
	cd api && uv run alembic revision --autogenerate -m "$(m)"

test: test-api test-web

test-api:
	cd api && uv run pytest

test-web:
	cd web && pnpm test

lint: lint-api lint-web

lint-api:
	cd api && uv run ruff check . && uv run ruff format --check .

lint-web:
	cd web && pnpm lint

typecheck: typecheck-api typecheck-web

typecheck-api:
	cd api && uv run mypy app

typecheck-web:
	cd web && pnpm typecheck

build:
	cd web && pnpm build

ci: lint typecheck test build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf web/.next web/out api/.venv

SHELL := /bin/bash
COMPOSE := docker compose
UV := cd backend && VIRTUAL_ENV= uv run

.PHONY: help up down logs ps infra models ingest ingest-tamir eval test test-integration lint fmt reset-data build

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

up: ## Build and start the whole stack (api, web, infra)
	$(COMPOSE) up -d --build

down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

logs: ## Tail logs
	$(COMPOSE) logs -f --tail=200

ps: ## Show service status
	$(COMPOSE) ps

infra: ## Start only postgres, redis, qdrant (for local dev / tests)
	$(COMPOSE) up -d postgres redis qdrant

build: ## Build api and web images
	$(COMPOSE) build api web

models: ## Pre-download embedder + reranker into ./data/models
	$(COMPOSE) run --rm ingest download-models

ingest: ## Download + build + index both datasets (default strategy, CORPUS_MAX_DOCS from .env)
	$(COMPOSE) run --rm ingest download --source all
	$(COMPOSE) run --rm ingest build --source all
	$(COMPOSE) run --rm ingest index

ingest-tamir: ## Quick demo ingest: only the tax dataset (~2 minutes on CPU)
	$(COMPOSE) run --rm ingest download --source tamir
	$(COMPOSE) run --rm ingest build --source tamir
	$(COMPOSE) run --rm ingest index

eval: ## Run the default evaluation suite and generate docs/EVAL_REPORT.md
	$(COMPOSE) run --rm eval all --suite default --yes
	$(COMPOSE) run --rm eval report --runs latest

test: ## Unit tests
	$(UV) pytest tests/unit -q

test-integration: ## Integration tests (needs `make infra`)
	$(UV) pytest tests/integration -q -m integration

lint: ## ruff + mypy
	$(UV) ruff check . && $(UV) ruff format --check . && $(UV) mypy legalrag

fmt: ## Auto-format backend
	$(UV) ruff check --fix . && $(UV) ruff format .

reset-data: ## Remove containers AND volumes (asks for confirmation)
	@read -p "This deletes Postgres/Qdrant volumes. Continue? [y/N] " ans; \
	if [ "$$ans" = "y" ]; then $(COMPOSE) down -v; else echo "aborted"; fi

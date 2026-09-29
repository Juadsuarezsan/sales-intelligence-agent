.PHONY: help venv install test lint format typecheck eval data batch demo notebook gitleaks run clean

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	test -x $(BIN)/python || $(PY) -m venv $(VENV)

install: venv ## Install the project with dev extras
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test: ## Run the test suite with coverage (fails under 70 %)
	$(BIN)/pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70

lint: ## ruff + black --check + mypy --strict
	$(BIN)/ruff check .
	$(BIN)/black --check .
	$(BIN)/mypy --strict src/
	$(BIN)/mypy --strict eval/ scripts/

format: ## Apply black and ruff fixes
	$(BIN)/ruff check --fix .
	$(BIN)/black .

typecheck: ## mypy --strict on src/
	$(BIN)/mypy --strict src/

data: ## Download the YC dataset and rebuild data/eval + data/MANIFEST.txt
	$(BIN)/python scripts/download_data.py

eval: ## Run baselines, agent and ablation; writes eval/runs/*.json and eval/RESULTS.md
	$(BIN)/python -m eval.run

batch: ## Process the first 100 eval companies with concurrency 4
	$(BIN)/python scripts/batch_run.py --limit 100 --concurrency 4

demo: ## Rebuild demo/predictions.json from the first 8 eval companies
	$(BIN)/python scripts/batch_run.py --limit 8 --concurrency 4 --out data/outputs --demo demo/predictions.json

notebook: ## Execute notebooks/demo.ipynb in place (stubs, no keys)
	$(BIN)/python scripts/run_notebook.py notebooks/demo.ipynb

gitleaks: ## Scan the repository for secrets (needs the gitleaks binary on PATH)
	gitleaks detect --no-banner --redact --source .

run: ## Start the API locally
	$(BIN)/uvicorn src.api.main:app --reload --port 8000

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

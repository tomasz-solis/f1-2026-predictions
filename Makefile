VENV_BIN ?= .venv/bin
PYTHON ?= $(VENV_BIN)/python
RUFF ?= $(VENV_BIN)/ruff
MYPY ?= $(VENV_BIN)/mypy
PYTEST ?= $(VENV_BIN)/pytest
PRECOMMIT ?= $(VENV_BIN)/pre-commit
ENV_FILE ?= .env.local
EVALUATION_ENV_ARG := $(if $(wildcard $(ENV_FILE)),--env-file $(ENV_FILE),)

GITHUB_SMOKE_TESTS := \
	tests/test_prediction_regression.py \
	tests/test_backtesting.py \
	tests/test_backtest_review_packet.py \
	tests/test_prediction_context_runtime.py \
	tests/test_generate_evaluation_report.py \
	tests/test_race_grid_blending.py

FOCUSED_QUALITY_TESTS := \
	tests/test_artifact_store.py \
	tests/test_prediction_logger.py \
	tests/test_shadow_challenger.py \
	tests/test_dashboard_analytics.py \
	tests/test_grid_validation.py \
	tests/test_qualifying_confidence_progression.py \
	tests/test_systematic_learning.py \
	tests/test_generate_evaluation_report.py

.PHONY: dev-sync ensure-python ensure-ruff ensure-mypy ensure-pytest ensure-precommit fmt lint typecheck typecheck-strict test test-focused test-github-full test-github-smoke test-github-chunk-a test-github-chunk-b test-github-chunk-c test-github-chunk-d test-github-chunk-q test-github-chunk-r test-github-chunk-s test-github-chunk-e test-live-fastf1 check precommit-install precommit evaluation-report evaluation-gate candidate-audit shadow-challenger-audit

dev-sync:
	uv sync --extra dev

ensure-python:
	@command -v "$(PYTHON)" >/dev/null || (echo "Missing $(PYTHON). Run: uv sync --extra dev"; exit 1)

ensure-ruff:
	@command -v "$(RUFF)" >/dev/null || (echo "Missing $(RUFF). Run: uv sync --extra dev"; exit 1)

ensure-mypy:
	@command -v "$(MYPY)" >/dev/null || (echo "Missing $(MYPY). Run: uv sync --extra dev"; exit 1)

ensure-pytest:
	@command -v "$(PYTEST)" >/dev/null || (echo "Missing $(PYTEST). Run: uv sync --extra dev"; exit 1)

ensure-precommit:
	@command -v "$(PRECOMMIT)" >/dev/null || (echo "Missing $(PRECOMMIT). Run: uv sync --extra dev"; exit 1)

fmt: ensure-ruff
	$(RUFF) check src tests scripts app.py predict_weekend.py --fix
	$(RUFF) format src tests scripts app.py predict_weekend.py

lint: ensure-ruff
	$(RUFF) check src tests scripts app.py predict_weekend.py

typecheck: ensure-mypy
	$(MYPY) src

typecheck-strict: ensure-mypy
	$(MYPY) --check-untyped-defs src

test: ensure-pytest
	$(PYTEST) -q

test-focused: ensure-pytest
	$(PYTEST) $(FOCUSED_QUALITY_TESTS) -q

test-github-full: ensure-pytest
	$(PYTEST) tests/ -v --cov=src --cov-report=xml --cov-report=term

test-github-smoke: ensure-pytest
	$(PYTEST) $(GITHUB_SMOKE_TESTS) -q

test-github-chunk-a: ensure-python ensure-pytest
	$(PYTHON) scripts/run_pytest_chunk.py --letters abc -- -q

test-github-chunk-b: ensure-python ensure-pytest
	$(PYTHON) scripts/run_pytest_chunk.py --letters defgh -- -q

test-github-chunk-c: ensure-python ensure-pytest
	$(PYTHON) scripts/run_pytest_chunk.py --letters ilmnop -- -q

test-github-chunk-d: ensure-python ensure-pytest
	$(PYTHON) scripts/run_pytest_chunk.py --letters qrs -- -q

test-github-chunk-q: ensure-python ensure-pytest
	$(PYTHON) scripts/run_pytest_chunk.py --letters q -- -q

test-github-chunk-r: ensure-python ensure-pytest
	$(PYTHON) scripts/run_pytest_chunk.py --letters r -- -q

test-github-chunk-s: ensure-python ensure-pytest
	$(PYTHON) scripts/run_pytest_chunk.py --letters s -- -q

test-github-chunk-e: ensure-python ensure-pytest
	$(PYTHON) scripts/run_pytest_chunk.py --letters tuvwxyz -- -q

test-live-fastf1: ensure-pytest
	FASTF1_LIVE_TESTS=1 $(PYTEST) tests/test_fastf1_live_refresh.py -m live_fastf1 -vv

check: lint typecheck test

precommit-install: ensure-precommit
	$(PRECOMMIT) install

precommit: ensure-precommit
	$(PRECOMMIT) run --all-files

evaluation-report: ensure-python
	$(PYTHON) scripts/generate_evaluation_report.py --year 2026 $(EVALUATION_ENV_ARG)

evaluation-gate: ensure-python
	$(PYTHON) scripts/generate_evaluation_report.py --year 2026 $(EVALUATION_ENV_ARG) --fail-on-gate

candidate-audit: ensure-python
	$(PYTHON) scripts/audit_model_candidates.py --year 2026 $(EVALUATION_ENV_ARG)

shadow-challenger-audit: ensure-python
	$(PYTHON) scripts/audit_shadow_challengers.py --year 2026 $(EVALUATION_ENV_ARG)

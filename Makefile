VENV_BIN ?= .venv/bin
PYTHON ?= $(VENV_BIN)/python
RUFF ?= $(VENV_BIN)/ruff
MYPY ?= $(VENV_BIN)/mypy
PYTEST ?= $(VENV_BIN)/pytest
PRECOMMIT ?= $(VENV_BIN)/pre-commit

GITHUB_SMOKE_TESTS := \
	tests/test_prediction_regression.py \
	tests/test_backtesting.py \
	tests/test_backtest_review_packet.py \
	tests/test_prediction_context_runtime.py \
	tests/test_generate_evaluation_report.py \
	tests/test_race_grid_blending.py

.PHONY: dev-sync ensure-python ensure-ruff ensure-mypy ensure-pytest ensure-precommit fmt lint typecheck typecheck-strict test test-github-full test-github-smoke test-live-fastf1 check precommit-install precommit evaluation-report

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

test-github-full: ensure-pytest
	$(PYTEST) tests/ -v --cov=src --cov-report=xml --cov-report=term

test-github-smoke: ensure-pytest
	$(PYTEST) $(GITHUB_SMOKE_TESTS) -q

test-live-fastf1: ensure-pytest
	FASTF1_LIVE_TESTS=1 $(PYTEST) tests/test_fastf1_live_refresh.py -m live_fastf1 -vv

check: lint typecheck test

precommit-install: ensure-precommit
	$(PRECOMMIT) install

precommit: ensure-precommit
	$(PRECOMMIT) run --all-files

evaluation-report: ensure-python
	$(PYTHON) scripts/generate_evaluation_report.py --year 2026

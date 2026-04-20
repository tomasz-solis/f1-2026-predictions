VENV_BIN ?= .venv/bin
RUFF ?= $(VENV_BIN)/ruff
MYPY ?= $(VENV_BIN)/mypy
PYTEST ?= $(VENV_BIN)/pytest

.PHONY: dev-sync ensure-venv fmt lint typecheck typecheck-strict test test-live-fastf1 check precommit-install precommit

dev-sync:
	uv sync --extra dev

ensure-venv:
	@test -x "$(PYTEST)" || (echo "Missing $(PYTEST). Run: uv sync --extra dev"; exit 1)

fmt: ensure-venv
	$(RUFF) check src tests scripts app.py predict_weekend.py --fix
	$(RUFF) format src tests scripts app.py predict_weekend.py

lint: ensure-venv
	$(RUFF) check src tests scripts app.py predict_weekend.py

typecheck: ensure-venv
	$(MYPY) src

typecheck-strict: ensure-venv
	$(MYPY) --check-untyped-defs src

test: ensure-venv
	$(PYTEST) -q

test-live-fastf1: ensure-venv
	FASTF1_LIVE_TESTS=1 $(PYTEST) tests/test_fastf1_live_refresh.py -m live_fastf1 -vv

check: lint typecheck test

precommit-install: ensure-venv
	$(VENV_BIN)/pre-commit install

precommit: ensure-venv
	$(VENV_BIN)/pre-commit run --all-files

evaluation-report: ensure-venv
	$(VENV_BIN)/python scripts/generate_evaluation_report.py --year 2026

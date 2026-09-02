# trackside-labs — agent guide

## Repo hygiene

`git status` ends every session clean. Every file is committed, gitignored, or
deleted — nothing sits untracked "for now". Untracked files are unprotected: they
survive no branch switch, no `git clean`, and no disk failure.

**Any path written at runtime gets its `.gitignore` entry in the same change that
creates it.** Generated artifacts belong in `ArtifactStore`, not in git. If an
artifact needs to be readable from a fresh clone, that is a deliberate decision
worth stating in the commit message, not a default.

Two traps this repo has already hit:

- `*.backup` does **not** match `<name>.json.rebuild_backup`. Check that a new
  ignore pattern actually matches with `git check-ignore -v <path>`.
- Tracking generated data rots. `data/car_characteristics_snapshot/` had stale
  files tracked under an obsolete naming scheme while the live ones, written by
  `safe_slug` in `src/utils/artifact_paths.py`, were untracked.

Work that is scoped but unimplemented — briefs, tests for helpers that do not
exist yet, shelved research — goes on a `shelved/*` branch, not into the working
tree. See `shelved/challenger-research` and `shelved/dnf-calibration`.

## Verifying before you claim done

**Ruff is pinned to the same exact version in three places, so local lint now
does prove something about CI.** As of 2026-09-02 that version is `0.15.1`, in:

- `pyproject.toml` (`ruff==0.15.1`)
- `.pre-commit-config.yaml` (`rev: v0.15.1`)
- `.github/workflows/lint.yml` (`pip install ruff==0.15.1`)

**Bump all three together, or the drift returns.** They were previously
`>=0.9,<1` / `v0.9.6` / `0.9.6`, which let the local venv run a formatter CI did
not have — three files sat unformatted on `master` because of it.

```bash
uv sync --extra dev
uv run ruff check src tests scripts app.py predict_weekend.py
uv run ruff format --check src tests scripts app.py predict_weekend.py
make typecheck MYPY=mypy
```

Tests run in alphabetic chunks in CI over **tracked files only**; a bare
`uv run pytest` picks up untracked files and can die at collection, stopping
every tracked test from running. Use `make test-github-chunk-N`.

## House conventions

- uv-first: `uv sync --extra dev`, `uv run <cmd>`.
- Model results and their verdicts live in `docs/MODEL_LEDGER.md`. Append and
  supersede; do not rewrite past entries when a later result contradicts them.
- Every measured claim records the champion baseline it was measured against.
  The prediction cache key does **not** cover code version, so a cached
  "champion" prediction may predate the model it is being credited to.

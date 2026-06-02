# Smoke-Session Inspector (Phase 2)

Read-only FastF1 session inspection used to lock the smoke-session set
defined in `docs/design/matched_lap_extractor_smoke_sessions.md`.

## Scope

In scope:

- load FastF1 sessions;
- count laps, retirements, weather samples, track-status events;
- identify qualifying-segment reach per driver;
- emit JSON + text evidence files the analyst attaches alongside the
  smoke-session lock.

Out of scope (Phase 3 - matched-lap extractor):

- pairing teammate laps;
- classifying laps as dry / wet / unreliable;
- emitting `matched_pair` or `skipped_pair` rows;
- deciding what counts as a comparable lap.

The deletion test, from master execution plan Phase 2:
could this package be deleted after Phase 2 closes without losing
anything Phase 3 needs to implement? If yes, scope is right. If no,
the extractor has leaked into Phase 2.

## Usage

```bash
python scripts/inspect_smoke_sessions.py \
    --cache-dir data/raw/.fastf1_cache \
    --output-dir data/diagnostics/smoke_session_inspections
```

For each smoke-session candidate, two evidence files are written to
the output directory:

- `<year>_<category>.json` - full structured summary;
- `<year>_<category>.txt` - short human-readable summary
  (also printed to stdout).

The session list is hard-coded in `scripts/inspect_smoke_sessions.py`
under `SMOKE_SESSIONS`. To inspect different sessions, either edit
that list or import `run_inspections` and pass a custom session list.
Track-status values in the text output are FastF1 status-row counts
(`SC_rows`, `VSC_rows`, etc.), not incident counts.

## Layout

```
src/diagnostics/smoke_inspector/
    __init__.py
    inspector.py    # pure summarizer functions over FastF1 DataFrames
    loader.py       # FastF1 dependency boundary
scripts/
    inspect_smoke_sessions.py   # CLI entry point
tests/
    test_smoke_inspector.py     # unit tests using synthetic frames
```

The inspector module does not import `fastf1` directly; the loader
isolates that dependency so the summarizers are testable without
FastF1 installed.

## After Phase 2 closes

Once the smoke-session doc is locked, this package is no longer needed.
Move it to the archive alongside `master_fix_plan.md`, or delete and
restore from git history if needed for retrospective work.

This package is not imported by the production prediction pipeline. If a
production module starts depending on it, the Phase 2 boundary has been
violated.

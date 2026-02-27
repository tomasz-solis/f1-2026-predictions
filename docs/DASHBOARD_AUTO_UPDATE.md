# Dashboard Update Behavior

This guide explains what the app updates automatically and what still requires an explicit script run.

## Automatic During `Generate Prediction`

When the user clicks **Generate Prediction** in `src/dashboard/pages.py` (called by `app.py`):

1. If **Force Data Refresh** is enabled (default OFF), the app clears FastF1 cache entries for the selected race.
2. `auto_update_if_needed(force_recheck=...)` checks for race-result learning updates.
3. If new races are found, it runs `auto_update_from_races()`.
4. `auto_update_practice_characteristics_if_needed(force_recheck=...)` checks/update FP-derived characteristics.
5. Streamlit caches are cleared so the prediction run in the same click uses fresh artifacts.
6. Competitive-session completion checks are re-run before using cached outputs.

This block is race-result ingestion.

### Practice-characteristics auto capture

During weekend predictions, the app also checks completed FP sessions and can update
car characteristics from FP telemetry (FP1/FP2/FP3). By default, the update step
uses cached session-state checks. With **Force Data Refresh** enabled, it re-checks
session completion even when the race was already processed.

State persistence:

- `runtime_state` table (`practice_characteristics` namespace) when DB read/write is enabled
- `data/systems/practice_characteristics_state.json` when file writes are enabled

Backlog coordination:

- `runtime_processing_locks` table lock leases prevent duplicate concurrent application
- lock contention is surfaced as deferred/retried backlog events

Behavior can be tuned in `config/default.yaml` under:

- `baseline_predictor.practice_capture.*`

## Automatic Session Data Retrieval

Qualifying prediction also auto-fetches the best available session data through `src/utils/fp_blending.py`.

- Normal: short-stint blend of `FP3 + FP2 + FP1` (FP3-weighted)
- Sprint (main qualifying): short-stint blend of `Sprint Qualifying + FP1 + Sprint`

## Manual / Explicit Workflows

### 1. Force race update manually

```bash
python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026
```

### 2. Update testing/practice directionality

```bash
python scripts/update_from_testing.py "Testing 1" --year 2026 --sessions "Day 1" --apply
```

This updater is manual.
By default it runs as dry-run; pass `--apply` to write updates.
In DB-enabled modes (`db_only`, `fallback`, `dual_write`), `--apply` writes through
`ArtifactStore` so Supabase stays in sync with file artifacts.

Clarification: dashboard FP auto-capture uses the same underlying testing updater logic for
completed race-weekend FP sessions, but explicit testing-event runs (for example pre-season testing)
still require manual script execution.

## Cache Locations

- Main FastF1 cache: `data/raw/.fastf1_cache`
- Testing updater cache (default): `data/raw/.fastf1_cache_testing`

If cache corruption is suspected for testing pulls, run with:

```bash
python scripts/update_from_testing.py "Testing 1" \
  --year 2026 \
  --sessions "Day 1" \
  --force-renew-cache \
  --apply
```

## What Is Not Automatic

- Pre-season testing directionality extraction.
- Historical notebook validation runs.
- Manual backfill decisions for special analysis workflows.

## Operational Notes

The dashboard auto-update depends on FastF1 schedule/session availability. If session data is delayed, updates may appear later even when race date has passed.

Competitive-session refresh uses fail-closed handling:

- unknown completion status does not silently downgrade ACTUAL grid source to PREDICTED
- transient FastF1 failures emit runtime alerts/counters for visibility

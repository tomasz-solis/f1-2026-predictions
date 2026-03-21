# Dashboard Update Behavior

This guide shows what the app does on a normal prediction click and what still needs a worker or explicit script run.

## During `Predict`

When the user clicks **Predict** in `src/dashboard/pages.py` (called by `app.py`):

1. Weekend format is resolved for the selected race.
2. Session-boundary freshness is checked for the selected race.
3. The dashboard loads the warmed persisted prediction for the current checkpoint.
4. If warmup has not caught up yet, the dashboard serves the latest warmed checkpoint instead.

The request path is read-only. It does not:

- run `auto_update_if_needed(...)`
- run `auto_update_practice_characteristics_if_needed(...)`
- clear FastF1 caches to force a refresh
- generate new prediction artifacts inline

### Practice-characteristics auto capture

Practice-derived car characteristics are still updated automatically, but by
warmup/session automation workers rather than by the dashboard request path.

State persistence:

- `runtime_state` table (`practice_characteristics` namespace) when DB read/write is enabled
- `data/systems/practice_characteristics_state.json` when file writes are enabled

Backlog coordination:

- `runtime_processing_locks` table lock leases prevent duplicate concurrent application
- lock contention is surfaced as deferred/retried backlog events

Behavior can be tuned in `config/default.yaml` under:

- `baseline_predictor.practice_capture.*`

## Session Data During Artifact Generation

When warmup or session automation generates prediction artifacts, qualifying
still uses the best available session data through `src/utils/fp_blending.py`.
That blending happens while persisted artifacts are built, not as extra work in
the Streamlit request path.

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

Clarification: warmup/session automation uses the same underlying testing updater logic for
completed race-weekend FP sessions, but explicit testing-event runs (for example pre-season
testing) still require manual script execution.

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

## What Is Not Automatic In The Dashboard

- Pre-season testing directionality extraction.
- Historical notebook validation runs.
- Manual backfill decisions for special analysis workflows.

## Operational Notes

The dashboard still performs lightweight schedule/boundary checks. If FastF1 session data is delayed,
the app may keep serving the last warmed checkpoint until warmup can persist a newer one.

When prediction artifacts are generated, competitive-session resolution uses fail-closed handling:

- unknown completion status does not silently downgrade ACTUAL grid source to PREDICTED
- transient FastF1 failures emit runtime alerts/counters for visibility

## Background Automation (No Click Required)

For fully automatic post-session updates (without a user click), run:

```bash
python scripts/run_session_automation.py --year 2026 --interval-seconds 300
```

This worker polls recent events, applies session updates, can auto-generate prediction snapshots
for the latest completed session, and reconciles actuals after race completion.

To warm dashboard precompute payloads ahead of user clicks, run:

```bash
python scripts/warmup_precompute.py --year 2026 --require-db
```

This worker is checkpoint-aware and idempotent. It warms the next 3 races,
stores missing weather scenarios only, and updates the ready-race horizon index
used by the race dropdown.

For production, keep the dashboard request path read-only so this warmup owns
freshness outside the Streamlit request thread.

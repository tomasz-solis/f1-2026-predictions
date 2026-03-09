# Persistence and Supabase

This page documents how persistence works in the current runtime path, including target-aware prediction accuracy storage.

## Current Runtime Integration

Artifact persistence is already used in active runtime paths:

- `src/predictors/baseline/data_mixin.py` (load core artifacts)
- `src/systems/updater.py` (save updated car characteristics)
- `src/utils/prediction_logger.py` (save/load prediction tracking payloads)
- `src/utils/accuracy_snapshots.py` (build per-target accuracy snapshot payloads)
- `src/dashboard/cache.py` (artifact version checks for cache invalidation)
- `src/predictors/baseline/race/preparation_mixin.py` (missing-driver debut-year lookup)

Core layer:

- `src/persistence/artifact_store.py`
- `src/persistence/config.py`
- `src/persistence/db.py`
- `src/persistence/runtime_state_store.py`
- `src/utils/operational_observability.py`

## Storage Modes

Storage mode comes from `USE_DB_STORAGE` (default: `file_only`):

- `file_only`: read/write local JSON only
- `db_only`: read/write Supabase only
- `fallback`: read DB first, then file fallback; writes DB only
- `dual_write`: write both DB and file; reads DB first, then file fallback

If mode is not `file_only`, these env vars are required:

- `SUPABASE_URL`
- `SUPABASE_KEY` (`service_role` key for backend writes)

`SUPABASE_URL` is validated at startup and must be an `https://` URL. Common typos
like `ttps://...` now fail fast with an explicit error.

## Supabase Assets In Repo

- SQL migration: `migrations/001_create_artifacts_table.sql`
- SQL migration: `migrations/002_create_runtime_state_and_operational_tables.sql`
- SQL migration: `migrations/003_harden_rls_policies.sql`
- Connection test: `scripts/test_supabase_connection.py`
- Backfill utility: `scripts/backfill_to_db.py` (migrates `driver_debuts.csv` too)
- Snapshot backfill utility: `scripts/backfill_accuracy_snapshots.py`
- Predictor + storage smoke test: `scripts/test_predictor_with_db.py`

No new Supabase tables or migrations are required for prediction accuracy. The existing generic `artifacts` table stores both raw prediction artifacts and derived accuracy snapshots.

## Prediction Accuracy Artifacts

The target-aware accuracy flow uses two artifact types:

- `prediction`
- `accuracy_snapshot`

`prediction` remains the source of truth. It stores:

- top-level legacy `qualifying` / `race` fields
- canonical `targets`
- canonical `actuals.targets`
- metadata including `weekend_format`

`accuracy_snapshot` stores one scored target at one checkpoint:

- artifact key: `YYYY::Race Name::CHECKPOINT::TARGET_KEY`
- metadata:
  - `year`
  - `race_name`
  - `checkpoint_session`
  - `weekend_format`
  - `target_key`
  - `target_session`
  - `predicted_at`
  - `generated_at`
  - `source_run_id`
  - `eligible`
- metrics:
  - `field_size`
  - `overall_mae`
  - `top_3_hits`
  - `top_3_pct`
  - `top_10_hits`
  - `top_10_pct`
  - `exact_accuracy`
  - `within_1`
  - `within_3`
  - `correlation`

The dashboard reads `accuracy_snapshot` artifacts first and falls back to raw prediction payloads when snapshots are missing.

## Runtime State and Operational Tables

When DB mode is enabled, dashboard runtime also uses:

- `runtime_state`:
  - namespace/key state for event-boundary snapshots and practice update progress
- `runtime_processing_locks`:
  - lease-based lock rows for practice backlog coordination across workers
- `operational_events`:
  - best-effort counters and alerts written by runtime observability hooks

## Baseline Artifacts

These keys are relevant for the baseline predictor stack:

- `car_characteristics` -> `2026::car_characteristics`
- `driver_characteristics` -> `2026::driver_characteristics`
- `track_characteristics` -> `2026::track_characteristics`
- `driver_debuts` -> `driver_debuts`

## Recommended Rollout Path

1. Run migrations in Supabase SQL Editor:
   - `migrations/001_create_artifacts_table.sql`
   - `migrations/002_create_runtime_state_and_operational_tables.sql`
   - `migrations/003_harden_rls_policies.sql`
2. Validate credentials and table access:
   - `uv run --active python scripts/test_supabase_connection.py`
3. Dry-run data migration:
   - `uv run --active python scripts/backfill_to_db.py --dry-run`
4. Run backfill with DB writes enabled:
   - set `USE_DB_STORAGE=dual_write` (or `db_only` for isolated testing)
   - `uv run --active python scripts/backfill_to_db.py`
5. Run predictor smoke test:
   - `uv run --active python scripts/test_predictor_with_db.py`
6. Backfill missing accuracy snapshots from stored prediction truth:
   - `uv run --active python scripts/backfill_accuracy_snapshots.py --year 2026 --dry-run`
   - `uv run --active python scripts/backfill_accuracy_snapshots.py --year 2026`
7. Verify debut artifact:
   - `driver_debuts::driver_debuts` should be present and readable via `ArtifactStore`.
8. Verify runtime tables:
   - prediction click writes/updates `runtime_state` rows
   - concurrent practice runs create lock contention in `runtime_processing_locks`
   - runtime alerts/counters appear in `operational_events`
9. Verify race-learning dedupe state:
   - namespace `race_learning` has per-season records in `runtime_state`
   - `auto_update_from_races()` does not re-learn already processed races after restart

If your tables already existed before these secure defaults, run
`migrations/003_harden_rls_policies.sql` as a one-time hardening step to enforce RLS,
add explicit `service_role` policies, and revoke `anon`/`authenticated` access.

## Current Caveats

- Prediction tracking UI now lists historical predictions through `ArtifactStore` with file fallback.
- Accuracy snapshots are derived data. If they are missing, the dashboard can recompute from raw prediction truth, but backfilling them improves chart load time and keeps automation outputs explicit.
- Historic sprint weekends may have real gaps for early main `Q/R` targets if those target forecasts were never stored.
- For dashboard usage during migration, `dual_write` remains the safest mode.
- File-based `list_artifacts()` fallback is intentionally minimal outside mapped artifact types.
- Runtime-state writes and external updater side effects are not a single distributed transaction.

Keep `file_only` as the default for local-only usage. Use `fallback` or `db_only` when you need DB-first reads, and `dual_write` when migrating while preserving local history files.

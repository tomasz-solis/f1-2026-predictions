# Formula 1 2026 Prediction Engine

This repository predicts F1 race weekends for the 2026 season using simulation-based models.

## What Runs In Production

The dashboard shell lives in:

- `app.py`
- `src/dashboard/layout.py`
- `src/dashboard/pages.py`

The main `Prediction` tab then runs through:

- `src/dashboard/live_prediction_flow.py`
- `src/dashboard/prediction_flow.py`
- `src/dashboard/update_flow.py`
- `src/dashboard/rendering.py`
- `src/dashboard/cache.py`

The other shipped tabs use:

- `src/dashboard/team_comparison.py`
- `src/dashboard/accuracy.py`
- `src/dashboard/accuracy_view.py`

Core prediction services on the active runtime path:

- `src/predictors/baseline_2026.py` (`Baseline2026Predictor`)
- `src/systems/systematic_learning.py`
- `src/systems/weight_schedule.py`
- `src/utils/fp_blending.py`

`src/predictors/qualifying.py` and `src/predictors/race.py` are compatibility wrappers that delegate to `Baseline2026Predictor`.
The user-facing tab is called `Prediction`, but the route is still handled by `render_live_prediction_page()` in code for backwards compatibility.

Current dashboard tabs:

- `Prediction`
- `Team Comparison`
- `Prediction Accuracy`
- `Model & Learning`
- `Contact`

If you want the component relationship map, see `ARCHITECTURE.md`.

## Predictor Structure

`Baseline2026Predictor` is a composed class, split into focused modules.

- `src/predictors/baseline/data_mixin.py`: artifact loading, blended team strength, compound selection helpers
- `src/predictors/baseline/qualifying_mixin.py`: qualifying, sprint-qualifying, and sprint-race entrypoints
- `src/predictors/baseline/race/`: race prep, params, and lap-by-lap prediction mixins
- `src/predictors/baseline_2026.py`: thin composition/entrypoint

## Quick Start

Runtime target: Python `3.11.x` (repository metadata/lockfile currently excludes `3.14`).

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local development, use the project `.venv`:

```bash
uv sync --extra dev
source .venv/bin/activate
```

If port `8501` is already in use:

```bash
streamlit run app.py --server.port 8502
```

## Prediction Logic

### Qualifying

- Builds team strength from baseline + testing directionality + current season performance (weight schedule).
- Builds a short-stint qualifying signal from available sessions (weighted by session relevance and recency).
  - Normal weekend: blends `FP3`, `FP2`, `FP1` (FP3-weighted)
  - Sprint weekend (main qualifying): blends `Sprint Qualifying`, `FP1`, `Sprint`
- If no weekend practice pace is available, uses testing short-run profile blend as fallback (when enough teams have profiles).
- If neither practice nor testing profile fallback is available, uses model-only path.
- Applies model-only teammate/experience stabilization and learned per-driver/per-teammate calibration adjustments.
- Runs Monte Carlo and returns median position with confidence bands.

### Race

Lap-by-lap Monte Carlo simulation (50 runs) with:
- Multi-compound pit strategy generation (FIA mandates ≥2 compounds per dry race)
- Tire degradation (compound-specific slopes), fresh tire advantage, fuel effect
- Traffic effect (P1-5: 5% better tire life, P16+: 5% worse)
- Track-specific pit loss (Monaco: 19s, Singapore: 24s)
- Grid influence, driver skill, lap-1 chaos, safety car luck, DNF probability
- Strategy timing bias (undercut/overcut-aware) using track overtaking and grid context
- Overtaking realism by zone (back/mid easier, front harder) with capped total position gains
- Learned per-driver/per-teammate calibration adjustments in race scoring
- Podium probability derived from ranked outcomes (with monotonic smoothing by final order)

Outputs: Finish order + compound strategy distribution + pit window histogram.

## Data Update Flows

### 1. Persisted predictions on Predict

When you click **Predict**, the app:

- resolves the current weekend format,
- checks whether a newer session boundary exists for the selected race,
- loads the matching warmed prediction from persisted storage,
- falls back to the last warmed checkpoint if the live boundary is ahead,
- does not refresh artifacts or rerun simulations in the request path.

### 1b. Optional background automation (no click required)

Run periodic automation to refresh right after session completion:

```bash
python scripts/run_session_automation.py --year 2026 --interval-seconds 300
```

This worker applies post-session updates, can auto-generate prediction snapshots,
and reconciles actuals after race completion.

### 1c. Warm precompute for instant dropdown load

Run checkpoint-aware warmup outside Streamlit so ready races load instantly:

```bash
python scripts/warmup_precompute.py --year 2026
```

Debug options:

```bash
python scripts/warmup_precompute.py --year 2026 --dry-run --verbose
python scripts/warmup_precompute.py --year 2026 --require-db
```

What it does:

- reads the current schedule and picks the next race plus a 3-race horizon
- resolves checkpoint readiness from actual session data
  - conventional: `PRE`, `FP1`, `FP2`, `FP3`, `Q`
  - sprint: `PRE`, `FP1`, `SQ`, `Sprint`, `Q`
- exits early (code `0`) when the expected checkpoint is not ready yet
- computes/stores missing base features once per `(race, checkpoint, artifact, boundary)`
- computes/stores only missing weather scenarios (`dry`, `mixed`, `rain`)
- updates the precompute horizon index used by the race dropdown filter

The Streamlit request path is intentionally read-only. Warmup owns prediction
freshness so user clicks do not mutate artifacts or rerun simulations.

Cron example (Render preheat every 5 minutes):

```bash
*/5 * * * * cd /path/to/formula1-2026 && /path/to/formula1-2026/.venv/bin/python scripts/warmup_precompute.py --year 2026 --require-db >> /tmp/f1_warmup.log 2>&1
```

If Supabase accumulates warmup rows from older artifact hashes, prune only the
runtime-state precompute cache with:

```bash
python scripts/prune_stale_precompute_state.py --year 2026 --require-db
python scripts/prune_stale_precompute_state.py --year 2026 --require-db --apply
```

This cleanup is intentionally narrow. It removes stale rows only from the
runtime-state warmup namespaces and keeps saved checkpoint prediction artifacts
used by the accuracy dashboard.

### 2. Manual race update

```bash
python scripts/update_from_race.py "Bahrain Grand Prix" --year 2026
```

### 3. Manual testing/practice directionality update

```bash
python scripts/update_from_testing.py "Testing 1" --year 2026 --sessions "Day 1" --apply
```

To combine all available testing days, omit `--sessions`:

```bash
python scripts/update_from_testing.py "Testing 1" --year 2026 --apply
```

Useful flags:

```bash
python scripts/update_from_testing.py "Testing 1" \
  --year 2026 \
  --backend auto \
  --cache-dir data/raw/.fastf1_cache_testing \
  --session-aggregation laps_weighted \
  --run-profile balanced \
  --force-renew-cache \
  --dry-run
```

Note: this script now defaults to dry-run mode; use `--apply` to persist changes.
In DB-enabled modes (`db_only`, `fallback`, `dual_write`), `--apply` also persists
`car_characteristics` to Supabase via `ArtifactStore`.

If testing/practice profiles are missing, Team Comparison shows an availability
message instead of neutral placeholder radar values.

Testing cache defaults to `data/raw/.fastf1_cache_testing`.

## Important Data Files

- `data/processed/car_characteristics/2026_car_characteristics.json`
- `data/processed/track_characteristics/2026_track_characteristics.json`
- `data/processed/driver_characteristics.json`

## Persistence and Supabase

Artifact persistence is wired through `ArtifactStore` in active runtime code paths:

- `src/predictors/baseline/data_mixin.py`
- `src/systems/updater.py`
- `src/utils/prediction_logger.py`
- `src/dashboard/cache.py`
- `src/predictors/baseline/race/preparation_mixin.py` (driver debut lookup for missing-driver fallback)

Prediction accuracy updates also write adaptive calibration state through:

- `src/systems/systematic_learning.py`
- `runtime_state` namespace `race_learning` (DB-backed modes)
- `data/learning_state.json` (file-backed fallback / local mode)

Storage mode is controlled by `USE_DB_STORAGE`:

- `file_only` (default)
- `db_only`
- `fallback`
- `dual_write`

When mode is not `file_only`, both `SUPABASE_URL` and `SUPABASE_KEY` are required.
Use a backend `service_role` key for `SUPABASE_KEY` (not anon).
`SUPABASE_URL` must be an `https://` URL.

Runtime state and operational telemetry are also persisted when DB mode is enabled:

- `src/persistence/runtime_state_store.py`
- `src/utils/operational_observability.py`
- `runtime_state` table (event-boundary + practice progress state)
- `runtime_processing_locks` table (practice backlog lock leases)
- `operational_events` table (counters + alerts stream)

Artifacts used in the baseline path include:

- `car_characteristics` (`2026::car_characteristics`)
- `driver_characteristics` (`2026::driver_characteristics`)
- `track_characteristics` (`2026::track_characteristics`)
- `driver_debuts` (`driver_debuts`)

Supabase assets in the repo:

- Migration: `migrations/001_create_artifacts_table.sql`
- Migration: `migrations/002_create_runtime_state_and_operational_tables.sql`
- Migration: `migrations/003_harden_rls_policies.sql`
- Connectivity check: `scripts/test_supabase_connection.py`
- Backfill utility: `scripts/backfill_to_db.py` (includes `driver_debuts.csv` migration)
- Targeted dashboard datapoint compare/sync: `scripts/sync_dashboard_datapoints_to_db.py`
- Predictor/storage smoke test: `scripts/test_predictor_with_db.py`

Rollout guidance:

- `file_only`: default local mode.
- `fallback`: DB-first reads with file fallback (recommended when validating Supabase reads).
- `dual_write`: safest migration mode if you still rely on local prediction-history files.

## Modules Outside Main Dashboard Path

- Bayesian ranking components (`src/models/bayesian.py`)
- Legacy learning-history module (`src/systems/learning.py`)
- Additional scripts and legacy-compatible interfaces

These are still useful for experiments and extensions, but the dashboard runtime
path is the baseline predictor stack listed above.

## Documentation

- `ARCHITECTURE.md`
- `CONFIGURATION.md`
- `docs/README.md`
- `docs/WEEKEND_PREDICTIONS.md`
- `docs/FP_BLENDING_SYSTEM.md`
- `docs/DASHBOARD_AUTO_UPDATE.md`
- `docs/PREDICTION_TRACKING.md`
- `docs/WEIGHT_SCHEDULE_GUIDE.md`
- `docs/COMPOUND_ANALYSIS.md` - Tire compound performance system
- `docs/PERSISTENCE_SUPABASE.md` - ArtifactStore modes, migration flow, and current rollout status
- `docs/WARMUP_PRECOMPUTE.md` - Scheduled checkpoint-aware warmup for instant dashboard load

## Tests

```bash
.venv/bin/pytest tests/
```

Project check pipeline (lint + mypy + pytest):

```bash
make check
```

Run checks automatically before every commit:

```bash
make precommit-install
```

Run the same hook suite manually at any time:

```bash
make precommit
```

Optional stricter mypy pass (checks bodies of untyped functions):

```bash
make typecheck-strict
```

Nightly live-network FastF1 checks (CI):

```bash
make test-live-fastf1
```

Targeted examples:

```bash
.venv/bin/pytest tests/test_baseline_2026_integration.py
.venv/bin/pytest tests/test_dashboard_smoke.py
.venv/bin/pytest tests/test_testing_updater.py
```

## Backtesting and Ablations

Run a baseline 2025 backtest (full schedule if available via FastF1):

```bash
python scripts/backtest_2025_season.py --year 2025
```

Fast iteration on a subset:

```bash
python scripts/backtest_2025_season.py --year 2025 --max-races 6
```

Run ablations with overfitting checks (train/test split + generalization gap):

```bash
python scripts/backtest_2025_season.py \
  --year 2025 \
  --max-races 8 \
  --experiment "higher_grid_anchor:baseline_predictor.race.grid_anchor.base=0.45" \
  --experiment "lower_sc_noise:baseline_predictor.race.safety_car_luck_range=0.15"
```

Outputs are written under `reports/backtest_2025/`:

- per-experiment race metrics (`race_results.csv`)
- summary payloads (`summary.json`)
- cross-experiment comparison (`experiment_comparison.csv`)
- recommendation report with guardrails against overfitting (`recommendations.md`)
- race MAE distribution plot when matplotlib is available (`race_mae_distribution.png`)

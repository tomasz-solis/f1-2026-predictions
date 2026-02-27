# Formula 1 2026 Prediction Engine

This repository predicts F1 weekends for the 2026 season using a simulation-first approach.

## What Runs In Production (Current Path)

The Streamlit app and the main weekend flow use:

- `app.py` (entrypoint)
- `src/dashboard/cache.py`
- `src/dashboard/layout.py`
- `src/dashboard/pages.py`
- `src/dashboard/prediction_flow.py`
- `src/dashboard/rendering.py`
- `src/dashboard/update_flow.py`
- `src/predictors/baseline_2026.py` (`Baseline2026Predictor`)
- `src/systems/systematic_learning.py`
- `src/systems/weight_schedule.py`
- `src/utils/fp_blending.py`

`src/predictors/qualifying.py` and `src/predictors/race.py` are compatibility wrappers that delegate to `Baseline2026Predictor`.

Current dashboard tabs:

- `Live Prediction`
- `Model & Learning`
- `Prediction Accuracy`
- `Contact`

## Predictor Structure (Current)

`Baseline2026Predictor` is now a composed class, not a single monolithic implementation file.

- `src/predictors/baseline/data_mixin.py`: artifact loading, blended team strength, compound selection helpers
- `src/predictors/baseline/qualifying_mixin.py`: qualifying and sprint-quali flow
- `src/predictors/baseline/race/`: race prep, params, and lap-by-lap prediction mixins
- `src/predictors/baseline_2026.py`: thin composition/entrypoint

## Quick Start

Runtime target: Python `3.11.x` (repository metadata/lockfile currently excludes `3.14`).

```bash
pip install -r requirements.txt
streamlit run app.py
```

If port `8501` is already in use:

```bash
streamlit run app.py --server.port 8502
```

## Prediction Logic (As Implemented)

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

### 1. Automatic in dashboard

When you click **Generate Prediction**, the app:

- optionally clears FastF1 race cache first (`Force Data Refresh`, default OFF),
- checks for race-result learning updates,
- checks completed FP sessions (FP1/FP2/FP3) for practice-based characteristic updates,
- clears Streamlit caches when refresh/update steps run so the same click uses fresh artifacts,
- always runs FastF1 completion checks for competitive sessions on each click,
- blocks silent ACTUAL -> PREDICTED downgrades when transient FastF1 failures occur.

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

If testing/practice profiles are missing, Team Comparison will show an availability message
instead of neutral placeholder radar values.

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
- Predictor/storage smoke test: `scripts/test_predictor_with_db.py`

Rollout guidance:

- `file_only`: default local mode.
- `fallback`: DB-first reads with file fallback (recommended when validating Supabase reads).
- `dual_write`: safest migration mode if you still rely on local prediction-history files.

## What Exists But Is Not The Main Dashboard Path

- Bayesian ranking components (`src/models/bayesian.py`)
- Legacy learning-history module (`src/systems/learning.py`)
- Additional scripts and legacy-compatible interfaces

These remain useful for experiments and extensions, but the app runtime path is the baseline predictor stack listed above.

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

## Tests

```bash
pytest tests/
```

Nightly live-network FastF1 checks (CI):

```bash
FASTF1_LIVE_TESTS=1 pytest tests/test_fastf1_live_refresh.py -m live_fastf1
```

Targeted examples:

```bash
pytest tests/test_baseline_2026_integration.py
pytest tests/test_dashboard_smoke.py
pytest tests/test_testing_updater.py
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

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

`src/predictors/__init__.py` re-exports `Baseline2026Predictor` so scripts can import from the package root.
The user-facing tab is called `Prediction`, but the route is still handled by `render_live_prediction_page()` in code for backward compatibility.

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

Session-data blending and any FastF1-dependent artifact updates happen during
warmup or session automation, not inside the click path itself.

### Serving architecture

The dashboard is intentionally persisted-only.

Clicking **Predict weekend** does not call FastF1, does not rebuild features,
and does not rerun simulations. It loads the latest warmed artifact that
matches the current checkpoint and artifact hash.

This is a product choice, not just an implementation shortcut:

- the page responds immediately instead of making users wait 30-40 seconds for a live recompute
- the same race/checkpoint request resolves the same way for every user on the same deployed revision
- FastF1 delays, cache misses, and rate-limit pain stay in the warmup worker instead of the request path

The tradeoff is freshness control. A user cannot force a one-off refresh from
the UI. If a newer session boundary is available but warmup has not caught up
yet, the dashboard keeps serving the last warmed checkpoint until the worker
persists the newer one.

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
python scripts/update_from_race.py "Australian Grand Prix" --year 2026
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

Testing note: Bahrain 2026 test data is useful for seeding car-characteristic
baselines, but raw test pace should not be treated as representative team order.

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
- Migration: `migrations/004_normalize_prediction_artifact_keys.sql`
- Connectivity check: `scripts/test_supabase_connection.py`
- Backfill utility: `scripts/backfill_to_db.py` (includes `driver_debuts.csv` migration)
- Dashboard artifact cleanup: `scripts/normalize_dashboard_artifacts_in_db.py`
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

These are still useful for experiments and extensions, but the live prediction
path is the baseline predictor stack listed above.

The Bayesian module updates driver skill after completed races. It is not the
main live prediction engine. Weekend predictions come from the Monte Carlo
qualifying and race pipeline plus the early-season team-strength blending logic.

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

## Model Performance

Latest checked numbers live in `data/backtesting/2025_backtest_results.json`.

Using `python scripts/backtest_2025_season.py --year 2025`, the current model
scored all 24 races available in the local cache. On that full run it averaged:

- qualifying MAE: `4.37` places
- race MAE: `4.83` places

For a fair naive comparison, use the 23-race overlap where both the model and a
`previous race classification` baseline were available:

- qualifying MAE: model `4.33`, naive `3.94`
- race MAE: model `4.84`, naive `4.59`
- race MAE improvement vs naive: `-0.25` places

The honest read is not flattering: on the full cached 2025 season, the naive
`previous race classification` baseline beats the model on both qualifying and
race MAE over the shared window.

The checked-in summary now keeps per-race predicted and actual top-10 rows, so
you can inspect real race-by-race misses directly in
`reports/backtest_2025/baseline/race_results_detailed.json` or the mirrored
`data/backtesting/2025_backtest_results.json`.

### Why the naive baseline wins on 2025 and why it will not in 2026

The naive baseline copies last race's finishing order as this race's prediction.
Under stable regulations from 2022 through 2025, where car performance barely
changes between rounds, that works surprisingly well. The model cannot beat it
because there is almost no new information to exploit. The same cars are fast at
every track, and the model's overhead from simulation variance and parameter
uncertainty costs more than it gains.

2026 is a different problem. New aerodynamic regulations, new power units,
active aero replacing DRS, and Cadillac joining as an 11th team mean there is
no "last race" to copy at the start of the season. The naive baseline is
literally undefined for the opening race and unreliable for the first handful
of rounds while the competitive order is still settling.

This is the part of the calendar where simulation-based prediction adds value:

- Pre-season: blends constructor reputation priors with testing telemetry profiles
  such as short-run pace, corner speeds, and tire degradation slopes to produce a
  credible pre-race-1 ranking.
- Early season: Bayesian form updates move driver ratings toward observed
  results, while the naive baseline is still anchored to Round 1.
- Probabilistic output: the model produces confidence bands, podium
  probabilities, and DNF risk per driver. The naive baseline produces one rank
  with no uncertainty estimate.

The 2025 backtest is here to show the model's floor. In the worst case of
stable regulations and almost no information advantage, it stays within 0.25
places of a strong heuristic. The model is built for the regulation-reset case
where that heuristic fails entirely.

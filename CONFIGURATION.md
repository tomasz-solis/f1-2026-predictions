# Configuration Guide

This project uses two main config files for different purposes:

- `config/default.yaml`: model and runtime parameters.
- `config/production_config.json`: strategy metadata used by helper utilities and experiments.

## 1. `config/default.yaml`

### What is actively used by the dashboard runtime path

`Baseline2026Predictor` reads values through `src/utils/config_loader.py`.
The most relevant active section is:

- `baseline_predictor.qualifying.*`
- `baseline_predictor.race.*`

Examples:

- `baseline_predictor.qualifying.noise_std_normal`
- `baseline_predictor.qualifying.team_weight`
- `baseline_predictor.qualifying.testing_short_run_modifier_scale`
- `baseline_predictor.compound_selection.*` (tire compound selection thresholds)
- `baseline_predictor.race.base_chaos.dry`
- `baseline_predictor.race.grid_weight_min`
- `baseline_predictor.race.pace_weight_base`
- `baseline_predictor.race.dnf_rate_final_cap`
- `baseline_predictor.race.testing_long_run_modifier_scale`
- `baseline_predictor.practice_capture.*` (dashboard FP auto-capture behavior)

### Other sections in `default.yaml`

Sections such as `bayesian`, `race`, and `qualifying` are still useful for
other modules/scripts, but they are not the primary scoring knobs for the
baseline predictor race/qualifying simulation path.

The top-level `learning` section is active. It controls adaptive calibration
sample gates, adjustment scales, and interval widening thresholds used by
`src/systems/systematic_learning.py`.

## 2. `config/production_config.json`

Used by `src/utils/config.py` (`ProductionConfig`) for strategy metadata and expected MAE references.

It is not the main parameter source for `Baseline2026Predictor` scoring in the dashboard path.

## 3. Persistence Environment Configuration

Artifact storage mode is controlled by environment variables in `src/persistence/config.py`.

### Storage mode

- `USE_DB_STORAGE=file_only` (default): file-backed artifacts only.
- `USE_DB_STORAGE=db_only`: Supabase-backed artifacts only.
- `USE_DB_STORAGE=fallback`: read DB first, fall back to file; writes DB.
- `USE_DB_STORAGE=dual_write`: write both DB and file (recommended migration mode).

### Supabase credentials

When `USE_DB_STORAGE` is not `file_only`, both are required:

- `SUPABASE_URL`
- `SUPABASE_KEY` (`service_role` key for backend writes)

`SUPABASE_URL` must be an `https://` URL (for example, `https://<project>.supabase.co`).

Related docs/scripts:

- `docs/PERSISTENCE_SUPABASE.md`
- `migrations/001_create_artifacts_table.sql`
- `migrations/002_create_runtime_state_and_operational_tables.sql`
- `migrations/003_harden_rls_policies.sql`
- `migrations/004_normalize_prediction_artifact_keys.sql`
- `scripts/test_supabase_connection.py`
- `scripts/normalize_dashboard_artifacts_in_db.py`
- `scripts/backfill_to_db.py`

In DB-backed modes, the same Supabase credentials are used for:

- artifact persistence (`artifacts`),
- runtime state persistence (`runtime_state`),
- practice backlog lock leases (`runtime_processing_locks`),
- operational counters/alerts stream (`operational_events`).

## Common Changes

### Change qualifying team vs driver weighting

Edit:

- `config/default.yaml` -> `baseline_predictor.qualifying.team_weight`
- `config/default.yaml` -> `baseline_predictor.qualifying.skill_weight`

These should sum to `1.0`.

### Change tire compound selection thresholds

Edit:

- `baseline_predictor.compound_selection.high_stress_threshold` (default 3.5)
- `baseline_predictor.compound_selection.low_stress_threshold` (default 2.5)
- `baseline_predictor.compound_selection.default_stress_fallback` (default 3.0)

### Change race volatility / chaos

Edit:

- `baseline_predictor.race.base_chaos.dry`
- `baseline_predictor.race.base_chaos.wet`
- `baseline_predictor.race.lap1_chaos.*`
- `baseline_predictor.race.teammate_variance_std`
- `baseline_predictor.race.track_chaos_multiplier`

### Change DNF behavior

Edit:

- `baseline_predictor.race.dnf_rate_historical_cap`
- `baseline_predictor.race.dnf_rate_final_cap`

### Change adaptive-learning safeguards

Edit:

- `learning.min_samples` - minimum stored samples before a learned driver or teammate correction can be applied
- `learning.driver_error_scale` - how strongly driver EMA error moves position scoring
- `learning.teammate_gap_scale` - how strongly teammate-gap error moves position scoring
- `learning.max_adjustment` - cap for learned position movement
- `learning.interval_min_samples` - minimum interval residual samples before learned interval widening applies
- `learning.interval_target_coverage` - target empirical coverage for learned interval radius
- `learning.interval_max_adjustment` - cap for learned interval radius

Learning updates are also gated in code: retrospective predictions, duplicate
run IDs, missing actuals, and tiny actual overlaps are skipped instead of
training adaptive calibration.

### Change lap-by-lap simulation parameters (NEW)

The race predictor now uses lap-by-lap simulation with tire degradation and pit stops. Edit:

**Tire physics:**
- `baseline_predictor.race.tire_physics.fresh_tire_advantage` - Initial pace advantage per compound (SOFT/MEDIUM/HARD)
- `baseline_predictor.race.tire_physics.fresh_tire_duration` - Laps fresh tire advantage lasts
- `baseline_predictor.race.tire_physics.default_deg_slope` - Fallback degradation if no compound data
- `baseline_predictor.race.tire_physics.clean_air_bonus` - P1-5 tire life advantage (default 0.05 = 5%)
- `baseline_predictor.race.tire_physics.traffic_deg_penalty` - P16+ tire life penalty (default 0.05 = 5%)

**Fuel effects:**
- `baseline_predictor.race.fuel.initial_load_kg` - Starting fuel load (default 110kg for 60-lap race)
- `baseline_predictor.race.fuel.burn_rate_kg_per_lap` - Fuel consumed per lap (default 1.5kg)
- `baseline_predictor.race.fuel.effect_per_lap` - Lap time penalty per 10kg fuel (default 0.035s)
- `baseline_predictor.race.fuel.deg_multiplier` - How fuel load affects tire degradation (default 0.10 = 10%)

**Pit stop strategy:**
- `baseline_predictor.race.tire_strategy.windows.one_stop` - Lap window for 1-stop (default [23, 37])
- `baseline_predictor.race.tire_strategy.windows.two_stop_first` - First stop window for 2-stop (default [15, 25])
- `baseline_predictor.race.tire_strategy.windows.two_stop_second` - Second stop window for 2-stop (default [35, 45])
- `baseline_predictor.race.tire_strategy.stop_probability` - Stress-based stop count probabilities
- `baseline_predictor.race.pit_stops.loss_duration` - Base pit stop time loss (track-specific override available)
- `baseline_predictor.race.pit_stops.overtake_loss_range` - Extra time loss if overtaken during stop

**Strategy constraints:**
- `baseline_predictor.race.strategy_constraints.min_pit_lap` - Earliest allowed pit lap (default 5)
- `baseline_predictor.race.strategy_constraints.max_pit_lap_from_end` - Latest allowed pit lap from end (default 5)
- `baseline_predictor.race.strategy_constraints.min_laps_between_stops` - Minimum stint length (default 8)
- `baseline_predictor.race.strategy_constraints.pit_lap_variance` - Randomness in pit timing (one_stop: 3.0, two_stop: 2.0)
- `baseline_predictor.race.strategy_constraints.strategy_optimality` - % of optimal strategies (default 0.60 = 60%)

**Lap time modeling:**
- `baseline_predictor.race.lap_time.reference_base` - Reference lap time in seconds (default 90.0)
- `baseline_predictor.race.lap_time.team_pace_penalty_range` - Max penalty for slowest team (default 5.0s)
- `baseline_predictor.race.lap_time.skill_improvement_max` - Max driver skill advantage (config currently 0.35; code fallback 0.5 if key missing)
- `baseline_predictor.race.lap_time.bounds` - Min/max lap time clipping (default [70.0, 120.0])

### Change persistence mode

Examples:

```bash
# conservative local mode (default)
export USE_DB_STORAGE=file_only

# migration mode: keep local files + write Supabase
export USE_DB_STORAGE=dual_write
export SUPABASE_URL=https://<project>.supabase.co
export SUPABASE_KEY=<key>
```

### Change FastF1/cache paths

Edit:

- `paths.*` in `config/default.yaml`
- env vars where supported by modules:
  - `F1_CONFIG` (alternate config file path),
  - `F1_DATA_DIR` (baseline predictor data root),
  - `F1_CACHE_DIR` (auto-updater FastF1 cache path)

### Run live FastF1 integration checks

```bash
export FASTF1_LIVE_TESTS=1
pytest tests/test_fastf1_live_refresh.py -m live_fastf1
```

## Validation Rules

`src/utils/config_loader.py` validates:

- required sections exist,
- critical values are present and in range,
- qualifying weights sum to ~1.0.

If validation fails, startup will raise an explicit error.

## Example: Read Config in Code

```python
from src.utils import config_loader

team_weight = config_loader.get("baseline_predictor.qualifying.team_weight", 0.7)
pace_weight = config_loader.get("baseline_predictor.race.pace_weight_base", 0.40)
```

## Safe Workflow For Config Changes

1. Edit `config/default.yaml`.
2. If persistence mode is DB-backed, verify connectivity first:
   - `python scripts/test_supabase_connection.py`
3. Run targeted tests:
   - `pytest tests/test_baseline_2026_integration.py`
   - `pytest tests/test_dashboard_smoke.py`
4. Run a dry prediction in the app/CLI and confirm behavior.

## Notes

- Qualifying FP blending is configuration-driven and confidence-adjusted (`baseline_predictor.qualifying.fp_blend_weight*` and `data_confidence.*`), not fixed 70/30.
- `src/predictors/qualifying.py` and `src/predictors/race.py` preserve legacy method signatures and delegate to baseline logic.
- For Supabase rollouts, start with `dual_write`, then move to `fallback` or `db_only` after validating migrations and runtime-state writes.

# Documentation Index

Start here if you want the docs that still match the current runtime.

## Start Here

- `../README.md`: project-level quick start and runtime summary
- `../ARCHITECTURE.md`: component map and data flow
- `../CONFIGURATION.md`: active vs secondary config paths

## Detailed Guides

### `WEIGHT_SCHEDULE_GUIDE.md`
Baseline/testing/current signal blending and race-by-race weight progression.

### `FP_BLENDING_SYSTEM.md`
Session blending for qualifying with priority rules for normal and sprint weekends.

### `WEEKEND_PREDICTIONS.md`
Normal vs sprint cascade output, ACTUAL vs PREDICTED grids, and session chaining.

### `DASHBOARD_AUTO_UPDATE.md`
Automatic vs manual updates during dashboard use and cache behavior.

### `PREDICTION_TRACKING.md`
Session-based prediction storage, attaching actual results, and accuracy metrics.

### `COMPOUND_ANALYSIS.md`
Tire compound performance collection, dynamic selection, and race prediction adjustments.

### `PERSISTENCE_SUPABASE.md`
ArtifactStore modes, Supabase migration workflow, and active artifact keys.

### `WARMUP_PRECOMPUTE.md`
Background warmup worker for checkpoint-aware precompute and ready-race horizon indexing.

## Validation Notebooks

- `../notebooks/model_development/validate_testing_predictions.ipynb`
- `../notebooks/model_development/test_weight_schedules.ipynb`

Use these as supporting analysis, not as a substitute for checking runtime code.

## Scope Note

If docs and code disagree, trust the code first. This set is meant to mirror the
current runtime path, not preserve old design notes.

# Prediction Tracking

This guide describes how predictions are stored by session and how actual results are attached later for accuracy analysis.

## Where It Lives

- Logger: `src/utils/prediction_logger.py`
- Session detection: `src/utils/session_detector.py`
- Metrics: `src/utils/prediction_metrics.py`
- Update script: `scripts/update_prediction_actuals.py`
- Dashboard usage: `src/dashboard/pages.py` (Prediction toggle + Accuracy page; entrypoint: `app.py`)

## How Saving Works

1. Enable **Save Predictions for Accuracy Tracking** in the **Settings** expander on the main page.
2. Click **Predict**.
3. App detects the latest completed session for the weekend type.
4. If no prediction exists yet for that race/session key, it writes one prediction artifact through `ArtifactStore`.

In dashboard flow, this is still max one saved prediction per race/session key.

## Storage Backend Behavior

Persistence mode is controlled by `USE_DB_STORAGE`:

- `file_only`: writes JSON files only.
- `db_only`: writes Supabase only.
- `fallback`: reads DB first and falls back to file; writes Supabase only.
- `dual_write`: writes both Supabase and files.

File root (when files are written):

- `data/predictions/<year>/<race_slug>/`

Example file names:

- `bahrain_grand_prix_fp1.json`
- `bahrain_grand_prix_fp2.json`
- `bahrain_grand_prix_fp3.json`
- `chinese_grand_prix_sq.json`
- `chinese_grand_prix_sprint.json`

## Session Detection Rules

Normal weekend tracked sessions:

- `FP1`, `FP2`, `FP3`

Sprint weekend tracked sessions:

- `FP1`, `SQ`, `Sprint`

The detector uses scheduled session time plus buffer duration to decide whether a session is completed.

## Saved Payload (Current Shape)

Each file contains:

- `metadata` (year, race, session, timestamp, weather, optional blend info, run_id)
- `qualifying.predicted_grid`
- `race.predicted_results`
- `actuals` placeholder (`qualifying`, `race`)

## Add Actual Results Later

```bash
python scripts/update_prediction_actuals.py "Bahrain Grand Prix" FP1 --year 2026
```

This script fetches `Q` and `R` results from FastF1 and writes them into the matching prediction file.

## Learning Update Side Effect

When actuals are attached (dashboard or script path calling `PredictionLogger.update_actuals()`):

- adaptive calibration is updated through `src/systems/systematic_learning.py`,
- per-driver and teammate-gap EMA errors are refreshed for `qualifying` and `race`,
- state is persisted to `data/learning_state.json`.

These learned adjustments are consumed by qualifying/race scoring in the baseline predictor.

## Accuracy View

In the dashboard **Prediction Accuracy** page, metrics are computed only for predictions that already include actual results.

Typical metrics shown:

- exact accuracy,
- MAE,
- within-band accuracy,
- correlation,
- winner/podium checks.

## Known Limits

1. Session labels must match saved keys (`FP1`, `FP2`, `FP3`, `SQ`, `Sprint`).
2. Actuals update depends on FastF1 availability for qualifying and race sessions.
3. If no session has completed yet, nothing is saved.
4. Dashboard accuracy history currently scans local files, so pure `db_only`/`fallback` mode may not show saved history unless files also exist.
5. Learning updates require matching driver identifiers between predicted payloads and actual results.

# Prediction Tracking

This is the storage path for checkpoint predictions, later actuals, and derived accuracy snapshots.

## Where It Lives

- Logger: `src/utils/prediction_logger.py`
- Session detection: `src/utils/session_detector.py`
- Target mapping: `src/utils/accuracy_targets.py`
- Metrics: `src/utils/prediction_metrics.py`
- Snapshot helpers: `src/utils/accuracy_snapshots.py`
- Update script: `scripts/update_prediction_actuals.py`
- Snapshot backfill: `scripts/backfill_accuracy_snapshots.py`
- Datapoint compare/sync: `scripts/sync_dashboard_datapoints_to_db.py`
- Dashboard usage: `src/dashboard/pages.py` (Prediction page + Accuracy page; entrypoint: `app.py`)

## How Saving Works

1. Click Predict.
2. App detects the latest completed session for the weekend type.
3. If no completed session exists yet, the checkpoint is saved as `PRE`.
4. The save path extracts every tracked target that is still a real forecast at that checkpoint.
5. If no prediction exists yet for that race/session key, it writes one prediction artifact through `ArtifactStore`.

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

## Checkpoints And Targets

Stored checkpoints:

- `PRE`
- `FP1`, `FP2`, `FP3`, `Q` on normal weekends
- `FP1`, `SQ`, `Sprint`, `Q` on sprint weekends

Canonical tracked targets:

- `main_qualifying`
- `grand_prix_race`
- `sprint_qualifying`
- `sprint_race`

Eligible checkpoints by target:

- Normal `main_qualifying`: `PRE`, `FP1`, `FP2`, `FP3`
- Normal `grand_prix_race`: `PRE`, `FP1`, `FP2`, `FP3`, `Q`
- Sprint `sprint_qualifying`: `PRE`, `FP1`
- Sprint `sprint_race`: `PRE`, `FP1`, `SQ`
- Sprint `main_qualifying`: `PRE`, `FP1`, `SQ`, `Sprint`
- Sprint `grand_prix_race`: `PRE`, `FP1`, `SQ`, `Sprint`, `Q`

The detector uses scheduled session time plus buffer duration to decide whether a session is completed.
`PRE` is the synthetic pre-weekend checkpoint used when no sessions have finished yet.

## Saved Payload (Current Shape)

Each file contains:

- `metadata` (year, race, session, timestamp, weather, optional blend info, run_id, `weekend_format`)
- `qualifying.predicted_grid`
- `race.predicted_results`
- `targets.<target_key>`
- `actuals.qualifying`
- `actuals.race`
- `actuals.targets.<target_key>`

Each target payload stores:

- `target_session`
- `predicted_order`
- `result_mode`
- `grid_source`
- `fp_blend_info`
- `mean_confidence`
- `eligible_at_save`

The top-level `qualifying` and `race` fields remain for backward compatibility. New accuracy code reads `targets` first and falls back to the legacy shape only when needed.

## Add Actual Results Later

```bash
python scripts/update_prediction_actuals.py "Bahrain Grand Prix" FP1 --year 2026
```

This script now fetches actual results per stored target session and writes them into the matching prediction artifact.

The accuracy page renders from stored predictions and snapshots first. Scheduled workers own freshness: session automation reconciles actuals after race completion, and warmup also runs the same reconciliation after precompute when `dashboard.prediction_precompute.reconcile_accuracy_after_warmup` is enabled. The Refresh Actuals button remains a repair control, not the normal update path.

To backfill missing snapshot artifacts from already stored prediction truth:

```bash
python scripts/backfill_accuracy_snapshots.py --year 2026 --dry-run
python scripts/backfill_accuracy_snapshots.py --year 2026
```

Backfill only writes snapshots for targets that already have stored actuals. It does not invent missing historic sprint targets.

To compare the local dashboard datapoints for one race against Supabase, and optionally
sync only those checkpoint artifacts:

```bash
uv run python scripts/sync_dashboard_datapoints_to_db.py \
  --env-file .env.local \
  --year 2026 \
  --race-name "Australian Grand Prix" \
  --checkpoint FP1 \
  --checkpoint FP2 \
  --checkpoint FP3
```

Add `--sync` to push the mismatched local rows to Supabase. Add
`--include-auxiliary-targets` on sprint weekends if you also want `sprint_qualifying`
and `sprint_race` snapshot rows.

If Supabase still has warmup rows from older artifact hashes, clean those with:

```bash
uv run python scripts/prune_stale_precompute_state.py \
  --env-file .env.local \
  --year 2026 \
  --require-db
```

Add `--apply` to delete the stale rows. This only prunes dashboard warmup cache
state in `runtime_state`; it does not remove saved checkpoint predictions or
accuracy snapshots.

## Learning Update Side Effect

When actuals are attached (dashboard or script path calling `PredictionLogger.update_actuals()`):

- adaptive calibration is updated through `src/systems/systematic_learning.py`,
- per-driver and teammate-gap EMA errors are refreshed for `qualifying` and `race`,
- state is persisted to `data/learning_state.json`.

These learned adjustments are consumed by qualifying/race scoring in the baseline predictor.

Learning is intentionally conservative. The update path skips:

- retrospective checkpoint reconstructions,
- duplicate `run_id` values,
- records with no actual results,
- records where too few predicted drivers overlap with actual results.

Skipped records do not mark the run ID as processed, so a later complete actual
update can still train the learner. Interval residual history uses the same
valid-session gate.

## Accuracy View

In the dashboard Prediction Accuracy page, metrics are computed per target.

Primary metrics:

- `overall_mae`
- `top_3_hits`
- `top_3_pct`
- `top_10_hits`
- `top_10_pct`

Supporting metrics kept in the payload and UI:

- `exact_accuracy`
- `within_1`
- `within_3`
- `correlation`
- `field_size`

The dashboard shows:

- KPI cards for main qualifying and Grand Prix race
- Weekend progression charts, split into normal and sprint weekends
- Season trend charts, split into normal and sprint weekends
- Optional drilldowns for `sprint_qualifying` and `sprint_race`
- A saved-predictions list with target-aware completion status

## Known Limits

1. Session labels must match saved keys exactly.
2. Actuals updates still depend on FastF1 availability for the needed target session.
3. Historic sprint weekends can have genuine gaps for early main `Q/R` targets if those forecasts were never stored.
4. Learning updates still depend on matching driver identifiers between predicted payloads and actual results.
5. Very small partial actual payloads are ignored until enough driver overlap exists.

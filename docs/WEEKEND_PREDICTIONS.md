# Weekend Prediction Flow

This guide describes the current dashboard behavior in `src/dashboard/pages.py` (entrypoint: `app.py`).

## Overview

The app produces a cascade of predictions based on weekend format.

- Normal weekend: 2 outputs
- Sprint weekend: 4 outputs

The race model can use ACTUAL grids from completed competitive sessions when available.

## Normal Weekend

Flow:

1. Qualifying prediction
2. Race prediction

Process:

- Predict qualifying grid.
- If qualifying session is already complete and results are available, replace predicted grid with ACTUAL qualifying results.
- Predict race from that grid.

## Sprint Weekend

Flow:

1. Sprint Qualifying prediction
2. Sprint Race prediction
3. Main Qualifying prediction
4. Main Race prediction

Process:

- Predict Sprint Qualifying.
- Sprint Race uses Sprint Qualifying grid (ACTUAL if available, otherwise predicted).
- Predict Main Qualifying.
- Main Race uses Main Qualifying grid (ACTUAL if available, otherwise predicted).

## ACTUAL vs PREDICTED Grid Source

Grid source is resolved in `fetch_grid_if_available()` in `src/dashboard/prediction_flow.py`.

- `ACTUAL`: completed competitive session results were fetched.
- `PREDICTED`: session is not yet complete, so model output is used.

If completion state is `unknown`, the flow fails closed for that session instead of silently falling back.

Competitive sessions checked for grid replacement:

- `SQ` (Sprint Qualifying)
- `Q` (Main Qualifying)

## Practice Data Use In This Flow

Practice/session blending is used in qualifying prediction through `Baseline2026Predictor.predict_qualifying()`.

Important details:

- The predictor builds a **short-stint weighted blend** from available sessions.
- If weekend practice pace is unavailable, it can use a **testing short-run profile fallback**.
- If both are unavailable, qualifying runs in **model-only** mode.

Session blend inputs from `src/utils/fp_blending.py`:

- Normal weekend: `FP3 + FP2 + FP1` (FP3-weighted)
- Sprint weekend (main qualifying): `Sprint Qualifying + FP1 + Sprint`

When no weekend practice data exists:

- `predict_qualifying()` sets `data_source` to `Testing short-run profile blend (no weekend practice data)` if fallback is available.
- Otherwise `data_source` is `Model-only (no practice/testing data)`.

In model-only mode, the qualifying stack also applies teammate/experience stabilization and learned calibration nudges.

## Sprint Race Adjustments

Sprint race prediction calls `predict_race(..., is_sprint=True)`.

Current sprint adjustments in baseline predictor:

- lower chaos level,
- extra grid weight influence.
- race still uses track-aware overtaking and strategy timing bias inputs.

## Prediction Tracking Integration

When tracking is enabled in the **Settings** expander:

- one prediction artifact is saved per detected completed session,
- dashboard flow enforces max one save per race/session key,
- sprint weekends save the main qualifying + main race outputs for scoring.

Storage backend depends on `USE_DB_STORAGE` (`file_only`, `db_only`, `fallback`, `dual_write`).

See `docs/PREDICTION_TRACKING.md` for file structure and update workflow.

## Known Limits

1. ACTUAL grid replacement depends on FastF1 data availability.
2. If completion status is unknown, prediction generation for that session is blocked until status can be resolved.
3. Weekend format detection depends on FastF1 event schedule (with local fallback in utilities).

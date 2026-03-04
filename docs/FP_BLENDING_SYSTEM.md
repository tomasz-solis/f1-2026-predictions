# Practice Session Blending

This page explains the qualifying blend implemented in `src/utils/fp_blending.py` and consumed by `Baseline2026Predictor.predict_qualifying()`.

## What It Does

Qualifying prediction combines:

- model team strength (from weight schedule), and
- team pace extracted from short-stint session signals.

Active formula:

```text
blended_strength = w * session_strength + (1 - w) * model_strength
```

The blend weight `w` is data-confidence aware in the active baseline path:

```text
w = clip(base_weight + (confidence - 0.5) * scale, min_weight, max_weight)
```

Default values in `config/default.yaml` are `base=0.70`, `scale=0.30`, `min=0.45`, and `max=0.85`.

## Session Inputs

### Normal weekend

`FP3 + FP2 + FP1` (weighted blend, FP3-heavy)

### Sprint weekend

`Sprint Qualifying + FP1 + Sprint` (weighted blend for main qualifying)

## How Session Strength Is Built

For each available session:

1. Load laps.
2. Build representative short-stint pace per driver (push-lap focused, TireLife-aware when available).
3. Compute median representative lap per team.
4. Scale teams to a 0-1 performance band (fastest = 1.0).
5. Combine sessions with fixed weights into one blended session-strength map.

## Fallbacks

- If no session data is available, qualifying uses model-only strength.
- If a team is missing from session data, that team keeps model-only strength.

## Where It Is Used

- `src/predictors/baseline_2026.py` (`predict_qualifying`)
- Dashboard prediction flow in `src/dashboard/prediction_flow.py` (called from `src/dashboard/pages.py`)

## Where It Is Not Used Directly

- Race scoring does not apply the FP blending function directly.
- Race model runs from grid + race simulation features.

## Practical Notes

- Blend source is shown in UI (`data_source` and `blend_used`).
- Session naming follows FastF1 conventions.
- Behavior depends on data availability and cache state.

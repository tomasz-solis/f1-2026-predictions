# Team-Strength Refit Candidate Test

- Built at: `2026-05-20T13:16:22.620176+00:00`
- Model version: `2.1`
- Status: `measured`
- Policy: `same_session_construct`

## Aggregate Held-Out Metrics

| Candidate | Folds | Rows | Weighted MSE | Weighted RMSE | MSE delta vs current |
| --- | ---: | ---: | ---: | ---: | ---: |
| `loo_2026_linear_refit` | 8 | 126 | 0.364 | 0.603 | -46.7% |
| `loo_2026_scale_only` | 8 | 126 | 0.366 | 0.605 | -46.4% |
| `current_frozen_mapping` | 8 | 126 | 0.683 | 0.826 | +0.0% |
| `uncertainty_only_current_medians` | 8 | 126 | 0.683 | 0.826 | +0.0% |

## By Construct

### Race

| Candidate | Folds | Rows | Weighted MSE | Weighted RMSE | MSE delta vs current |
| --- | ---: | ---: | ---: | ---: | ---: |
| `loo_2026_linear_refit` | 4 | 61 | 0.397 | 0.630 | -62.4% |
| `loo_2026_scale_only` | 4 | 61 | 0.416 | 0.645 | -60.6% |
| `current_frozen_mapping` | 4 | 61 | 1.058 | 1.029 | +0.0% |
| `uncertainty_only_current_medians` | 4 | 61 | 1.058 | 1.029 | +0.0% |

### Qualifying

| Candidate | Folds | Rows | Weighted MSE | Weighted RMSE | MSE delta vs current |
| --- | ---: | ---: | ---: | ---: | ---: |
| `loo_2026_scale_only` | 4 | 65 | 0.318 | 0.564 | -3.9% |
| `current_frozen_mapping` | 4 | 65 | 0.331 | 0.575 | +0.0% |
| `uncertainty_only_current_medians` | 4 | 65 | 0.331 | 0.575 | +0.0% |
| `loo_2026_linear_refit` | 4 | 65 | 0.333 | 0.577 | +0.4% |

## Winner Summary

- `combined`:
  - loo_2026_linear_refit: 6 wins, 2 losses, 0 ties versus current
  - loo_2026_scale_only: 6 wins, 2 losses, 0 ties versus current
  - uncertainty_only_current_medians: 0 wins, 0 losses, 8 ties versus current
- `race`:
  - loo_2026_linear_refit: 4 wins, 0 losses, 0 ties versus current
  - loo_2026_scale_only: 4 wins, 0 losses, 0 ties versus current
  - uncertainty_only_current_medians: 0 wins, 0 losses, 4 ties versus current
- `qualifying`:
  - loo_2026_linear_refit: 2 wins, 2 losses, 0 ties versus current
  - loo_2026_scale_only: 2 wins, 2 losses, 0 ties versus current
  - uncertainty_only_current_medians: 0 wins, 0 losses, 4 ties versus current

## Decision Assessment

- State: `refit_candidate_worth_full_prediction_replay`
- Recommendation: Run a full prediction replay before shipping. Construct MSE supports more testing, not an automatic production refit.
- loo_2026_linear_refit improves row-weighted held-out MSE by 46.7% versus the frozen mapping.
- race best row-weighted candidate: loo_2026_linear_refit (-62.4% vs current).
- race uncertainty-only keeps identical medians, so MSE is unchanged.
- qualifying best row-weighted candidate: loo_2026_scale_only (-3.9% vs current).
- qualifying uncertainty-only keeps identical medians, so MSE is unchanged.

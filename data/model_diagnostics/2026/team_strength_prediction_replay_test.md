# Team-Strength Prediction Replay Test

- Built at: `2026-05-20T16:00:18.439535+00:00`
- Model version: `2.1`
- Status: `measured`
- Candidate: `loo_2026_scale_only`

## Aggregate

| Scope | Rows | Current MSE | Candidate MSE | MSE delta | Current MAE | Candidate MAE | MSE wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `combined` | 34 | 28.767 | 28.176 | -2.1% | 3.979 | 3.936 | 18 |

## Race Targets Only

| Scope | Rows | Current MSE | Candidate MSE | MSE delta | Current MAE | Candidate MAE | MSE wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `combined` | 20 | 33.277 | 32.427 | -2.6% | 4.382 | 4.314 | 12 |

## By Target

| Group | Rows | Current MSE | Candidate MSE | MSE delta | Current MAE | Candidate MAE | MSE wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `grand_prix_race` | 14 | 39.558 | 38.357 | -3.0% | 4.799 | 4.688 | 10 |
| `main_qualifying` | 14 | 22.325 | 22.104 | -1.0% | 3.403 | 3.396 | 6 |
| `sprint_race` | 6 | 18.621 | 18.591 | -0.2% | 3.409 | 3.439 | 2 |

## Decision Assessment

- State: `not_enough_evidence`
- Recommendation: Do not ship a median-changing race refit from this replay.
- Race-target position MSE delta: -2.6%.
- Race-target MSE wins: 12/20.
- All-target position MSE delta: -2.1%.

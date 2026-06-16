# Shadow Challenger Audit - 2026

- Challenger version: `target_form_blend_v1`
- Prediction artifacts loaded: **28**

## Target Scores

| Target | Events | Challenger events | Champion MAE | Challenger MAE | Improvement |
|---|---:|---:|---:|---:|---:|
| Grand Prix Race | 7 | 6 | 5.121 | 5.076 | 0.045 |
| Main Qualifying | 7 | 6 | 3.106 | 2.500 | 0.606 |
| Sprint Qualifying | 3 | 2 | 3.409 | 3.364 | 0.045 |
| Sprint Race | 3 | 2 | 3.000 | 2.909 | 0.091 |

## Checkpoint Decay

### Grand Prix Race

| Checkpoint | Events | Mean MAE | Median MAE |
|---|---:|---:|---:|
| `PRE` | 7 | 4.662 | 4.909 |
| `FP1` | 6 | 5.061 | 5.045 |
| `FP2` | 4 | 4.909 | 4.636 |
| `FP3` | 3 | 5.273 | 4.727 |
| `SQ` | 3 | 5.333 | 5.182 |

### Main Qualifying

| Checkpoint | Events | Mean MAE | Median MAE |
|---|---:|---:|---:|
| `PRE` | 7 | 2.870 | 2.909 |
| `FP1` | 6 | 3.470 | 3.455 |
| `FP2` | 4 | 3.909 | 3.955 |
| `FP3` | 3 | 3.606 | 3.636 |
| `SQ` | 3 | 2.788 | 2.455 |

### Sprint Qualifying

| Checkpoint | Events | Mean MAE | Median MAE |
|---|---:|---:|---:|
| `PRE` | 3 | 2.788 | 3.000 |
| `FP1` | 3 | 3.121 | 3.273 |

### Sprint Race

| Checkpoint | Events | Mean MAE | Median MAE |
|---|---:|---:|---:|
| `PRE` | 3 | 3.515 | 3.636 |
| `FP1` | 3 | 3.818 | 3.909 |
| `SQ` | 3 | 3.061 | 3.182 |

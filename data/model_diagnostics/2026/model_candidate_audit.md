# Model Candidate Audit - 2026

Candidates are evaluated in expanding-window order. The first scored
event seeds history; each later event can only use previous completed
actuals and prior residuals.

- Prediction artifacts loaded: **28**
- Schedule rows: **23**

## Qualifying

- Selected events: **7**
- Scored candidate events: **6**

### Promotion Readout

- Recommendation: **HOLD**
- Reference: `raw_model`
- Best challenger: `rolling_actual_2` (recent_actual_only)
- Improvement vs reference: **0.652 MAE**
- Blocking reasons: Only 6 scored challenger events; require 8 before promotion. | Best candidate is recent-actual-only; keep it as a challenger until a model-aware blend or residual layer matches it without dropping feature explainability.

### Saved Signal Coverage

- Model-diagnostic metadata coverage: **0.0%** (0/7)
- Characteristics-profile evidence: **0** events
- Residual-model flag evidence: **0** events

| Candidate | Events | Mean MAE | Median MAE | vs raw | vs prev-race naive |
|---|---:|---:|---:|---:|---:|
| `rolling_actual_2` | 6 | 2.621 | 2.591 | 0.652 | 0.364 |
| `rolling_actual_3` | 6 | 2.667 | 2.682 | 0.606 | 0.318 |
| `fixed_blend_model_0.4` | 6 | 2.864 | 2.864 | 0.409 | 0.121 |
| `fixed_blend_model_0.2` | 6 | 2.894 | 2.909 | 0.379 | 0.091 |
| `fixed_blend_model_0.6` | 6 | 2.924 | 2.682 | 0.348 | 0.061 |
| `previous_race_naive` | 6 | 2.985 | 2.955 | 0.288 | 0.000 |
| `fixed_blend_model_0.8` | 6 | 3.106 | 3.182 | 0.167 | -0.121 |
| `team_bias_alpha_0.25` | 6 | 3.212 | 3.273 | 0.061 | -0.227 |
| `raw_model` | 6 | 3.273 | 3.409 | 0.000 | -0.288 |
| `driver_bias_alpha_0.25` | 6 | 3.288 | 3.273 | -0.015 | -0.303 |
| `driver_bias_alpha_0.50` | 6 | 3.318 | 3.273 | -0.045 | -0.333 |
| `team_bias_alpha_0.50` | 6 | 3.348 | 3.318 | -0.076 | -0.364 |

### conventional

| Candidate | Events | Mean MAE |
|---|---:|---:|
| `rolling_actual_2` | 3 | 2.455 |
| `rolling_actual_3` | 3 | 2.485 |
| `fixed_blend_model_0.4` | 3 | 2.848 |
| `fixed_blend_model_0.2` | 3 | 2.879 |
| `previous_race_naive` | 3 | 3.000 |
| `fixed_blend_model_0.6` | 3 | 3.061 |
| `fixed_blend_model_0.8` | 3 | 3.242 |
| `team_bias_alpha_0.25` | 3 | 3.424 |

### sprint_qualifying

| Candidate | Events | Mean MAE |
|---|---:|---:|
| `fixed_blend_model_0.6` | 3 | 2.788 |
| `rolling_actual_2` | 3 | 2.788 |
| `rolling_actual_3` | 3 | 2.848 |
| `fixed_blend_model_0.4` | 3 | 2.879 |
| `fixed_blend_model_0.2` | 3 | 2.909 |
| `fixed_blend_model_0.8` | 3 | 2.970 |
| `previous_race_naive` | 3 | 2.970 |
| `team_bias_alpha_0.25` | 3 | 3.000 |

## Race

- Selected events: **7**
- Scored candidate events: **6**

### Promotion Readout

- Recommendation: **HOLD**
- Reference: `raw_model`
- Best challenger: `fixed_blend_model_0.8` (model_recent_actual_blend)
- Improvement vs reference: **0.136 MAE**
- Blocking reasons: Only 6 scored challenger events; require 8 before promotion. | Best candidate improvement vs reference is 0.136; require at least 0.150 MAE.

### Saved Signal Coverage

- Model-diagnostic metadata coverage: **0.0%** (0/7)
- Characteristics-profile evidence: **0** events
- Residual-model flag evidence: **0** events

| Candidate | Events | Mean MAE | Median MAE | vs raw | vs prev-race naive |
|---|---:|---:|---:|---:|---:|
| `fixed_blend_model_0.8` | 6 | 3.848 | 3.045 | 0.136 | 0.652 |
| `rolling_actual_3` | 6 | 3.894 | 3.409 | 0.091 | 0.606 |
| `raw_model` | 6 | 3.985 | 3.182 | 0.000 | 0.515 |
| `team_bias_alpha_0.50` | 6 | 4.000 | 3.409 | -0.015 | 0.500 |
| `fixed_blend_model_0.6` | 6 | 4.015 | 3.318 | -0.030 | 0.485 |
| `team_bias_alpha_0.25` | 6 | 4.015 | 3.273 | -0.030 | 0.485 |
| `driver_bias_alpha_0.25` | 6 | 4.030 | 3.318 | -0.045 | 0.470 |
| `team_bias_alpha_0.75` | 6 | 4.030 | 3.545 | -0.045 | 0.470 |
| `driver_bias_alpha_0.50` | 6 | 4.091 | 3.409 | -0.106 | 0.409 |
| `driver_bias_alpha_0.75` | 6 | 4.106 | 3.727 | -0.121 | 0.394 |
| `fixed_blend_model_0.4` | 6 | 4.121 | 3.455 | -0.136 | 0.379 |
| `rolling_actual_2` | 6 | 4.121 | 3.545 | -0.136 | 0.379 |

### conventional

| Candidate | Events | Mean MAE |
|---|---:|---:|
| `rolling_actual_3` | 3 | 4.364 |
| `fixed_blend_model_0.8` | 3 | 4.727 |
| `driver_bias_alpha_0.75` | 3 | 4.909 |
| `raw_model` | 3 | 4.909 |
| `rolling_actual_2` | 3 | 4.909 |
| `fixed_blend_model_0.6` | 3 | 4.939 |
| `driver_bias_alpha_0.25` | 3 | 4.939 |
| `fixed_blend_model_0.4` | 3 | 4.970 |

### sprint_qualifying

| Candidate | Events | Mean MAE |
|---|---:|---:|
| `team_bias_alpha_0.50` | 3 | 2.939 |
| `fixed_blend_model_0.8` | 3 | 2.970 |
| `team_bias_alpha_0.75` | 3 | 3.030 |
| `raw_model` | 3 | 3.061 |
| `team_bias_alpha_0.25` | 3 | 3.061 |
| `driver_bias_alpha_0.50` | 3 | 3.091 |
| `fixed_blend_model_0.6` | 3 | 3.091 |
| `driver_bias_alpha_0.25` | 3 | 3.121 |

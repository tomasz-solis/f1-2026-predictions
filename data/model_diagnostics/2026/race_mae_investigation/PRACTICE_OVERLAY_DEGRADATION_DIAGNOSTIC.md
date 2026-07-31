# Practice-overlay PRE->FP degradation diagnostic (read-only)

Generated: 2026-07-20T09:06:27.706730Z

## Finding and recommendation (read this first)

The served forecast's race-pace prediction likely DOES degrade through a race weekend on matched-subset evidence: conditional_actual_grid finisher_mae worsens PRE->FP3 by ~0.57 (main-dry, n=4) against a seed-noise floor of ~0.02-0.15 -- the gap is 4-28x seed noise, not a small-n artifact. Qualifying-grid quality itself is flat (PRE 3.879 -> FP3 3.871 grid_mae) -- the damage is entirely in the race-pace/car-characteristics overlay, not grid propagation. Root mechanism: the car-characteristics EWMA (update_from_testing_sessions, new_weight=0.25) has no evidence-robustness gate and compounds every session applied within a checkpoint's cumulative session list, reaching ~58% pull toward this weekend's own (thin, single-session) practice data by FP3 -- more trust than the data quality justifies, with no floor comparable to r0's or Q1's real gates. Sprint-dry evidence (n=2) is too thin to confirm or rule out the same pattern; it should not be treated as either confirming or contradicting the main-dry result.

**Recommendation (NOT implemented this round):** Gate the car-characteristics EWMA on evidence robustness the same way r0's race-practice-evidence already is (a minimum clean-lap/stint threshold per session before update_from_testing_sessions trusts it at new_weight=0.25; thinner sessions get a proportionally smaller weight, mirroring MIN_R0_TEAM_COVERAGE's existing pattern) -- OR cap the cumulative practice pull (e.g. via directionality_scale / a session-count-aware ceiling) so FP3 cannot exceed FP1's per-session trust by compounding. Do NOT implement either this round -- this is diagnosis only, for approval.

## Mechanism trace (real config numbers, read live from config/default.yaml)

### Layer 1 -- qualifying-side FP blend (ruled out)

qualifying_mixin.py's own team-skill-from-FP-performance blend (_resolve_fp_blend_weight -> _adjust_stored_checkpoint_blend_weight). Confidence-scaled by checkpoint (FP1<FP2<FP3) but then hard-capped.
```json
{
  "description": "qualifying_mixin.py's own team-skill-from-FP-performance blend (_resolve_fp_blend_weight -> _adjust_stored_checkpoint_blend_weight). Confidence-scaled by checkpoint (FP1<FP2<FP3) but then hard-capped.",
  "stored_checkpoint_blend_weight_cap": 0.25,
  "stored_checkpoint_blend_weight_multiplier": 1.0,
  "observed_fp_blend_weight_used": {
    "FP1": 0.25,
    "FP2": 0.25,
    "FP3": 0.25
  },
  "verdict": "FLAT across FP1/FP2/FP3 (cap dominates) -- NOT the degradation driver."
}
```

### Layer 2 -- car-characteristics EWMA (dominant contributor)

src/systems/testing_updater.py update_from_testing_sessions, invoked once per session via src/utils/historical_replay.py:_apply_session_update, called from ProductionReplayBackend._checkpoint_state_for. Feeds team strength / tire degradation used by BOTH qualifying and race simulation.
```json
{
  "description": "src/systems/testing_updater.py update_from_testing_sessions, invoked once per session via src/utils/historical_replay.py:_apply_session_update, called from ProductionReplayBackend._checkpoint_state_for. Feeds team strength / tire degradation used by BOTH qualifying and race simulation.",
  "new_weight_config_value": 0.25,
  "new_weight_matches_live_update_flow_default": true,
  "note": "src/dashboard/update_flow.py (live automation) reads the SAME baseline_predictor.practice_capture.new_weight config key -- this is not a replay-only artifact.",
  "sessions_available_is_cumulative_per_checkpoint": true,
  "evidence_robustness_gate": null,
  "gate_comment": "No lap-count/stint-count floor exists at this layer (unlike r0's MIN_R0_TEAM_COVERAGE gate or Q1's raw-lap requirement) -- a single thin, unrepresentative practice session is blended at the exact same weight as a robust one.",
  "cumulative_practice_weight_by_checkpoint": {
    "FP1": {
      "n_sessions_applied": 1,
      "cumulative_practice_weight": 0.25
    },
    "FP2": {
      "n_sessions_applied": 2,
      "cumulative_practice_weight": 0.4375
    },
    "FP3": {
      "n_sessions_applied": 3,
      "cumulative_practice_weight": 0.5781
    }
  },
  "verdict": "Compounds through the weekend by construction (25% -> 43.75% -> 57.8% pull toward this weekend's own single-session snapshots) -- the dominant contributor. Explains why conditional_actual_grid (grid held fixed) still degrades: this layer feeds race pace directly, independent of the qualifying-side cap."
}
```

### Prior corroborating evidence

```json
{
  "path": "data/model_diagnostics/2026/practice_signal_blend_probe.md",
  "commit": "160ddc26 feat(diagnostics): probe practice-signal trust in qualifying checkpoints",
  "finding": "A different, shallower probe (output-rank blend, not the characteristics-level EWMA) already found w=0.00 (no practice blend) beats w>=0.5 at FP2/FP3 pooled qualifying MAE, concluding stored-checkpoint blend caps should be reduced -- independent evidence pointing the same direction."
}
```

## Matched-subset numbers

## main_dry_pre_fp2_fp3 (n=4 matched events: ['2026_01_australian_grand_prix', '2026_03_japanese_grand_prix', '2026_06_monaco_grand_prix', '2026_08_austrian_grand_prix'])

### Qualifying grid_mae

| checkpoint | mean | n_events | mean_per_event_seed_std |
|---|---|---|---|
| PRE | 3.879 | 4 | 0.152 |
| FP2 | 3.795 | 4 | 0.043 |
| FP3 | 3.871 | 4 | 0.050 |

### race_views: conditional_actual_grid

| metric | checkpoint | mean | n_events | mean_per_event_seed_std |
|---|---|---|---|---|
| finisher_mae | PRE | 2.868 | 4 | 0.150 |
| finisher_mae | FP2 | 3.274 | 4 | 0.013 |
| finisher_mae | FP3 | 3.436 | 4 | 0.020 |
| weighted_mae | PRE | 3.887 | 4 | 0.218 |
| weighted_mae | FP2 | 4.538 | 4 | 0.011 |
| weighted_mae | FP3 | 4.582 | 4 | 0.017 |
| top_heavy_weighted_mae | PRE | 3.809 | 4 | 0.186 |
| top_heavy_weighted_mae | FP2 | 4.248 | 4 | 0.016 |
| top_heavy_weighted_mae | FP3 | 4.388 | 4 | 0.027 |

### race_views: end_to_end_predicted_grid

| metric | checkpoint | mean | n_events | mean_per_event_seed_std |
|---|---|---|---|---|
| finisher_mae | PRE | 4.164 | 4 | 0.158 |
| finisher_mae | FP2 | 4.536 | 4 | 0.061 |
| finisher_mae | FP3 | 4.894 | 4 | 0.101 |
| weighted_mae | PRE | 5.041 | 4 | 0.255 |
| weighted_mae | FP2 | 4.831 | 4 | 0.057 |
| weighted_mae | FP3 | 5.043 | 4 | 0.096 |
| top_heavy_weighted_mae | PRE | 4.944 | 4 | 0.159 |
| top_heavy_weighted_mae | FP2 | 5.275 | 4 | 0.061 |
| top_heavy_weighted_mae | FP3 | 5.616 | 4 | 0.120 |

### Per-event delta (worst-degrading events first)

`conditional_actual_grid`:
```json
[
  {
    "event_id": "2026_08_austrian_grand_prix",
    "PRE": 1.8148148148148149,
    "FP3": 2.574074074074074,
    "delta": 0.7592592592592591
  },
  {
    "event_id": "2026_01_australian_grand_prix",
    "PRE": 3.3958333333333335,
    "FP3": 4.145833333333333,
    "delta": 0.7499999999999996
  },
  {
    "event_id": "2026_06_monaco_grand_prix",
    "PRE": 4.3125,
    "FP3": 4.875,
    "delta": 0.5625
  },
  {
    "event_id": "2026_03_japanese_grand_prix",
    "PRE": 1.95,
    "FP3": 2.15,
    "delta": 0.19999999999999996
  }
]
```
`end_to_end_predicted_grid`:
```json
[
  {
    "event_id": "2026_01_australian_grand_prix",
    "PRE": 4.5625,
    "FP3": 5.854166666666667,
    "delta": 1.291666666666667
  },
  {
    "event_id": "2026_03_japanese_grand_prix",
    "PRE": 3.9499999999999997,
    "FP3": 4.883333333333334,
    "delta": 0.933333333333334
  },
  {
    "event_id": "2026_08_austrian_grand_prix",
    "PRE": 2.685185185185185,
    "FP3": 3.2962962962962963,
    "delta": 0.6111111111111112
  },
  {
    "event_id": "2026_06_monaco_grand_prix",
    "PRE": 5.458333333333333,
    "FP3": 5.541666666666667,
    "delta": 0.08333333333333393
  }
]
```
`qualifying`:
```json
[
  {
    "event_id": "2026_01_australian_grand_prix",
    "PRE": 5.2727272727272725,
    "FP3": 5.454545454545454,
    "delta": 0.18181818181818166
  },
  {
    "event_id": "2026_03_japanese_grand_prix",
    "PRE": 4.878787878787879,
    "FP3": 5.0606060606060606,
    "delta": 0.18181818181818166
  },
  {
    "event_id": "2026_08_austrian_grand_prix",
    "PRE": 2.757575757575758,
    "FP3": 2.7878787878787876,
    "delta": 0.030303030303029832
  },
  {
    "event_id": "2026_06_monaco_grand_prix",
    "PRE": 2.606060606060606,
    "FP3": 2.1818181818181817,
    "delta": -0.4242424242424243
  }
]
```

## sprint_dry_pre_fp1 (n=2 matched events: ['2026_02_chinese_grand_prix', '2026_09_british_grand_prix'])

### Qualifying grid_mae

| checkpoint | mean | n_events | mean_per_event_seed_std |
|---|---|---|---|
| PRE | 3.212 | 2 | 0.155 |
| FP1 | 3.091 | 2 | 0.037 |

### race_views: conditional_actual_grid

| metric | checkpoint | mean | n_events | mean_per_event_seed_std |
|---|---|---|---|---|
| finisher_mae | PRE | 3.358 | 2 | 0.027 |
| finisher_mae | FP1 | 3.292 | 2 | 0.000 |
| weighted_mae | PRE | 3.934 | 2 | 0.011 |
| weighted_mae | FP1 | 3.881 | 2 | 0.000 |
| top_heavy_weighted_mae | PRE | 4.113 | 2 | 0.019 |
| top_heavy_weighted_mae | FP1 | 4.060 | 2 | 0.000 |

### race_views: end_to_end_predicted_grid

| metric | checkpoint | mean | n_events | mean_per_event_seed_std |
|---|---|---|---|---|
| finisher_mae | PRE | 4.722 | 2 | 0.112 |
| finisher_mae | FP1 | 4.731 | 2 | 0.063 |
| weighted_mae | PRE | 6.444 | 2 | 0.535 |
| weighted_mae | FP1 | 5.453 | 2 | 0.568 |
| top_heavy_weighted_mae | PRE | 5.830 | 2 | 0.133 |
| top_heavy_weighted_mae | FP1 | 5.756 | 2 | 0.081 |

### Per-event delta (worst-degrading events first)

`conditional_actual_grid`:
```json
[
  {
    "event_id": "2026_09_british_grand_prix",
    "PRE": 4.05,
    "FP1": 4.05,
    "delta": 0.0
  },
  {
    "event_id": "2026_02_chinese_grand_prix",
    "PRE": 2.6666666666666665,
    "FP1": 2.533333333333333,
    "delta": -0.1333333333333333
  }
]
```
`end_to_end_predicted_grid`:
```json
[
  {
    "event_id": "2026_09_british_grand_prix",
    "PRE": 4.8,
    "FP1": 4.883333333333334,
    "delta": 0.08333333333333393
  },
  {
    "event_id": "2026_02_chinese_grand_prix",
    "PRE": 4.644444444444445,
    "FP1": 4.577777777777778,
    "delta": -0.06666666666666643
  }
]
```
`qualifying`:
```json
[
  {
    "event_id": "2026_09_british_grand_prix",
    "PRE": 2.6666666666666665,
    "FP1": 2.6363636363636362,
    "delta": -0.030303030303030276
  },
  {
    "event_id": "2026_02_chinese_grand_prix",
    "PRE": 3.757575757575758,
    "FP1": 3.5454545454545454,
    "delta": -0.21212121212121238
  }
]
```


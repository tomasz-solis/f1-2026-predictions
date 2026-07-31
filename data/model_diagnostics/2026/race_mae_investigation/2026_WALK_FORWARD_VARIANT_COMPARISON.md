# 2026 Walk-Forward Race-MAE Variant Comparison (Phase 2)

Generated: 2026-07-19T10:27:54.242460Z
Events in catalog: 9 (['2026_01_australian_grand_prix', '2026_02_chinese_grand_prix', '2026_03_japanese_grand_prix', '2026_04_miami_grand_prix', '2026_05_canadian_grand_prix', '2026_06_monaco_grand_prix', '2026_07_barcelona_grand_prix', '2026_08_austrian_grand_prix', '2026_09_british_grand_prix'])
Wet events excluded from all variant scoring: ['2026_04_miami_grand_prix', '2026_05_canadian_grand_prix']

## Variant status

- `q0_driver_state`: scored (7 event/checkpoint rows)
- `q1_qualifying_practice`: REFUSED -- ValueError: walk-forward replay produced no eligible scored events
- `r0_long_run`: scored (7 event/checkpoint rows)
- `r1_joint_grid`: scored (7 event/checkpoint rows)
- `r1_r2_no_anchor`: REFUSED -- ValueError: prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=19, p5=11, p95=17)
- `r1_r2_source_anchor`: REFUSED -- ValueError: walk-forward replay produced no eligible scored events
- `r2_no_anchor`: REFUSED -- ValueError: prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=19, p5=11, p95=17)
- `r2_source_anchor`: REFUSED -- ValueError: walk-forward replay produced no eligible scored events

## Race-view comparison (mean across scored events, PRE checkpoint)

| variant | view | role | mae | finisher_mae | weighted_mae | top_heavy_weighted_mae | top_3_pct | top_10_pct | winner_acc_% | spearman | kendall | n_events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| champion | conditional_actual_grid | champion | 3.701 | 3.083 | 3.735 | 3.827 | 69.841 | 75.714 | 52.381 | 0.646 | 0.537 | 7 |
| q0_driver_state | conditional_actual_grid | challenger | 3.701 | 3.083 | 3.735 | 3.827 | 69.841 | 75.714 | 52.381 | 0.646 | 0.537 | 7 |
| champion | end_to_end_predicted_grid | champion | 4.874 | 4.342 | 5.370 | 5.095 | 42.857 | 67.143 | 14.286 | 0.484 | 0.375 | 7 |
| q0_driver_state | end_to_end_predicted_grid | challenger | 4.939 | 4.405 | 5.315 | 5.107 | 41.270 | 65.714 | 14.286 | 0.480 | 0.366 | 7 |
| champion | conditional_actual_grid | champion | 3.701 | 3.083 | 3.735 | 3.827 | 69.841 | 75.714 | 52.381 | 0.646 | 0.537 | 7 |
| r0_long_run | conditional_actual_grid | challenger | 3.701 | 3.083 | 3.735 | 3.827 | 69.841 | 75.714 | 52.381 | 0.646 | 0.537 | 7 |
| champion | end_to_end_predicted_grid | champion | 4.874 | 4.342 | 5.370 | 5.095 | 42.857 | 67.143 | 14.286 | 0.484 | 0.375 | 7 |
| r0_long_run | end_to_end_predicted_grid | challenger | 4.874 | 4.342 | 5.370 | 5.095 | 42.857 | 67.143 | 14.286 | 0.484 | 0.375 | 7 |
| champion | conditional_actual_grid | champion | 3.701 | 3.083 | 3.735 | 3.827 | 69.841 | 75.714 | 52.381 | 0.646 | 0.537 | 7 |
| r1_joint_grid | conditional_actual_grid | challenger | 3.701 | 3.083 | 3.735 | 3.827 | 69.841 | 75.714 | 52.381 | 0.646 | 0.537 | 7 |
| champion | end_to_end_predicted_grid | champion | 4.874 | 4.342 | 5.370 | 5.095 | 42.857 | 67.143 | 14.286 | 0.484 | 0.375 | 7 |
| r1_joint_grid | end_to_end_predicted_grid | challenger | 4.801 | 4.270 | 5.230 | 5.052 | 44.444 | 65.714 | 19.048 | 0.490 | 0.380 | 7 |

## q0 conditional_actual_grid invariance check

```json
{
  "checked": true,
  "checkpoints_checked": 7,
  "passed": true,
  "mismatches": []
}
```

## Per-round trend (champion finisher_mae, conditional_actual_grid, PRE)

| round | event | finisher_mae |
|---|---|---|
| 1 | 2026_01_australian_grand_prix | 3.396 |
| 2 | 2026_02_chinese_grand_prix | 2.667 |
| 3 | 2026_03_japanese_grand_prix | 1.950 |
| 6 | 2026_06_monaco_grand_prix | 4.312 |
| 7 | 2026_07_barcelona_grand_prix | 3.392 |
| 8 | 2026_08_austrian_grand_prix | 1.815 |
| 9 | 2026_09_british_grand_prix | 4.050 |

## DNF floor (champion, conditional_actual_grid, PRE)

| event | all_driver_mae | finisher_mae | dnf_mae_contribution |
|---|---|---|---|
| 2026_01_australian_grand_prix | 4.788 | 3.396 | 1.392 |
| 2026_02_chinese_grand_prix | 3.879 | 2.667 | 1.212 |
| 2026_03_japanese_grand_prix | 2.212 | 1.950 | 0.262 |
| 2026_06_monaco_grand_prix | 5.515 | 4.312 | 1.203 |
| 2026_07_barcelona_grand_prix | 3.273 | 3.392 | -0.119 |
| 2026_08_austrian_grand_prix | 1.879 | 1.815 | 0.064 |
| 2026_09_british_grand_prix | 4.364 | 4.050 | 0.314 |

## Sprint vs main weekend format (mae, PRE, conditional_actual_grid, champion)

- sprint: {"mean": 4.121212121212121, "std": 0.2424242424242422, "n_events": 2}
- main: {"mean": 3.5333333333333337, "std": 1.4179227942680979, "n_events": 5}

## Driver cohort decomposition (champion, finishers only, all cached checkpoints/seeds)

```json
{
  "computed": true,
  "basis": "champion conditional_actual_grid, finishers only, all cached checkpoints/seeds pooled",
  "by_cohort": {
    "established": {
      "finisher_mae": 2.955056179775281,
      "n_driver_observations": 267
    },
    "rookie": {
      "finisher_mae": 3.119047619047619,
      "n_driver_observations": 42
    },
    "second_year": {
      "finisher_mae": 3.526315789473684,
      "n_driver_observations": 57
    }
  }
}
```


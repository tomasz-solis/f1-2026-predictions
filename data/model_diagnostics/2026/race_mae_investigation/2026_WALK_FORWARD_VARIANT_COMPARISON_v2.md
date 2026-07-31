# 2026 Walk-Forward Race-MAE Variant Comparison v2 (Phase 3: FP checkpoints + 500-sim + research-gate-relaxed Q1/R2-source-anchor)

Generated: 2026-07-19T21:29:54.043354Z
Supersedes: nothing -- phase-2 (PRE-only, 20-sim) report kept as-is at `2026_walk_forward_variant_comparison.json` / `.md`. This report adds the FP-checkpoint, 500-sim campaign and the research-gate-relaxed Q1/r2_source_anchor runs.

## Runs merged

```json
[
  {
    "tag": "",
    "found": true,
    "qualifying_simulations": 20,
    "race_simulations": 20,
    "checkpoints_filter": [
      "PRE"
    ],
    "main_checkpoints_filter": null,
    "sprint_checkpoints_filter": null
  },
  {
    "tag": "fp_hisim2",
    "found": true,
    "qualifying_simulations": 500,
    "race_simulations": 500,
    "checkpoints_filter": null,
    "main_checkpoints_filter": [
      "FP2",
      "FP3"
    ],
    "sprint_checkpoints_filter": [
      "FP1"
    ]
  },
  {
    "tag": "relaxed_gate",
    "found": true,
    "qualifying_simulations": 500,
    "race_simulations": 500,
    "checkpoints_filter": null,
    "main_checkpoints_filter": [
      "FP2",
      "FP3"
    ],
    "sprint_checkpoints_filter": [
      "FP1"
    ]
  },
  {
    "tag": "q1_track_classes",
    "found": true,
    "qualifying_simulations": 500,
    "race_simulations": 500,
    "checkpoints_filter": null,
    "main_checkpoints_filter": [
      "FP2",
      "FP3"
    ],
    "sprint_checkpoints_filter": [
      "FP1"
    ]
  },
  {
    "tag": "q1_retro",
    "found": true,
    "qualifying_simulations": 500,
    "race_simulations": 500,
    "checkpoints_filter": null,
    "main_checkpoints_filter": null,
    "sprint_checkpoints_filter": null
  }
]
```

## Variant status

- `q0_driver_state`: scored (tag='')
- `q0_driver_state__fp_hisim2`: scored (tag='fp_hisim2'), 2 checkpoint refusal(s)
- `q1_qualifying_practice`: refused -- walk-forward replay produced no eligible scored events
- `q1_qualifying_practice__q1_retro`: scored (tag='q1_retro'), 5 checkpoint refusal(s), research_gate_relaxation=[{'component': 'q1', 'original_threshold': 8, 'relaxed_threshold': 4, 'shrinkage_applied': 0.625, 'training_events_used': 5}]
- `q1_qualifying_practice__q1_track_classes`: refused -- walk-forward replay produced no eligible scored events
- `q1_qualifying_practice__relaxed_gate`: refused -- walk-forward replay produced no eligible scored events
- `r0_long_run`: scored (tag='')
- `r0_long_run__fp_hisim2`: scored (tag='fp_hisim2'), 2 checkpoint refusal(s)
- `r1_joint_grid`: scored (tag='')
- `r1_joint_grid__fp_hisim2`: scored (tag='fp_hisim2'), 2 checkpoint refusal(s)
- `r1_r2_no_anchor`: refused -- prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=19, p5=11, p95=17)
- `r1_r2_source_anchor`: refused -- walk-forward replay produced no eligible scored events
- `r1_r2_source_anchor__relaxed_gate`: refused -- prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=9, p5=10, p95=15)
- `r2_no_anchor`: refused -- prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=19, p5=11, p95=17)
- `r2_source_anchor`: refused -- walk-forward replay produced no eligible scored events
- `r2_source_anchor__relaxed_gate`: refused -- prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=9, p5=10, p95=15)

## Per-event-checkpoint refusals (loud, never silently dropped)

- `q0_driver_state__fp_hisim2` 2026_07_barcelona_grand_prix FP2 (CheckpointInputUnavailable): 2026_07_barcelona_grand_prix FP2: required session 'Practice 1' could not be extracted: Could not replay Barcelona Grand Prix FP1: C:\Users\tomas\Documents\repos\trackside-labs\data\raw\.fastf1_cache: Sessions were found, but no usable team telemetry could be extracted yet. This usually means the session has too little completed running. Detected sessions: ['Barcelona Grand Prix::FP1']. Extraction diagnostics: ['FP1: teams=1 mapped=0 perf_teams=0 tire_teams=0 selected_laps=0 profile=balanced']
- `q0_driver_state__fp_hisim2` 2026_07_barcelona_grand_prix FP3 (CheckpointInputUnavailable): 2026_07_barcelona_grand_prix FP3: required session 'Practice 1' could not be extracted: Could not replay Barcelona Grand Prix FP1: C:\Users\tomas\Documents\repos\trackside-labs\data\raw\.fastf1_cache: Sessions were found, but no usable team telemetry could be extracted yet. This usually means the session has too little completed running. Detected sessions: ['Barcelona Grand Prix::FP1']. Extraction diagnostics: ['FP1: teams=1 mapped=0 perf_teams=0 tire_teams=0 selected_laps=0 profile=balanced']
- `q1_qualifying_practice__q1_retro` 2026_08_austrian_grand_prix PRE (CheckpointInputUnavailable): 2026_08_austrian_grand_prix PRE: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix
- `q1_qualifying_practice__q1_retro` 2026_08_austrian_grand_prix FP1 (CheckpointInputUnavailable): 2026_08_austrian_grand_prix FP1: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix
- `q1_qualifying_practice__q1_retro` 2026_08_austrian_grand_prix FP2 (CheckpointInputUnavailable): 2026_08_austrian_grand_prix FP2: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix
- `q1_qualifying_practice__q1_retro` 2026_08_austrian_grand_prix FP3 (CheckpointInputUnavailable): 2026_08_austrian_grand_prix FP3: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix
- `q1_qualifying_practice__q1_retro` 2026_09_british_grand_prix PRE (CheckpointInputUnavailable): 2026_09_british_grand_prix PRE: q1 research fit refused -- the practice-comparison fitter needs a real FP checkpoint (PRE has no practice evidence to fit from in this research pass)
- `r0_long_run__fp_hisim2` 2026_07_barcelona_grand_prix FP2 (CheckpointInputUnavailable): 2026_07_barcelona_grand_prix FP2: required session 'Practice 1' could not be extracted: Could not replay Barcelona Grand Prix FP1: C:\Users\tomas\Documents\repos\trackside-labs\data\raw\.fastf1_cache: Sessions were found, but no usable team telemetry could be extracted yet. This usually means the session has too little completed running. Detected sessions: ['Barcelona Grand Prix::FP1']. Extraction diagnostics: ['FP1: teams=1 mapped=0 perf_teams=0 tire_teams=0 selected_laps=0 profile=balanced']
- `r0_long_run__fp_hisim2` 2026_07_barcelona_grand_prix FP3 (CheckpointInputUnavailable): 2026_07_barcelona_grand_prix FP3: required session 'Practice 1' could not be extracted: Could not replay Barcelona Grand Prix FP1: C:\Users\tomas\Documents\repos\trackside-labs\data\raw\.fastf1_cache: Sessions were found, but no usable team telemetry could be extracted yet. This usually means the session has too little completed running. Detected sessions: ['Barcelona Grand Prix::FP1']. Extraction diagnostics: ['FP1: teams=1 mapped=0 perf_teams=0 tire_teams=0 selected_laps=0 profile=balanced']
- `r1_joint_grid__fp_hisim2` 2026_07_barcelona_grand_prix FP2 (CheckpointInputUnavailable): 2026_07_barcelona_grand_prix FP2: required session 'Practice 1' could not be extracted: Could not replay Barcelona Grand Prix FP1: C:\Users\tomas\Documents\repos\trackside-labs\data\raw\.fastf1_cache: Sessions were found, but no usable team telemetry could be extracted yet. This usually means the session has too little completed running. Detected sessions: ['Barcelona Grand Prix::FP1']. Extraction diagnostics: ['FP1: teams=1 mapped=0 perf_teams=0 tire_teams=0 selected_laps=0 profile=balanced']
- `r1_joint_grid__fp_hisim2` 2026_07_barcelona_grand_prix FP3 (CheckpointInputUnavailable): 2026_07_barcelona_grand_prix FP3: required session 'Practice 1' could not be extracted: Could not replay Barcelona Grand Prix FP1: C:\Users\tomas\Documents\repos\trackside-labs\data\raw\.fastf1_cache: Sessions were found, but no usable team telemetry could be extracted yet. This usually means the session has too little completed running. Detected sessions: ['Barcelona Grand Prix::FP1']. Extraction diagnostics: ['FP1: teams=1 mapped=0 perf_teams=0 tire_teams=0 selected_laps=0 profile=balanced']

## r2 interval-coherence finding (first-class result, not a bug)

First-class result, not a bug: every r2 variant that reached scoring (r2_no_anchor and r1_r2_no_anchor at PRE/20-sim in the untangled baseline run; r2_source_anchor and r1_r2_source_anchor at FP-checkpoint/500-sim with the calibrated anchor in relaxed_gate) refused whole-variant with the SAME class of error: a simulated finish position fell outside the champion-computed p5-p95 grid interval. r2_source_anchor's calibrated anchor weight (fit for real from r2_no_anchor's own pre-anchor simulated positions, shrunk toward champion's resolved weight) does not fix this -- it still fails. champion's own (uncalibrated-away) grid anchor is load-bearing for keeping simulated positions inside the declared uncertainty band; removing or only-partially-restoring it breaks interval coherence regardless of calibration. Recommendation: interval recalibration (widening p5-p95 or re-deriving it jointly with any r2 variant) is future work. The coherence validator itself (validate_qualifying_grid) is intentionally NOT weakened to force a score -- doing so would hide a real miscalibration, not fix it.

| variant_key | run_tag | error |
|---|---|---|
| `r1_r2_no_anchor` | None | prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=19, p5=11, p95=17) |
| `r1_r2_source_anchor` | None | walk-forward replay produced no eligible scored events |
| `r2_no_anchor` | None | prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=19, p5=11, p95=17) |
| `r2_source_anchor` | None | walk-forward replay produced no eligible scored events |
| `r1_r2_source_anchor__relaxed_gate` | None | prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=9, p5=10, p95=15) |
| `r2_source_anchor__relaxed_gate` | None | prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=9, p5=10, p95=15) |

## Q1 eligibility table (every event-checkpoint ever attempted)

| event_id | checkpoint | outcome | reason / relaxation | run_tag |
|---|---|---|---|---|
| 2026_08_austrian_grand_prix | FP1 | refused | 2026_08_austrian_grand_prix FP1: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix | 'q1_retro' |
| 2026_08_austrian_grand_prix | FP2 | refused | 2026_08_austrian_grand_prix FP2: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix | 'q1_retro' |
| 2026_08_austrian_grand_prix | FP3 | refused | 2026_08_austrian_grand_prix FP3: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix | 'q1_retro' |
| 2026_08_austrian_grand_prix | PRE | refused | 2026_08_austrian_grand_prix PRE: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix | 'q1_retro' |
| 2026_09_british_grand_prix | FP1 | scored | research_gate_relaxation={'component': 'q1', 'original_threshold': 8, 'relaxed_threshold': 4, 'shrinkage_applied': 0.625, 'training_events_used': 5} | 'q1_retro' |
| 2026_09_british_grand_prix | PRE | refused | 2026_09_british_grand_prix PRE: q1 research fit refused -- the practice-comparison fitter needs a real FP checkpoint (PRE has no practice evidence to fit from in this research pass) | 'q1_retro' |

## Q1 practice-challenger runtime activation (did the fitted model actually get used?)

`q1_qualifying_practice__q1_retro`: any_activated=False
```json
[
  {
    "event_id": "2026_09_british_grand_prix",
    "checkpoint": "FP1",
    "seed": 17,
    "used": false,
    "fallback_reason": "no_raw_practice_laps"
  },
  {
    "event_id": "2026_09_british_grand_prix",
    "checkpoint": "FP1",
    "seed": 91,
    "used": false,
    "fallback_reason": "no_raw_practice_laps"
  },
  {
    "event_id": "2026_09_british_grand_prix",
    "checkpoint": "FP1",
    "seed": 42,
    "used": false,
    "fallback_reason": "no_raw_practice_laps"
  }
]
```

## Production-untouched statement

All research this round (fp_hisim2, relaxed_gate, q1_track_classes, q1_retro campaigns; the retrospective_diagnostic hatch; the q1 outer-gate pooling fix) ran through ProductionReplayBackend, a read-only research harness that drives the real production predictor with research-only config overrides passed in-memory per prediction call. config/production_config.json was read, never written (sha256 confirmed unchanged below). config/default.yaml's model_variant stayed 'champion' throughout. No champion weights, active artifacts, prediction artifacts, or data/evaluation/ contents were overwritten. The served weekend forecast (Belgian GP, live as of 2026-07-19) was never touched by any research run.

- `config/production_config.json` sha256: `c690aa54e054f05a65f7ce565f0c195533723beaa21951ec63ac9daf4fbb96e1`
- expected: `c690aa54e054f05a65f7ce565f0c195533723beaa21951ec63ac9daf4fbb96e1`
- byte-identical: **True**
- `config/default.yaml`: `model_variant: champion`

## Real anchor calibration + identity guard (research-gate-relaxed variants)

### `q1_qualifying_practice__q1_retro`

Fitted anchor calibrations (fold -> {status, calibrated_weight, shrinkage_weight}):
```json
{}
```
Folds flagged `ineffective_for_fold` (a race-affecting component produced a champion-identical finish order; not scored as a differentiated result):
```json
[]
```

## Q1 / r2_source_anchor research-gate-relaxation: what the relaxation does and doesn't buy

**q1_qualifying_practice**: Final state after three rounds of real fixes (track-class binding, per-fold Bradley-Terry fit, the retrospective_diagnostic chronology hatch, and the outer-gate track-class-aware pooling fix): the fitter's own contract requires an exact (checkpoint, session_kind, track_class) group (docs/QUALIFYING_RACE_CHALLENGER.md); track class comes from the curated data/historical_replay/2026/track_class_by_event.json binding. One fold is now genuinely eligible and genuinely fits: British GP FP1 (5 prior permanent-class training events, real Bradley-Terry model, real launch envelope, retrospective_diagnostic=true throughout). But even there the challenger did not actually activate at inference: qualifying_mixin.py's own runtime guard requires raw per-lap FP telemetry for the TARGET event's own practice session (session_laps_by_type), and the research backend replays every checkpoint in practice_signal_mode='stored_profiles' (aggregated team/driver profiles only) -- so session_laps_by_type is always {} in this harness, for every variant, by construction. The fold scored, is fully disclosed (qualifying_practice_challenger.used=false, fallback_reason='no_raw_practice_laps' on every cached prediction), and is champion-identical as a DIRECT, transparent consequence -- not a silently-hidden fallback. Making the backend load raw per-lap telemetry for every replayed checkpoint is a materially larger change (affects every variant's replay path, reintroduces the FastF1 telemetry-thinness fragility already hit once this project on Barcelona FP1) and was not authorized this round -- reported as a structural finding, not fixed. See q1_practice_activation below.

**r2_source_anchor**: This IS a real calibration: fit_source_specific_grid_anchors runs on genuine (simulated_position, grid_position, actual_position) rows recovered from r2_no_anchor's own predictions on prior events (its anchor weight is fixed at 0.0, so its predicted position literally is the pre-anchor simulated position -- no simulator internals touched). The calibrated weight is shrunk toward champion's own resolved weight in proportion to n_training_events/8 and injected via baseline_predictor.race.grid_anchor.source_calibrated.actual_starting_grid. See research_gate_relaxation_detail per variant below for the real fitted values per fold, and ineffective_folds for any fold where the calibrated prediction still matched champion exactly (flagged, not hidden). end_to_end_predicted_grid was not calibrated (only the conditional_actual_grid source detail) given the time budget, so that view legitimately keeps champion's fallback weight.

## Champion checkpoint progression (end-to-end finisher_mae, closing the grid-propagation gap)

| checkpoint | end_to_end finisher_mae | conditional finisher_mae |
|---|---|---|
| FP1 | 4.731 | 3.292 |
| FP2 | 4.536 | 3.274 |
| FP3 | 4.894 | 3.436 |
| PRE | 4.342 | 3.083 |

## Race-view comparison per scored run

| variant_key | checkpoint | view | role | mae | finisher_mae | weighted_mae | top_heavy_weighted_mae | winner_acc_% | n_events |
|---|---|---|---|---|---|---|---|---|---|
| q0_driver_state:champion | PRE | conditional_actual_grid | champion | 3.701 | 3.083 | 3.735 | 3.827 | 52.381 | 7 |
| q0_driver_state:challenger | PRE | conditional_actual_grid | challenger | 3.701 | 3.083 | 3.735 | 3.827 | 52.381 | 7 |
| q0_driver_state:champion | PRE | end_to_end_predicted_grid | champion | 4.874 | 4.342 | 5.370 | 5.095 | 14.286 | 7 |
| q0_driver_state:challenger | PRE | end_to_end_predicted_grid | challenger | 4.939 | 4.405 | 5.315 | 5.107 | 14.286 | 7 |
| q0_driver_state__fp_hisim2:champion | FP2 | conditional_actual_grid | champion | 3.962 | 3.274 | 4.538 | 4.248 | 25.000 | 4 |
| q0_driver_state__fp_hisim2:challenger | FP2 | conditional_actual_grid | challenger | 3.962 | 3.274 | 4.538 | 4.248 | 25.000 | 4 |
| q0_driver_state__fp_hisim2:champion | FP2 | end_to_end_predicted_grid | champion | 5.083 | 4.536 | 4.831 | 5.275 | 25.000 | 4 |
| q0_driver_state__fp_hisim2:challenger | FP2 | end_to_end_predicted_grid | challenger | 5.174 | 4.659 | 5.069 | 5.340 | 0.000 | 4 |
| q0_driver_state__fp_hisim2:champion | FP3 | conditional_actual_grid | champion | 4.068 | 3.436 | 4.582 | 4.388 | 25.000 | 4 |
| q0_driver_state__fp_hisim2:challenger | FP3 | conditional_actual_grid | challenger | 4.068 | 3.436 | 4.582 | 4.388 | 25.000 | 4 |
| q0_driver_state__fp_hisim2:champion | FP3 | end_to_end_predicted_grid | champion | 5.432 | 4.894 | 5.043 | 5.616 | 25.000 | 4 |
| q0_driver_state__fp_hisim2:challenger | FP3 | end_to_end_predicted_grid | challenger | 5.545 | 5.011 | 5.259 | 5.708 | 8.333 | 4 |
| q0_driver_state__fp_hisim2:champion | FP1 | conditional_actual_grid | champion | 4.045 | 3.292 | 3.881 | 4.060 | 50.000 | 2 |
| q0_driver_state__fp_hisim2:challenger | FP1 | conditional_actual_grid | challenger | 4.045 | 3.292 | 3.881 | 4.060 | 50.000 | 2 |
| q0_driver_state__fp_hisim2:champion | FP1 | end_to_end_predicted_grid | champion | 5.576 | 4.731 | 5.453 | 5.756 | 0.000 | 2 |
| q0_driver_state__fp_hisim2:challenger | FP1 | end_to_end_predicted_grid | challenger | 5.621 | 4.764 | 5.723 | 5.821 | 0.000 | 2 |
| q1_qualifying_practice__q1_retro:champion | FP1 | conditional_actual_grid | champion | 4.364 | 4.050 | 5.045 | 4.410 | 0.000 | 1 |
| q1_qualifying_practice__q1_retro:challenger | FP1 | conditional_actual_grid | challenger | 4.364 | 4.050 | 5.045 | 4.410 | 0.000 | 1 |
| q1_qualifying_practice__q1_retro:champion | FP1 | end_to_end_predicted_grid | champion | 4.970 | 4.883 | 5.353 | 5.333 | 0.000 | 1 |
| q1_qualifying_practice__q1_retro:challenger | FP1 | end_to_end_predicted_grid | challenger | 4.970 | 4.883 | 5.353 | 5.333 | 0.000 | 1 |
| r0_long_run:champion | PRE | conditional_actual_grid | champion | 3.701 | 3.083 | 3.735 | 3.827 | 52.381 | 7 |
| r0_long_run:challenger | PRE | conditional_actual_grid | challenger | 3.701 | 3.083 | 3.735 | 3.827 | 52.381 | 7 |
| r0_long_run:champion | PRE | end_to_end_predicted_grid | champion | 4.874 | 4.342 | 5.370 | 5.095 | 14.286 | 7 |
| r0_long_run:challenger | PRE | end_to_end_predicted_grid | challenger | 4.874 | 4.342 | 5.370 | 5.095 | 14.286 | 7 |
| r0_long_run__fp_hisim2:champion | FP2 | conditional_actual_grid | champion | 3.962 | 3.274 | 4.538 | 4.248 | 25.000 | 4 |
| r0_long_run__fp_hisim2:challenger | FP2 | conditional_actual_grid | challenger | 3.962 | 3.274 | 4.538 | 4.248 | 25.000 | 4 |
| r0_long_run__fp_hisim2:champion | FP2 | end_to_end_predicted_grid | champion | 5.083 | 4.536 | 4.831 | 5.275 | 25.000 | 4 |
| r0_long_run__fp_hisim2:challenger | FP2 | end_to_end_predicted_grid | challenger | 5.083 | 4.536 | 4.831 | 5.275 | 25.000 | 4 |
| r0_long_run__fp_hisim2:champion | FP3 | conditional_actual_grid | champion | 4.068 | 3.436 | 4.582 | 4.388 | 25.000 | 4 |
| r0_long_run__fp_hisim2:challenger | FP3 | conditional_actual_grid | challenger | 4.068 | 3.436 | 4.582 | 4.388 | 25.000 | 4 |
| r0_long_run__fp_hisim2:champion | FP3 | end_to_end_predicted_grid | champion | 5.432 | 4.894 | 5.043 | 5.616 | 25.000 | 4 |
| r0_long_run__fp_hisim2:challenger | FP3 | end_to_end_predicted_grid | challenger | 5.432 | 4.894 | 5.043 | 5.616 | 25.000 | 4 |
| r0_long_run__fp_hisim2:champion | FP1 | conditional_actual_grid | champion | 4.045 | 3.292 | 3.881 | 4.060 | 50.000 | 2 |
| r0_long_run__fp_hisim2:challenger | FP1 | conditional_actual_grid | challenger | 4.045 | 3.292 | 3.881 | 4.060 | 50.000 | 2 |
| r0_long_run__fp_hisim2:champion | FP1 | end_to_end_predicted_grid | champion | 5.576 | 4.731 | 5.453 | 5.756 | 0.000 | 2 |
| r0_long_run__fp_hisim2:challenger | FP1 | end_to_end_predicted_grid | challenger | 5.576 | 4.731 | 5.453 | 5.756 | 0.000 | 2 |
| r1_joint_grid:champion | PRE | conditional_actual_grid | champion | 3.701 | 3.083 | 3.735 | 3.827 | 52.381 | 7 |
| r1_joint_grid:challenger | PRE | conditional_actual_grid | challenger | 3.701 | 3.083 | 3.735 | 3.827 | 52.381 | 7 |
| r1_joint_grid:champion | PRE | end_to_end_predicted_grid | champion | 4.874 | 4.342 | 5.370 | 5.095 | 14.286 | 7 |
| r1_joint_grid:challenger | PRE | end_to_end_predicted_grid | challenger | 4.801 | 4.270 | 5.230 | 5.052 | 19.048 | 7 |
| r1_joint_grid__fp_hisim2:champion | FP2 | conditional_actual_grid | champion | 3.962 | 3.274 | 4.538 | 4.248 | 25.000 | 4 |
| r1_joint_grid__fp_hisim2:challenger | FP2 | conditional_actual_grid | challenger | 3.962 | 3.274 | 4.538 | 4.248 | 25.000 | 4 |
| r1_joint_grid__fp_hisim2:champion | FP2 | end_to_end_predicted_grid | champion | 5.083 | 4.536 | 4.831 | 5.275 | 25.000 | 4 |
| r1_joint_grid__fp_hisim2:challenger | FP2 | end_to_end_predicted_grid | challenger | 5.121 | 4.584 | 4.905 | 5.311 | 25.000 | 4 |
| r1_joint_grid__fp_hisim2:champion | FP3 | conditional_actual_grid | champion | 4.068 | 3.436 | 4.582 | 4.388 | 25.000 | 4 |
| r1_joint_grid__fp_hisim2:challenger | FP3 | conditional_actual_grid | challenger | 4.068 | 3.436 | 4.582 | 4.388 | 25.000 | 4 |
| r1_joint_grid__fp_hisim2:champion | FP3 | end_to_end_predicted_grid | champion | 5.432 | 4.894 | 5.043 | 5.616 | 25.000 | 4 |
| r1_joint_grid__fp_hisim2:challenger | FP3 | end_to_end_predicted_grid | challenger | 5.424 | 4.902 | 5.015 | 5.587 | 25.000 | 4 |
| r1_joint_grid__fp_hisim2:champion | FP1 | conditional_actual_grid | champion | 4.045 | 3.292 | 3.881 | 4.060 | 50.000 | 2 |
| r1_joint_grid__fp_hisim2:challenger | FP1 | conditional_actual_grid | challenger | 4.045 | 3.292 | 3.881 | 4.060 | 50.000 | 2 |
| r1_joint_grid__fp_hisim2:champion | FP1 | end_to_end_predicted_grid | champion | 5.576 | 4.731 | 5.453 | 5.756 | 0.000 | 2 |
| r1_joint_grid__fp_hisim2:challenger | FP1 | end_to_end_predicted_grid | challenger | 5.515 | 4.656 | 5.124 | 5.690 | 0.000 | 2 |

## Per-seed spread (mean per-event std of finisher_mae across the 3 seeds)

| variant_key | checkpoint:view:role | mean_per_event_seed_std | n_events |
|---|---|---|---|
| q0_driver_state | FP1:conditional_actual_grid:challenger | 0.000 | 2 |
| q0_driver_state | FP1:conditional_actual_grid:champion | 0.000 | 2 |
| q0_driver_state | FP1:end_to_end_predicted_grid:challenger | 0.090 | 2 |
| q0_driver_state | FP1:end_to_end_predicted_grid:champion | 0.063 | 2 |
| q0_driver_state | FP2:conditional_actual_grid:challenger | 0.013 | 4 |
| q0_driver_state | FP2:conditional_actual_grid:champion | 0.013 | 4 |
| q0_driver_state | FP2:end_to_end_predicted_grid:challenger | 0.025 | 4 |
| q0_driver_state | FP2:end_to_end_predicted_grid:champion | 0.061 | 4 |
| q0_driver_state | FP3:conditional_actual_grid:challenger | 0.020 | 4 |
| q0_driver_state | FP3:conditional_actual_grid:champion | 0.020 | 4 |
| q0_driver_state | FP3:end_to_end_predicted_grid:challenger | 0.069 | 4 |
| q0_driver_state | FP3:end_to_end_predicted_grid:champion | 0.101 | 4 |
| q0_driver_state | PRE:conditional_actual_grid:challenger | 0.121 | 7 |
| q0_driver_state | PRE:conditional_actual_grid:champion | 0.121 | 7 |
| q0_driver_state | PRE:end_to_end_predicted_grid:challenger | 0.133 | 7 |
| q0_driver_state | PRE:end_to_end_predicted_grid:champion | 0.129 | 7 |
| q0_driver_state__fp_hisim2 | FP1:conditional_actual_grid:challenger | 0.000 | 2 |
| q0_driver_state__fp_hisim2 | FP1:conditional_actual_grid:champion | 0.000 | 2 |
| q0_driver_state__fp_hisim2 | FP1:end_to_end_predicted_grid:challenger | 0.090 | 2 |
| q0_driver_state__fp_hisim2 | FP1:end_to_end_predicted_grid:champion | 0.063 | 2 |
| q0_driver_state__fp_hisim2 | FP2:conditional_actual_grid:challenger | 0.013 | 4 |
| q0_driver_state__fp_hisim2 | FP2:conditional_actual_grid:champion | 0.013 | 4 |
| q0_driver_state__fp_hisim2 | FP2:end_to_end_predicted_grid:challenger | 0.025 | 4 |
| q0_driver_state__fp_hisim2 | FP2:end_to_end_predicted_grid:champion | 0.061 | 4 |
| q0_driver_state__fp_hisim2 | FP3:conditional_actual_grid:challenger | 0.020 | 4 |
| q0_driver_state__fp_hisim2 | FP3:conditional_actual_grid:champion | 0.020 | 4 |
| q0_driver_state__fp_hisim2 | FP3:end_to_end_predicted_grid:challenger | 0.069 | 4 |
| q0_driver_state__fp_hisim2 | FP3:end_to_end_predicted_grid:champion | 0.101 | 4 |
| q0_driver_state__fp_hisim2 | PRE:conditional_actual_grid:challenger | 0.121 | 7 |
| q0_driver_state__fp_hisim2 | PRE:conditional_actual_grid:champion | 0.121 | 7 |
| q0_driver_state__fp_hisim2 | PRE:end_to_end_predicted_grid:challenger | 0.133 | 7 |
| q0_driver_state__fp_hisim2 | PRE:end_to_end_predicted_grid:champion | 0.129 | 7 |
| q1_qualifying_practice__q1_retro | FP1:conditional_actual_grid:challenger | 0.000 | 1 |
| q1_qualifying_practice__q1_retro | FP1:conditional_actual_grid:champion | 0.000 | 2 |
| q1_qualifying_practice__q1_retro | FP1:end_to_end_predicted_grid:challenger | 0.062 | 1 |
| q1_qualifying_practice__q1_retro | FP1:end_to_end_predicted_grid:champion | 0.063 | 2 |
| q1_qualifying_practice__q1_retro | FP2:conditional_actual_grid:champion | 0.013 | 4 |
| q1_qualifying_practice__q1_retro | FP2:end_to_end_predicted_grid:champion | 0.061 | 4 |
| q1_qualifying_practice__q1_retro | FP3:conditional_actual_grid:champion | 0.020 | 4 |
| q1_qualifying_practice__q1_retro | FP3:end_to_end_predicted_grid:champion | 0.101 | 4 |
| q1_qualifying_practice__q1_retro | PRE:conditional_actual_grid:champion | 0.121 | 7 |
| q1_qualifying_practice__q1_retro | PRE:end_to_end_predicted_grid:champion | 0.129 | 7 |
| r0_long_run | FP1:conditional_actual_grid:challenger | 0.000 | 2 |
| r0_long_run | FP1:conditional_actual_grid:champion | 0.000 | 2 |
| r0_long_run | FP1:end_to_end_predicted_grid:challenger | 0.063 | 2 |
| r0_long_run | FP1:end_to_end_predicted_grid:champion | 0.063 | 2 |
| r0_long_run | FP2:conditional_actual_grid:challenger | 0.013 | 4 |
| r0_long_run | FP2:conditional_actual_grid:champion | 0.013 | 4 |
| r0_long_run | FP2:end_to_end_predicted_grid:challenger | 0.061 | 4 |
| r0_long_run | FP2:end_to_end_predicted_grid:champion | 0.061 | 4 |
| r0_long_run | FP3:conditional_actual_grid:challenger | 0.020 | 4 |
| r0_long_run | FP3:conditional_actual_grid:champion | 0.020 | 4 |
| r0_long_run | FP3:end_to_end_predicted_grid:challenger | 0.101 | 4 |
| r0_long_run | FP3:end_to_end_predicted_grid:champion | 0.101 | 4 |
| r0_long_run | PRE:conditional_actual_grid:challenger | 0.121 | 7 |
| r0_long_run | PRE:conditional_actual_grid:champion | 0.121 | 7 |
| r0_long_run | PRE:end_to_end_predicted_grid:challenger | 0.129 | 7 |
| r0_long_run | PRE:end_to_end_predicted_grid:champion | 0.129 | 7 |
| r0_long_run__fp_hisim2 | FP1:conditional_actual_grid:challenger | 0.000 | 2 |
| r0_long_run__fp_hisim2 | FP1:conditional_actual_grid:champion | 0.000 | 2 |
| r0_long_run__fp_hisim2 | FP1:end_to_end_predicted_grid:challenger | 0.063 | 2 |
| r0_long_run__fp_hisim2 | FP1:end_to_end_predicted_grid:champion | 0.063 | 2 |
| r0_long_run__fp_hisim2 | FP2:conditional_actual_grid:challenger | 0.013 | 4 |
| r0_long_run__fp_hisim2 | FP2:conditional_actual_grid:champion | 0.013 | 4 |
| r0_long_run__fp_hisim2 | FP2:end_to_end_predicted_grid:challenger | 0.061 | 4 |
| r0_long_run__fp_hisim2 | FP2:end_to_end_predicted_grid:champion | 0.061 | 4 |
| r0_long_run__fp_hisim2 | FP3:conditional_actual_grid:challenger | 0.020 | 4 |
| r0_long_run__fp_hisim2 | FP3:conditional_actual_grid:champion | 0.020 | 4 |
| r0_long_run__fp_hisim2 | FP3:end_to_end_predicted_grid:challenger | 0.101 | 4 |
| r0_long_run__fp_hisim2 | FP3:end_to_end_predicted_grid:champion | 0.101 | 4 |
| r0_long_run__fp_hisim2 | PRE:conditional_actual_grid:challenger | 0.121 | 7 |
| r0_long_run__fp_hisim2 | PRE:conditional_actual_grid:champion | 0.121 | 7 |
| r0_long_run__fp_hisim2 | PRE:end_to_end_predicted_grid:challenger | 0.129 | 7 |
| r0_long_run__fp_hisim2 | PRE:end_to_end_predicted_grid:champion | 0.129 | 7 |
| r1_joint_grid | FP1:conditional_actual_grid:challenger | 0.000 | 2 |
| r1_joint_grid | FP1:conditional_actual_grid:champion | 0.000 | 2 |
| r1_joint_grid | FP1:end_to_end_predicted_grid:challenger | 0.016 | 2 |
| r1_joint_grid | FP1:end_to_end_predicted_grid:champion | 0.063 | 2 |
| r1_joint_grid | FP2:conditional_actual_grid:challenger | 0.013 | 4 |
| r1_joint_grid | FP2:conditional_actual_grid:champion | 0.013 | 4 |
| r1_joint_grid | FP2:end_to_end_predicted_grid:challenger | 0.032 | 4 |
| r1_joint_grid | FP2:end_to_end_predicted_grid:champion | 0.061 | 4 |
| r1_joint_grid | FP3:conditional_actual_grid:challenger | 0.020 | 4 |
| r1_joint_grid | FP3:conditional_actual_grid:champion | 0.020 | 4 |
| r1_joint_grid | FP3:end_to_end_predicted_grid:challenger | 0.059 | 4 |
| r1_joint_grid | FP3:end_to_end_predicted_grid:champion | 0.101 | 4 |
| r1_joint_grid | PRE:conditional_actual_grid:challenger | 0.121 | 7 |
| r1_joint_grid | PRE:conditional_actual_grid:champion | 0.121 | 7 |
| r1_joint_grid | PRE:end_to_end_predicted_grid:challenger | 0.131 | 7 |
| r1_joint_grid | PRE:end_to_end_predicted_grid:champion | 0.129 | 7 |
| r1_joint_grid__fp_hisim2 | FP1:conditional_actual_grid:challenger | 0.000 | 2 |
| r1_joint_grid__fp_hisim2 | FP1:conditional_actual_grid:champion | 0.000 | 2 |
| r1_joint_grid__fp_hisim2 | FP1:end_to_end_predicted_grid:challenger | 0.016 | 2 |
| r1_joint_grid__fp_hisim2 | FP1:end_to_end_predicted_grid:champion | 0.063 | 2 |
| r1_joint_grid__fp_hisim2 | FP2:conditional_actual_grid:challenger | 0.013 | 4 |
| r1_joint_grid__fp_hisim2 | FP2:conditional_actual_grid:champion | 0.013 | 4 |
| r1_joint_grid__fp_hisim2 | FP2:end_to_end_predicted_grid:challenger | 0.032 | 4 |
| r1_joint_grid__fp_hisim2 | FP2:end_to_end_predicted_grid:champion | 0.061 | 4 |
| r1_joint_grid__fp_hisim2 | FP3:conditional_actual_grid:challenger | 0.020 | 4 |
| r1_joint_grid__fp_hisim2 | FP3:conditional_actual_grid:champion | 0.020 | 4 |
| r1_joint_grid__fp_hisim2 | FP3:end_to_end_predicted_grid:challenger | 0.059 | 4 |
| r1_joint_grid__fp_hisim2 | FP3:end_to_end_predicted_grid:champion | 0.101 | 4 |
| r1_joint_grid__fp_hisim2 | PRE:conditional_actual_grid:challenger | 0.121 | 7 |
| r1_joint_grid__fp_hisim2 | PRE:conditional_actual_grid:champion | 0.121 | 7 |
| r1_joint_grid__fp_hisim2 | PRE:end_to_end_predicted_grid:challenger | 0.131 | 7 |
| r1_joint_grid__fp_hisim2 | PRE:end_to_end_predicted_grid:champion | 0.129 | 7 |

## q0 conditional_actual_grid invariance check

```json
{
  "q0_driver_state": {
    "checked": true,
    "checkpoints_checked": 7,
    "passed": true,
    "mismatches": []
  },
  "q0_driver_state__fp_hisim2": {
    "checked": true,
    "checkpoints_checked": 10,
    "passed": true,
    "mismatches": []
  }
}
```

## Driver cohort decomposition (by run)

```json
{
  "q0_driver_state": {
    "computed": true,
    "basis": "champion conditional_actual_grid, finishers only, checkpoints ['PRE']",
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
  },
  "r0_long_run": {
    "computed": true,
    "basis": "champion conditional_actual_grid, finishers only, checkpoints ['PRE']",
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
  },
  "r1_joint_grid": {
    "computed": true,
    "basis": "champion conditional_actual_grid, finishers only, checkpoints ['PRE']",
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
  },
  "q0_driver_state__fp_hisim2": {
    "computed": true,
    "basis": "champion conditional_actual_grid, finishers only, checkpoints ['FP1', 'FP2', 'FP3']",
    "by_cohort": {
      "established": {
        "finisher_mae": 3.328125,
        "n_driver_observations": 384
      },
      "rookie": {
        "finisher_mae": 3.95,
        "n_driver_observations": 60
      },
      "second_year": {
        "finisher_mae": 2.6049382716049383,
        "n_driver_observations": 81
      }
    }
  },
  "r0_long_run__fp_hisim2": {
    "computed": true,
    "basis": "champion conditional_actual_grid, finishers only, checkpoints ['FP1', 'FP2', 'FP3']",
    "by_cohort": {
      "established": {
        "finisher_mae": 3.328125,
        "n_driver_observations": 384
      },
      "rookie": {
        "finisher_mae": 3.95,
        "n_driver_observations": 60
      },
      "second_year": {
        "finisher_mae": 2.6049382716049383,
        "n_driver_observations": 81
      }
    }
  },
  "r1_joint_grid__fp_hisim2": {
    "computed": true,
    "basis": "champion conditional_actual_grid, finishers only, checkpoints ['FP1', 'FP2', 'FP3']",
    "by_cohort": {
      "established": {
        "finisher_mae": 3.328125,
        "n_driver_observations": 384
      },
      "rookie": {
        "finisher_mae": 3.95,
        "n_driver_observations": 60
      },
      "second_year": {
        "finisher_mae": 2.6049382716049383,
        "n_driver_observations": 81
      }
    }
  },
  "q1_qualifying_practice__q1_retro": {
    "computed": true,
    "basis": "champion conditional_actual_grid, finishers only, checkpoints ['FP1']",
    "by_cohort": {
      "established": {
        "finisher_mae": 3.3461538461538463,
        "n_driver_observations": 78
      },
      "rookie": {
        "finisher_mae": 3.5,
        "n_driver_observations": 12
      },
      "second_year": {
        "finisher_mae": 3.6,
        "n_driver_observations": 15
      }
    }
  }
}
```

## Sprint vs main weekend format, DNF floor, per-round trend (from first scored run)

```json
{
  "weekend_format": {
    "PRE:conditional_actual_grid:champion": {
      "sprint": {
        "mean": 4.121212121212121,
        "std": 0.2424242424242422,
        "n_events": 2
      },
      "main": {
        "mean": 3.5333333333333337,
        "std": 1.4179227942680979,
        "n_events": 5
      }
    },
    "PRE:conditional_actual_grid:challenger": {
      "sprint": {
        "mean": 4.121212121212121,
        "std": 0.2424242424242422,
        "n_events": 2
      },
      "main": {
        "mean": 3.5333333333333337,
        "std": 1.4179227942680979,
        "n_events": 5
      }
    },
    "PRE:end_to_end_predicted_grid:champion": {
      "sprint": {
        "mean": 5.651515151515152,
        "std": 0.7727272727272729,
        "n_events": 2
      },
      "main": {
        "mean": 4.5636363636363635,
        "std": 1.3227680579640109,
        "n_events": 5
      }
    },
    "PRE:end_to_end_predicted_grid:challenger": {
      "sprint": {
        "mean": 5.757575757575757,
        "std": 0.545454545454545,
        "n_events": 2
      },
      "main": {
        "mean": 4.612121212121212,
        "std": 1.2469835044708515,
        "n_events": 5
      }
    }
  },
  "per_round_trend_champion_finisher_mae": [
    {
      "event_id": "2026_01_australian_grand_prix",
      "round_number": 1,
      "checkpoint": "PRE",
      "finisher_mae": 3.3958333333333335
    },
    {
      "event_id": "2026_02_chinese_grand_prix",
      "round_number": 2,
      "checkpoint": "PRE",
      "finisher_mae": 2.6666666666666665
    },
    {
      "event_id": "2026_03_japanese_grand_prix",
      "round_number": 3,
      "checkpoint": "PRE",
      "finisher_mae": 1.95
    },
    {
      "event_id": "2026_06_monaco_grand_prix",
      "round_number": 6,
      "checkpoint": "PRE",
      "finisher_mae": 4.3125
    },
    {
      "event_id": "2026_07_barcelona_grand_prix",
      "round_number": 7,
      "checkpoint": "PRE",
      "finisher_mae": 3.392156862745098
    },
    {
      "event_id": "2026_08_austrian_grand_prix",
      "round_number": 8,
      "checkpoint": "PRE",
      "finisher_mae": 1.8148148148148149
    },
    {
      "event_id": "2026_09_british_grand_prix",
      "round_number": 9,
      "checkpoint": "PRE",
      "finisher_mae": 4.05
    }
  ],
  "dnf_floor_champion_pre_conditional": [
    {
      "event_id": "2026_01_australian_grand_prix",
      "all_driver_mae": 4.787878787878788,
      "finisher_mae": 3.3958333333333335,
      "dnf_mae_contribution": 1.3920454545454546
    },
    {
      "event_id": "2026_02_chinese_grand_prix",
      "all_driver_mae": 3.878787878787879,
      "finisher_mae": 2.6666666666666665,
      "dnf_mae_contribution": 1.2121212121212124
    },
    {
      "event_id": "2026_03_japanese_grand_prix",
      "all_driver_mae": 2.2121212121212124,
      "finisher_mae": 1.95,
      "dnf_mae_contribution": 0.2621212121212124
    },
    {
      "event_id": "2026_06_monaco_grand_prix",
      "all_driver_mae": 5.515151515151516,
      "finisher_mae": 4.3125,
      "dnf_mae_contribution": 1.2026515151515156
    },
    {
      "event_id": "2026_07_barcelona_grand_prix",
      "all_driver_mae": 3.272727272727273,
      "finisher_mae": 3.392156862745098,
      "dnf_mae_contribution": -0.11942959001782505
    },
    {
      "event_id": "2026_08_austrian_grand_prix",
      "all_driver_mae": 1.878787878787879,
      "finisher_mae": 1.8148148148148149,
      "dnf_mae_contribution": 0.06397306397306401
    },
    {
      "event_id": "2026_09_british_grand_prix",
      "all_driver_mae": 4.363636363636363,
      "finisher_mae": 4.05,
      "dnf_mae_contribution": 0.3136363636363635
    }
  ]
}
```


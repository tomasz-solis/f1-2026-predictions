# 2026 Walk-Forward Race-MAE Variant Comparison v3 (Phase 4: structural-identity guard + matched-subset progression)

Generated: 2026-07-20T06:43:23.049954Z
Supersedes: 2026_walk_forward_variant_comparison_v2.json (kept, not deleted).

## r0 structural reclassification (first-class finding)

r0_long_run's FP-checkpoint results in fp_hisim2 are RECLASSIFIED: structurally untested (champion-identical: no raw practice laps in replay), not a scored differentiated result. Every r0 challenger prediction at FP1/FP2/FP3 (qualifying grid is trivially identical -- r0 has no qualifying component -- but BOTH race views are also identical, which is not expected: r0's long-run pace effect should show up in race simulation regardless of which grid it started from). Cause: race_practice_challenger.applied=false, fallback_reason='insufficient_field_evidence_coverage' on every cached FP-checkpoint prediction -- the same class of gap that neutralized Q1 (this harness's stored_profiles checkpoint-replay mode never loads raw per-driver practice evidence, for any variant). r0 at PRE remains a legitimate, documented no-op (no practice sessions exist pre-weekend, so there is nothing to extract regardless of harness mode).

## Structural-identity flags (generalized guard, retroactive from cache)

Legend: `legitimate_architectural_invariance` = expected (grid-only component, conditional_actual_grid view discards the predicted grid) -- not a bug. `structural_identity` = champion-identical where the component was expected to differ -- flagged, not silently scored.

### `q0_driver_state` -- {'structural_identity': 8, 'legitimate_architectural_invariance': 21}

### `q0_driver_state__fp_hisim2` -- {'structural_identity': 4, 'legitimate_architectural_invariance': 30}

### `q1_qualifying_practice__q1_retro` -- {'structural_identity': 6, 'legitimate_architectural_invariance': 3}
Disclosed reasons: ['no_raw_practice_laps']

### `r0_long_run` -- {'structural_identity': 42}
Disclosed reasons: ['missing_race_practice_evidence']

### `r0_long_run__fp_hisim2` -- {'structural_identity': 60}
Disclosed reasons: ['insufficient_field_evidence_coverage']

### `r1_joint_grid` -- {'legitimate_architectural_invariance': 21}

### `r1_joint_grid__fp_hisim2` -- {'legitimate_architectural_invariance': 30}

## Champion checkpoint progression -- MATCHED event subsets (restated conclusion)

### Main-dry PRE vs FP2 vs FP3 (n=4 matched events: ['2026_01_australian_grand_prix', '2026_03_japanese_grand_prix', '2026_06_monaco_grand_prix', '2026_08_austrian_grand_prix'])

| checkpoint | end_to_end finisher_mae | conditional finisher_mae |
|---|---|---|
| PRE | 4.164 | 2.868 |
| FP2 | 4.536 | 3.274 |
| FP3 | 4.894 | 3.436 |

### Sprint-dry PRE vs FP1 (n=2 matched events: ['2026_02_chinese_grand_prix', '2026_09_british_grand_prix'])

| checkpoint | end_to_end finisher_mae | conditional finisher_mae |
|---|---|---|
| PRE | 4.722 | 3.358 |
| FP1 | 4.731 | 3.292 |

## Champion checkpoint progression -- UNMATCHED (SUPERSEDED, kept for audit trail only)

This table compares checkpoints scored over DIFFERENT event counts (e.g. PRE over 7 events vs FP1 over 2) -- confounded, do not draw conclusions from it. See the matched-subset tables above for the restated conclusion.

| checkpoint | end_to_end finisher_mae | conditional finisher_mae |
|---|---|---|
| FP1 | 4.731 | 3.292 |
| FP2 | 4.536 | 3.274 |
| FP3 | 4.894 | 3.436 |
| PRE | 4.342 | 3.083 |

## Future work: raw-laps replay handoff

Loading raw per-lap FastF1 telemetry for every replayed checkpoint (instead of the current stored_profiles aggregated-only mode) would let both Q1 and r0 actually activate in this harness. Scoped as its own future handoff, not implemented this round: see docs/RAW_LAPS_REPLAY_HANDOFF.md.

## Q1 eligibility table

| event_id | checkpoint | outcome | reason / relaxation | run_tag |
|---|---|---|---|---|
| 2026_08_austrian_grand_prix | FP1 | refused | 2026_08_austrian_grand_prix FP1: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix | 'q1_retro' |
| 2026_08_austrian_grand_prix | FP2 | refused | 2026_08_austrian_grand_prix FP2: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix | 'q1_retro' |
| 2026_08_austrian_grand_prix | FP3 | refused | 2026_08_austrian_grand_prix FP3: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix | 'q1_retro' |
| 2026_08_austrian_grand_prix | PRE | refused | 2026_08_austrian_grand_prix PRE: q1 research fit refused -- only 3 prior dry 'permanent'-class event(s) < required 4 (research floor) for 2026_08_austrian_grand_prix | 'q1_retro' |
| 2026_09_british_grand_prix | FP1 | scored | research_gate_relaxation={'component': 'q1', 'original_threshold': 8, 'relaxed_threshold': 4, 'shrinkage_applied': 0.625, 'training_events_used': 5} | 'q1_retro' |
| 2026_09_british_grand_prix | PRE | refused | 2026_09_british_grand_prix PRE: q1 research fit refused -- the practice-comparison fitter needs a real FP checkpoint (PRE has no practice evidence to fit from in this research pass) | 'q1_retro' |

## Production-untouched statement

All research this round (fp_hisim2, relaxed_gate, q1_track_classes, q1_retro campaigns; the retrospective_diagnostic hatch; the q1 outer-gate pooling fix) ran through ProductionReplayBackend, a read-only research harness that drives the real production predictor with research-only config overrides passed in-memory per prediction call. config/production_config.json was read, never written (sha256 confirmed unchanged below). config/default.yaml's model_variant stayed 'champion' throughout. No champion weights, active artifacts, prediction artifacts, or data/evaluation/ contents were overwritten. The served weekend forecast (Belgian GP, live as of 2026-07-19) was never touched by any research run.

- `config/production_config.json` sha256: `c690aa54e054f05a65f7ce565f0c195533723beaa21951ec63ac9daf4fbb96e1`
- byte-identical to expected: **True**
- `config/default.yaml`: `model_variant: champion`

## Full variant status (see v2 report for the full race-view/decomposition tables, unchanged)

- `q0_driver_state`: scored (tag='')
- `q0_driver_state__fp_hisim2`: scored (tag='fp_hisim2')
- `q1_qualifying_practice`: refused -- walk-forward replay produced no eligible scored events
- `q1_qualifying_practice__q1_retro`: scored (tag='q1_retro')
- `q1_qualifying_practice__q1_track_classes`: refused -- walk-forward replay produced no eligible scored events
- `q1_qualifying_practice__relaxed_gate`: refused -- walk-forward replay produced no eligible scored events
- `r0_long_run`: scored (tag='')
- `r0_long_run__fp_hisim2`: scored (tag='fp_hisim2')
- `r1_joint_grid`: scored (tag='')
- `r1_joint_grid__fp_hisim2`: scored (tag='fp_hisim2')
- `r1_r2_no_anchor`: refused -- prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=19, p5=11, p95=17)
- `r1_r2_source_anchor`: refused -- walk-forward replay produced no eligible scored events
- `r1_r2_source_anchor__relaxed_gate`: refused -- prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=9, p5=10, p95=15)
- `r2_no_anchor`: refused -- prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=19, p5=11, p95=17)
- `r2_source_anchor`: refused -- walk-forward replay produced no eligible scored events
- `r2_source_anchor__relaxed_gate`: refused -- prediction.finish_order is invalid: Grid position must lie inside p5-p95 interval (got position=9, p5=10, p95=15)


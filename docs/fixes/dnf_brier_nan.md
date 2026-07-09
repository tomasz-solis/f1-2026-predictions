# DNF Brier Skill NaN + Output Calibration

Status: implemented (shrinkage default off)
Date: 2026-07-06

Two related DNF-calibration problems surfaced by the 2026 live evaluation.

## 1. `brier_skill_score: NaN` in the evaluation report

`data/evaluation/2026_evaluation_report.json` carried a bare `NaN` for
`race_dnf_calibration.brier_skill_score` — invalid strict JSON and a useless
metric.

Root cause: an event with zero retirements has `base_rate = 0`, so
`baseline_brier = 0` and the per-event skill score is undefined (NaN) even
though its Brier score is finite. `_build_dnf_calibration_section` in
`scripts/generate_evaluation_report.py` filtered rows on `brier_score` only,
then weighted-averaged the per-event skill scores, so one zero-DNF event
poisoned the whole aggregate. `json.dump(..., allow_nan=True)` (the default)
then serialized the poison as a bare `NaN` literal.

Fix:

- the aggregate skill is now derived from the pooled components,
  `1 - weighted_brier / weighted_baseline`, and reported as `null` when the
  pooled baseline is zero (no DNFs anywhere);
- the per-event NaN in `compute_dnf_calibration` is kept — an undefined skill
  for a zero-DNF event is honest;
- the report is sanitized (`NaN`/`inf` → `null`) and written with
  `allow_nan=False` so any future non-finite value fails loudly instead of
  producing invalid JSON.

Regression tests: `tests/test_eval_metrics_split.py`
(`test_dnf_calibration_zero_dnf_event_scores_brier_but_not_skill`) and
`tests/test_generate_evaluation_report.py` (aggregate + sanitizer cases).

## 2. Raw `dnf_probability` overforecasts retirement risk

With the NaN fixed, the honest number is negative skill: the emitted
probabilities score a worse Brier than a naive base-rate forecast
(0.046 vs ~0.038 on 13 scored 2026 race events, 286 driver observations,
11 DNFs).

Probe: `scripts/probe_dnf_calibration.py` scores post-hoc shrinkage
transforms `p' = λ·p + (1-λ)·r` on the stored predictions, where `r` is the
DNF base rate over prior completed events only (leakage-safe expanding
window). Result (`data/model_diagnostics/2026/dnf_calibration_probe.md` when
run with the repo defaults):

| λ | Pooled Brier |
|---|---:|
| 0.00 (base rate only) | 0.0384 |
| **0.25** | **0.0370** |
| 0.50 | 0.0377 |
| 1.00 (current output) | 0.0457 |

Fix: an output-layer calibration in
`src/predictors/baseline/race/result_processing.py`
(`calibrated_dnf_probability`) with config knobs
`baseline_predictor.race.dnf_probability_shrinkage_lambda` (default 1.0 — no
behaviour change) and `dnf_probability_base_rate` (default 0.04). It shrinks
only the *reported* probability (dashboard "DNF Risk %" and the Brier
diagnostic); the Monte Carlo DNF sampling inputs are untouched, so finish
orders and intervals do not move.

Applied: `default.yaml` now ships `dnf_probability_shrinkage_lambda: 0.25`
(the schema Field default stays 1.0 as the conservative fallback for configs
that omit the key). This changes only the *reported* probability; because CI
runs on Ubuntu, the golden regression's exact match (macOS+3.11 only) is not
exercised there and the cross-environment check does not compare
`dnf_probability`, so the flip is CI-safe. Trade-off: the reported range
compresses to ~[0.03, 0.12] (relative, not absolute, risk); 0.5 keeps more
spread at nearly the same Brier and is the conservative alternative. The
sample is still small (11 DNFs), so revisit as races accumulate.

Out of scope / follow-up: recalibrating the simulation-input DNF rates
(`_compute_driver_dnf_probability` in
`src/predictors/baseline/race/preparation_flow.py`) would change finish-order
predictions and needs challenger-grade evidence per `docs/MODEL_PROMOTION.md`.
See also `LIMITATIONS.md` §10.

# Model Promotion Gates

This project treats model changes as challengers until they prove they help.
That matters because reset-year signals are easy to double count: a testing
team seed, residual model, and calibration layer can each look reasonable alone
but regress when stacked.

## Runtime Safety

Production defaults stay conservative:

- residual models are disabled by default,
- residual models are skipped when the active team seed is `testing_model`,
- stacking with `testing_model` requires an explicit ablation opt-in,
- conformal calibration is evaluated as interval calibration, not as a ranking fix.

## Promotion Gate

Implementation: `src/analysis/promotion_gate.py`.

A challenger must pass all of these before it is treated as stackable:

- combined race + qualifying central MAE improves enough to matter,
- race MAE does not regress beyond tolerance,
- qualifying MAE does not regress beyond tolerance,
- winner accuracy does not drop,
- top-3 accuracy does not drop beyond tolerance,
- race MAE is not worse on more weekends than it improves,
- qualifying MAE is not worse on more weekends than it improves.

The gate returns both a boolean and concrete block reasons. Reports should show
those reasons instead of reducing the result to a vague pass/fail.

## Movement Diagnostics

Implementation: `src/analysis/component_diagnostics.py`.

Movement diagnostics compare:

```text
champion prediction -> challenger prediction -> actual result
```

per race, session, and driver. The report counts:

- moved closer,
- moved farther,
- unchanged,
- MAE before,
- MAE after,
- mean movement,
- mean reported residual or learned adjustment when available.

This is especially useful for residual models. If a residual model improves one
mean metric but moves most drivers farther from actual positions, it should not
be promoted.

## Adaptive Learning Gate

Implementation: `src/systems/systematic_learning.py`.

The learner updates only from usable actual outcomes. It skips:

- retrospective checkpoint reconstructions,
- duplicate run IDs,
- records with no actual results,
- records with too few overlapping drivers between prediction and actuals.

Skipped partial records do not mark the run ID as processed. If a complete
actual payload arrives later for the same run, it can still train the learner.

## Research Workflow

1. Run the champion and challenger with isolated data roots.
2. Generate component ablations with `scripts/evaluate_testing_team_seed_model.py`.
3. Read the promotion gate and movement diagnostics together.
4. Keep generated reports and model artifacts out of ordinary code commits.
5. Promote only the smallest component stack that passes across holdouts and the live slice.

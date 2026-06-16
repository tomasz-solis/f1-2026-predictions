# Model Promotion Gates

This project treats model changes as challengers until they prove they help.
That matters because reset-year signals are easy to double count: a testing
team seed, residual model, and calibration layer can each look reasonable alone
but regress when stacked.

## Current Release Posture

Active model version: `2.3`.

`2.3` keeps the existing champion predictor in production and adds
target-specific shadow challengers. The challenger outputs are recorded for
audit; they are not dashboard-facing promotion decisions.

Targets are evaluated separately:

- `main_qualifying`
- `grand_prix_race`
- `sprint_qualifying`
- `sprint_race`

The current rule is to revisit promotion after races 8, 9, and 10. A challenger
may improve one target without being promoted to every target.

## Runtime Safety

Production defaults stay conservative:

- residual models are disabled by default,
- residual models are skipped when the active team seed is `testing_model`,
- stacking with `testing_model` requires an explicit ablation opt-in,
- conformal calibration is evaluated as interval calibration, not as a ranking fix.
- shadow challengers are saved as diagnostics and must not overwrite champion
  predictions without a promotion decision.

## Production Readiness Gate

Implementation: `scripts/generate_evaluation_report.py`.

The production gate writes `production_gate` into
`data/evaluation/2026_evaluation_report.json` and mirrors the result in
`docs/MODEL_CALIBRATION.md`.

The gate requires:

- a fresh evaluation report after the latest completed race weekend,
- at least 5 scored completed race weekends,
- positive qualifying MAE improvement over previous-race naive baseline,
- positive race MAE improvement over previous-race naive baseline,
- empirical qualifying interval coverage near the nominal 90% band,
- no unresolved high-miss systematic-bias bucket.

Run it with:

```bash
make evaluation-gate
```

Candidate and shadow diagnostics:

```bash
make candidate-audit
make shadow-challenger-audit
```

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

## Shadow Challenger Workflow

Implementation:

- `src/models/shadow_challenger.py`
- `scripts/audit_shadow_challengers.py`
- `scripts/audit_model_candidates.py`

Shadow challengers must use only prior completed actuals and current saved
champion predictions. Same-race actuals are leakage and must be excluded.

The audit reports:

- target-specific champion vs challenger MAE,
- number of comparable scored events,
- checkpoint MAE decay,
- best candidate family by target/session type.

## Research Workflow

1. Run the champion and challenger with isolated data roots.
2. Generate component ablations with `scripts/evaluate_testing_team_seed_model.py`.
3. Read the promotion gate and movement diagnostics together.
4. Read the model candidate and shadow challenger audits.
5. Keep exploratory artifacts out of ordinary code commits unless they are part
   of the release evidence.
6. Promote only the smallest component stack that passes across holdouts and the live slice.

# Next Fixes

This plan starts from the latest guarded testing-seed result:

- Old full challenger was hurt by stacking testing-derived team seeds with residual models.
- Guarded challenger removes that harmful stack.
- Guarded 2022 replay is slightly positive, but not strongly better.
- Guarded 2026 live slice is roughly neutral.
- Holdouts still do not pass, so this is not a promotion-ready model. It is a safer model-composition rule and a better research method.

The goal is to turn this into a real model improvement without breaking the online app.

## 1. Commit Only The Safe Guardrail

Commit the smallest production-safe change first.

What to include:

- Preserve team seed metadata from the loaded car-characteristics payload.
- Detect when the active team seed is `testing_model`.
- Skip qualifying residuals when `testing_model` is active unless explicitly opted in.
- Skip race residuals when `testing_model` is active unless explicitly opted in.
- Keep the opt-in flags defaulted to `false`.
- Keep tests that prove both the default skip and the explicit opt-in path.

What not to include:

- Generated reports.
- Model artifacts.
- Backtest result JSON files.
- New experimental priors.
- Full research output directories.
- Any unrelated app, dashboard, or data-generation changes.

Suggested commit label:

```text
Guard residual models from testing-seed stack
```

Why this is safe:

- It does not enable a new model by default.
- It prevents a measured bad interaction.
- It preserves the ability to run the old stack deliberately in ablations.
- It reduces risk for the online app because residual artifacts cannot silently combine with the testing seed.

Validation before commit:

```bash
uv run pytest tests/test_residual_models.py tests/test_learning_config_paths.py
uv run ruff check config/default.yaml src/utils/config_schema.py src/predictors/baseline/data_mixin.py src/predictors/baseline/qualifying_mixin.py src/predictors/baseline/race/prediction_mixin.py tests/test_residual_models.py
```

Manual staging warning:

Use `git add -p`. Do not stage whole files unless the diff contains only the guardrail work.

```bash
git add -p config/default.yaml
git add -p src/utils/config_schema.py
git add -p src/predictors/baseline/data_mixin.py
git add -p src/predictors/baseline/qualifying_mixin.py
git add -p src/predictors/baseline/race/prediction_mixin.py
git add -p tests/test_residual_models.py
git diff --cached
```

## 2. Keep The Guarded Challenger As A Baseline, Not A Promotion

Use the guarded run as the new comparison point for future research, but do not call it a proven model upgrade.

Current evidence:

- 2022 guarded challenger:
  - Race MAE improvement: `+0.019`
  - Qualifying MAE improvement: `+0.038`
  - Top-3 delta: `-1.59 pp`
  - Winner delta: `0.0 pp`
- 2026 guarded challenger:
  - Race MAE improvement: `0.000`
  - Qualifying MAE improvement: `-0.030`
  - Top-3 delta: `0.0 pp`
  - Winner delta: `0.0 pp`

Interpretation:

- This removes the large regression from the old full challenger.
- It does not yet create a strong predictor.
- It should become the safety baseline for reset-year experiments.

Next command to keep around:

```bash
uv run python scripts/evaluate_testing_team_seed_model.py \
  --output-dir reports/reset_guarded_testing_seed \
  --reuse-existing-artifacts \
  --season-year 2022 \
  --live-year 2026 \
  --live-max-races 3
```

## 3. Add A Promotion Gate For Component Stacking

Do not let new layers stack automatically just because they exist.

Add a simple rule:

- A component can run alone for research.
- A component can stack with another component only after the combined stack passes an ablation gate.
- A component that improves one headline metric but hurts central MAE should stay experimental.

Suggested stack gate:

- Race MAE improvement must be positive or neutral within `0.02`.
- Qualifying MAE improvement must be positive or neutral within `0.02`.
- Winner accuracy must not drop.
- Top-3 accuracy must not drop by more than `2 pp`.
- Race worse/better count should not show broad degradation.
- Qualifying worse/better count should not show broad degradation.

Apply this to:

- `testing_seed + qualifying_residual`
- `testing_seed + race_residual`
- `testing_seed + conformal`
- `testing_seed + both residuals`
- `full_challenger`

Expected outcome from current data:

- `testing_seed_only`: keep as a guarded experiment.
- `qualifying_residual_only`: do not promote.
- `race_residual_only`: do not promote.
- `conformal_only`: keep as calibration-only research.
- `testing_seed_plus_residuals`: block by default.
- `full_challenger`: block by default.

## 4. Rework The Qualifying Residual Model

The qualifying residual model is the clearest failed component.

Evidence:

- 2022 qualifying residual only:
  - Race MAE delta: `-0.210`
  - Qualifying MAE delta: `-0.357`
  - Winner delta: `-28.57 pp`
  - Qualifying worse/better count: `16/4`
- 2026 qualifying residual only:
  - Race MAE delta: `+0.121`
  - Qualifying MAE delta: `-0.455`
  - Qualifying worse/better count: `2/0`

Do not tune this lightly. The model is learning a correction that looks useful in places but damages the actual qualifying order.

Step-by-step:

1. Export signed qualifying residual adjustments by race, team, driver, and data regime.
2. Compare adjustment direction against the actual error direction.
3. Count how often the residual model moves drivers closer versus farther from the result.
4. Split results by `testing_fallback`, `practice_backed`, and `checkpoint_backed`.
5. Split by early-season races versus later races.
6. Check whether the model overcorrects high-ranked baseline positions.
7. Check whether reset-year teams and rookies receive unstable adjustments.
8. Add a report table with:
   - mean adjustment,
   - mean absolute adjustment,
   - closer count,
   - farther count,
   - unchanged count,
   - MAE before,
   - MAE after.
9. Only after that, try reducing the clip from `2.0` to `0.5` or `1.0`.
10. Re-run ablation before keeping any change.

Promotion rule for qualifying residuals:

- Must improve 2022 full replay qualifying MAE by at least `0.10`.
- Must not hurt race MAE by more than `0.02`.
- Must not lower winner accuracy.
- Must improve or preserve qualifying worse/better count.
- Must not regress live 2026 qualifying MAE.

## 5. Rework Race Residuals Only After Qualifying Is Stable

Race residuals are not the main problem, but they are not clearly useful either.

Evidence:

- 2022 race residual only:
  - Race MAE delta: roughly `0.000`
  - Qualifying MAE delta: `0.000`
  - Race worse/better count: `9/9`
- 2026 race residual only:
  - Race MAE delta: `-0.030`
  - Race worse/better count: `1/1`

Interpretation:

- This is noise-level behavior.
- It should not be promoted.
- It should not be stacked with reset-year testing seed by default.

Step-by-step:

1. Keep race residual disabled by default.
2. Add a diagnostic report for race residual adjustments.
3. Compare race-advantage movement against actual positions gained.
4. Split by predicted-grid versus actual-grid runs.
5. Split by overtaking difficulty and tire stress.
6. Check whether the model is merely adding noise around already reasonable race simulations.
7. Try smaller `positions_to_race_advantage_scale` values only after diagnostics show useful directionality.
8. Re-run ablation.

Promotion rule for race residuals:

- Must improve race MAE by at least `0.05` on 2022 full replay.
- Must not hurt qualifying-derived race flow.
- Must not reduce winner accuracy.
- Must improve race worse/better count.
- Must remain neutral or better on live 2026.

## 6. Keep Conformal Calibration Separate From Ranking Quality

Conformal calibration should improve intervals, not be treated as a ranking fix.

Current evidence:

- 2022 conformal only:
  - Race MAE delta: `-0.005`
  - Qualifying MAE delta: `0.000`
- 2026 conformal only:
  - Race MAE delta: `+0.030`
  - Qualifying MAE delta: `0.000`

Interpretation:

- The ranking impact is basically neutral.
- It may still be useful for uncertainty, but it should be evaluated with coverage and width, not central MAE.

Step-by-step:

1. Keep conformal disabled by default unless the product view needs calibrated intervals.
2. Evaluate conformal on:
   - empirical coverage,
   - interval width,
   - miss distance,
   - coverage by regime.
3. Do not promote it based on race MAE.
4. Require coverage improvement without absurd interval widening.
5. Keep conformal changes separate from residual and seed changes.

Promotion rule for conformal:

- Coverage should move toward `90%`.
- Width should remain explainable.
- Average miss distance should not increase materially.
- Central ranking metrics should stay neutral.

## 7. Improve Testing Seed Without Overclaiming

Testing seed is the most promising piece, but the evidence is weak.

Current evidence:

- 2022 testing seed only:
  - Race MAE delta: `+0.019`
  - Qualifying MAE delta: `+0.038`
  - Top-3 delta: `-1.59 pp`
  - Winner delta: `0.0 pp`
- 2026 testing seed only:
  - Race MAE delta: `0.000`
  - Qualifying MAE delta: `-0.030`
  - Top-3 delta: `0.0 pp`
  - Winner delta: `0.0 pp`

Interpretation:

- The testing seed is not harmful on the latest guarded evidence.
- It is not yet a strong upgrade.
- It should be improved through better team-prior calibration, not by stacking residuals.

Step-by-step:

1. Add a team-level report comparing champion prior versus testing seed values.
2. Highlight teams with the largest preseason delta.
3. For each large delta, show:
   - raw testing model score,
   - champion prior score,
   - bounded gap,
   - delta multiplier,
   - applied delta,
   - uncertainty.
4. Compare those deltas against actual first-three-race team strength.
5. Identify whether the seed is too conservative or too aggressive.
6. Tune only the bounded-delta logic, not downstream residuals.
7. Re-run `testing_seed_only` ablation.

Promotion rule for testing seed:

- Must beat champion on 2022 full replay by more than noise.
- Must remain neutral or better on live 2026.
- Must not reduce winner accuracy.
- Must not materially hurt top-3 accuracy.
- Must show team-level deltas that make domain sense.

## 8. Re-run Holdouts With The Guarded Stack

The current top-level holdout rows in the guarded report still came from the earlier holdout summary. That is useful continuity, but it means the holdout story is not fully refreshed around the new guardrail.

Step-by-step:

1. Run fresh holdouts without `--reuse-existing-holdouts`.
2. Keep `--live-max-races 3` for live 2026.
3. Compare old and new holdout rows.
4. Confirm whether the guardrail improves holdouts or only season/live challenger runs.

Command:

```bash
uv run python scripts/evaluate_testing_team_seed_model.py \
  --output-dir reports/reset_guarded_testing_seed_fresh_holdouts \
  --reuse-existing-artifacts \
  --season-year 2022 \
  --live-year 2026 \
  --live-max-races 3
```

Inspect:

```bash
cat reports/reset_guarded_testing_seed_fresh_holdouts/summary.md
```

Decision:

- If holdouts still fail, do not promote testing seed as a default.
- If holdouts improve but live stays neutral, keep it as a candidate.
- If both improve, then consider a staged promotion.

## 9. Add A Compact Research Decision Record

The project needs a short decision artifact for this work, separate from raw reports.

Create a markdown file under `reports/` or `docs/` with:

- Question tested.
- Baseline.
- Challenger.
- Ablation variants.
- Key results.
- Decision.
- What remains open.

Suggested title:

```text
Testing Seed Residual Stack Decision
```

Core conclusion:

```text
Do not stack residual models with testing-derived reset-year team seeds by default.
Testing seed alone is safer and roughly neutral/slightly positive.
Residual stacking caused broad qualifying degradation and live 2026 regression.
```

## 10. Protect The Online App

Before anything reaches the online app, verify that the runtime default remains conservative.

Checklist:

- `qualifying_residual_model.enabled` is `false`.
- `race_residual_model.enabled` is `false`.
- `allow_with_testing_seed` is `false` for both residual configs.
- No generated `data/processed/model_artifacts` files are committed accidentally.
- No research report directories are committed accidentally.
- No local backtest output JSON is committed accidentally.
- Existing online app prediction path still starts with the same active production data.

Run:

```bash
git diff --cached --name-only
git diff --cached
```

If staged files include generated reports or artifacts, unstage them.

## 11. Future Model Work

Once the guardrail is committed, work on actual predictive lift in this order.

1. Fresh guarded holdouts.
2. Team-seed diagnostics.
3. Qualifying residual diagnostic report.
4. Smaller qualifying residual clipping experiment.
5. Race residual diagnostic report.
6. Conformal coverage-only review.
7. Testing-seed bounded-delta tuning.
8. Full reset-style benchmark rerun.

Do not skip straight to a full challenger promotion. The ablation proved that the model can look plausible while the component stack is wrong.

## 12. Success Criteria

A future model upgrade should meet all of these:

- Improves 2022 full replay race MAE by at least `0.05`.
- Improves 2022 full replay qualifying MAE by at least `0.05`.
- Does not hurt winner accuracy.
- Does not hurt top-3 accuracy by more than `2 pp`.
- Is neutral or better on live 2026 first-three-race slice.
- Does not fail all reset holdouts.
- Beats or narrows the gap to the naive previous-race baseline on overlap metrics.
- Has a clear ablation explanation showing which component caused the gain.

If a change cannot meet those criteria, keep it as research.

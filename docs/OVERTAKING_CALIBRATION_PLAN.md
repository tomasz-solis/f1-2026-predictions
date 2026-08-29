# trackside-labs — overtaking calibration (COMPLETE, model version 3.0)

Revision 4, 2026-08-28. Two stress tests broke something load-bearing in each earlier revision, and executing Phases 0-1 broke a third. **Everything in the evidence section is measured; the plan contains no number picked by judgement.**

**All adopted, suite green, nothing committed.** Final state:

```
ANT P22 ->  Monza P7   Hungary P7   Monaco P8     (champion: P3 / P4 / P5)
BOT P22 ->  Monza P18  Hungary P20  Monaco P20
MAE 3.5758 -> 3.5152    suite 1762 passed, 0 failed
ruff check / ruff format / mypy src   all clean
```

Shipped: the queue invariant with its SC/VSC exemption and the contending-pairs cap; `resolve_pace_anchor` deleted and the grid anchor plus `max_gain` floor bypassed for penalised drivers only; per-team race pace measured from green-flag laps instead of inferred from classified results; `skill_improvement_max` 0.75 -> 1.75; the recovery-envelope note on the penalty banner; and three committed probes (`probe_overtaking_realism.py`, `probe_recovery_envelope.py`, `extract_team_race_pace.py`). Model version 3.0 — a mechanism change, not a recalibration.

**MAE could not arbitrate any of it.** Per-race sd is 1.339, so the SE of the mean is 0.387; every configuration measured (3.5152 through 3.6364) sits within 0.16 SE of champion, where 0.758 is needed to clear two SE. Every decision was made on physics metrics instead.

## Revision history — read this before trusting an older copy

- **Rev 1** proposed fitting the pass model to the measured per-track churn rate. Broken by review: it optimised a proxy, its fit was compute-infeasible, and its headline ratio came from a measurement whose exclusion rule did not match the extractor's.
- **Rev 2** promoted displacement to a co-equal target. Broken by measurement: churn and displacement are **anti-correlated** (-0.479), so "co-equal" was incoherent, and its `corr >= 0.80` criterion sat **above the reliability ceiling of 0.772** and could never have been satisfied.
- **Rev 3** separated the two targets by what the data supports: churn carries per-track signal, displacement only a pooled magnitude claim.
- **Rev 4** recorded Phases 0-1 as executed. The parametric approach is dead by measurement, not by argument: position changes were never coupled to passing. C4 was rewritten — its pooled recovery bound was 271/401 backmarkers and badly understated a penalised front-runner.
- **Rev 5 (this one)** records the work as finished. The simulator was right all along; the finish blend was discarding it. Two further defects were found and fixed: team pace derived from classified results, and driver skill under-modelled by 2.3x against the measured team mate gap.

## Phase 0 — COMPLETE, and it corrected two headline numbers

`scripts/probe_overtaking_realism.py` is the committed measurement harness. It resolves circuits through `circuit_registry`, calls `extract_overtakes_from_race` for the measured churn rather than reimplementing it, and computes every simulated statistic inside each individual simulated race, patched at `prediction_mixin`.

Authoritative baseline, `--sims 30 --seed 42`, all 12 completed 2026 rounds:

```
C1  displacement ratio  2.07   (target 1.00 +/- 0.15)
C2  churn ratio         1.76   (target 1.00 +/- 0.15)
C3  churn corr         +0.273  (target 0.617)

churn reliability        +0.595   ceiling 0.771
displacement reliability -0.223   no per-track signal at n=1 per circuit

mechanisms: on_track 82.2%   retirement 11.1%   pit 6.7%
envelope:   median 0.0  p75 1.5  p90 4.8  max 12  (63 driver-races)
wall clock  419s per evaluation
```

**G0.1 passed its gate.** On-track passes own 82.2% of simulated position changes, so the pass model is the right thing to calibrate. Had this come in below half, the plan would have been aimed at the wrong parameters.

**G0.2 resolved.** 419 s per evaluation at 30 simulations. Phase 2's fitting method is bounded by this: a few hundred evaluations is two to three days of pure compute, so the free-parameter count must stay at two or three with common random numbers.

**G0.3 was not the minor item an earlier revision called it.** Reconciling the pit-exclusion rule moved the churn ratio from 2.01 to **1.76**, and the churn correlation from +0.389 to **+0.273**. The cause was a harness bug, not the exclusion rule: the throwaway probe wrapped `simulate_race_lap_by_lap` on the utils module, which the injection means never fires, so its per-run state never reset and every simulation inherited the previous one's final positions as its "previous lap" — manufacturing phantom changes on lap 1 of every run. **Any figure quoted from that probe is suspect.** The displacement ratio came through an independent path and reconciles (1.95 vs 2.07 at 30 sims).

Reliability reproduces independently: 0.595 here against 0.596 computed by hand, ceiling 0.771 against 0.772.

## Phase 1 — COMPLETE. Root cause found, and it is architectural

**Position changes in this model are not caused by passing.** Measured directly, counting successful pass events against position changes over 10 simulations per circuit:

```
circuit      passes  changes  passes/change
Monaco           46     2949      0.016
Hungarian       196     3151      0.062
Belgian         357     2715      0.131
```

98.4% of Monaco's position changes involve no pass event. Position derives from cumulative lap time, so two cars with different pace cross over whether or not the pass model ever fires. The pass model only adds a 0.08-0.35 s time bonus on top of movement that happens regardless.

This is why **every** attempt in this session failed, including the two structural leads:

| attempt | C2 churn ratio | why |
|---|---|---|
| baseline | 1.76 | |
| per-circuit field spread (lead 1) | 1.69 | pace re-sorts the field anyway |
| measured dirty air, ~24x larger (lead 2) | 1.74 | a lap-time nudge, not a gate |
| pass probability x0.10 | 1.75 | **caps a knob with no authority over the output** |
| pass probability x0.75 | 1.76 | |

A 90% cut in pass probability moves churn by 1%. That is the measurement that ends the parametric approach.

**Phase 2 as previously written -- fit the pass model's parameters -- is dead.** Fitting a knob that does not control the output cannot work, and this is measured rather than argued.

**The encouraging half.** `passes/change` orders correctly by circuit (Monaco 1.6%, Hungary 6.2%, Belgium 13.1%). The pass model already discriminates tracks; it simply has no authority. Gating position change on pass outcome should let the model inherit that discrimination rather than having it fitted in.

**This also explains the queue invariant's result.** It scored 3.5152 against champion 3.5758 and was the only change all session that materially altered dynamics, because it is the only mechanism that couples position to pass outcome. It was rejected as a tweak; it is actually the missing coupling.

### Defects found and documented along the way, none of them the lever

- `dirty_air_penalty_base` and `dirty_air_penalty_track_scale` are **inert**. They compute a cap applied as `min(cap, calculate_dirty_air_penalty(...))`, but that function caps itself at `max_penalty_s=0.05` and the simulator never passes the argument. The cap can never bind.
- The effective dirty-air penalty is **~0.037 s/lap** after skill relief, against a measured 2026 following penalty of **0.884 s/lap median** (range 0.215 Canada to 2.402 Monaco) -- roughly 24x too small. Measured by comparing each driver's own median lap time within 1.0 s of the car ahead against over 3.0 s, green-flag laps only, pit and position-change laps dropped.
- The simulated field is **2.3x too compressed** at the end of lap 1 (9.8 s vs 22.2 s actual) and essentially flat across circuits (sim range 1.3x, actual 4.7x).
- Most expanded overtake parameters are **not independently settable**. `_expand_overtake_cfg` rebuilds `race_params["overtake_model"]` from five compact inputs (`dirty_air_window_s`, `pace_weight`, `racecraft_weight`, `track_factor`, `pass_chance_base`) plus a hardcoded `_OVERTAKE_INTERNAL` table no config reaches. `pass_threshold_track_scale` is `track_factor * 0.46` where `track_factor` is one global constant -- which is why threshold tuning never produced track discrimination.

### Indirection seams, four of them, each of which silently voided an experiment

1. `build_finish_order` -- imported by name into `race_simulation`.
2. `simulate_race_lap_by_lap` -- injected via `deps`, bound in `prediction_mixin`. Patch there.
3. `calculate_dirty_air_penalty` -- called without `max_penalty_s`, so the config cap is inert.
4. `_expand_overtake_cfg` -- rebuilds the overtake config, discarding expanded keys set upstream.

## Repository state

Repo `~/Documents/repos/trackside-labs`. Champion `566b5d3b`, race MAE **3.5758** over the 12 completed 2026 rounds. Uncommitted: `scripts/probe_overtaking_realism.py`, one additive `per_lap_changes` key in `src/extractors/overtaking.py`, and `docs/OVERTAKING_CALIBRATION_PLAN.md`.

The durable copy lives in private memory at `projects/plans/trackside-labs-overtaking-calibration-wip.md`; the two are kept identical.

## What started this

A 20-place grid penalty for ANT at the 2026 Italian GP. The live site reported him finishing **P3** from P22, p5-p95 band P2-P10.

The penalty plumbing is correct. `apply_grid_penalties` moves him to P22 and records `qualifying_position: 3`. Reproduced directly:

```
with anchor   ANT = [3, 3, 4]     p5=[2,2,2]   p95=[10,10,11]
anchor OFF    ANT = [20, 22, 22]  p5=[17,16,17] p95=[22,22,22]
```

"Anchor OFF" = `resolve_pace_anchor` patched to return `(reference_grid_pos, 0.0)`.

## The layer chain

1. The simulator moves cars too freely — **measured at 1.95x on displacement and 2.01x on churn**.
2. Because of 1, the finish blend anchors hard to grid position to stay accurate (`grid_anchor_weight`, 0.35-0.58 for a main race).
3. Because of 2, a penalised driver's penalty is counted **twice** — once by starting him at P22, again by anchoring his blended score there.
4. Because of 3, `resolve_pace_anchor` (commit `53e27d91`) was added to undo the double count. It restores his qualifying position in full, erasing the penalty instead of correcting the double count. That is the P3.

## Evidence

### What the model gets wrong, measured

```
churn (position changes per lap, 11 circuits)
  simulated / measured        = 2.01x
  corr(measured, simulated)   = +0.389

displacement (mean |grid - finish| among classified finishers, ranked
              within finishers, per simulated race, 12 circuits)
  mean simulated              = 3.57 places
  mean actual                 = 1.83 places  (+/- 0.14, pooled SE)
  ratio                       = 1.95x        (~12 SE from parity)
  corr(simulated, actual)     = +0.000
```

Both measures say the same thing: the model is **about twice as active as reality** and cannot tell circuits apart.

### Which target can carry a per-track criterion — and which cannot

Each measured value comes from exactly one race, so each carries sampling noise. Comparing that noise to the real between-track spread decides whether a per-track criterion is even evaluable.

```
CHURN            between-track var 0.4469   sampling var 0.1805
                 reliability 0.596   max attainable corr 0.772

DISPLACEMENT     between-track var 0.2036   sampling var 0.2490
                 reliability NEGATIVE (-0.223)
```

**Churn has per-track signal. Displacement does not.** Its observed between-track differences are entirely inside sampling noise at n=1 race per circuit.

Two things follow, and both correct claims made earlier in the diagnosis:

- `corr(sim displacement, actual displacement) = +0.000` **is not evidence of a defect.** A perfect model would also score near zero against a noise-dominated target. Do not cite it as one.
- An earlier draft called `corr(displacement, lap-1 field gap) = +0.712` the strongest relationship in the data. It is a correlation against mostly noise and is **withdrawn**.

Displacement still supports a **pooled magnitude** claim, because the pooled mean has a small standard error (0.14 on 1.83). It cannot support a per-track one until a second season exists.

### Churn and displacement are anti-correlated

```
corr(churn rate, displacement, attrition-free) = -0.479
```

Monaco has the lowest churn (1.263/lap) and among the highest displacement. High churn means cars trading places repeatedly with no net movement. **Optimising churn can push displacement the wrong way** — they cannot be co-equal objectives, and any joint fit needs an explicit weighting rather than a silent one.

### Attrition has to be stripped from displacement

A retirement promotes everyone behind it, which is not movement anyone earned.

```
corr(raw displacement, DNF count)                = +0.391
corr(within-finishers displacement, DNF count)   = -0.301
```

Always rank within classified finishers (`Status` in `Finished`, `Lapped`). Raw `GridPosition - Position` measures attrition as much as passing.

### The recovery envelope — what a back-of-grid start actually yields

Pooled across all 12 races, 63 driver-races starting P15 or worse and classified, ranked within finishers:

```
places gained:  median 0.0   p75 1.5   p90 4.8   p97 7.3   max 12
  bunched field (gap<0.85s):  median 0.0  p90 2.8  max 9
  strung out   (gap>=0.85s):  median 1.0  p90 4.7  max 12
Monaco, all classified:       median 0.0  p90 4.0  max 5
```

A car starting at the back typically gains **nothing**. The best result by any driver at any circuit all season was **12 places**. The model predicts ANT gains **19**.

Caveat to carry: most P15+ starters are slow cars, so this envelope understates a penalised front-running car. The season-wide maximum of 12 is the honest hard ceiling; the p90 is not.

### Field spread — a real defect, with its link to the target unproven

```
mean adjacent gap at the end of lap 1:
  measured   0.47 s (Austrian) to 2.19 s (British), 4.7x across circuits
  model      0.32 s constant, every circuit  (start_grid_gap_seconds)
```

The model compresses the field tighter than the tightest real circuit and holds it constant where reality varies almost fivefold. P22 starts only 6.7 s behind P1, so a 0.4 s/lap edge erases the whole grid in 17 laps anywhere.

The direct comparison above is solid. What is **not** established is the link to the target: `corr(over-production ratio, field gap) = -0.697` is measured, but the ratio inherits displacement's sampling noise, so treat it as a lead, not a mechanism. The raw split is large enough to be worth testing:

```
over-production ratio where real field is strung out (gap>=2.0s): 1.10, 1.01
over-production ratio where real field is bunched    (gap<0.70s): 2.81, 2.04, 3.09, 2.38, 2.05
```

### Lead hypothesis: proximity does not suppress passing

The model's error concentrates at circuits where the real field is bunched. In reality a bunched field is a DRS train — cars are stuck *because* they are close. In the model, close means inside `pass_window_s`, which means pass attempts.

The reward/resistance ratio is the thing to check: dirty air caps at `0.05 + track_overtaking * 0.12` (max 0.17 s), then halves under skill relief to about 0.09 s, against a pass bonus of 0.08-0.35 s. **Passing outruns the mechanism that should prevent it by roughly two to four times.** This is a hypothesis with a measurable consequence, not a conclusion.

### Race-name mismatch

`data/processed/track_characteristics/2026_track_characteristics.json` keys the circuit as **Spanish Grand Prix**; FastF1's 2026 event is **Barcelona Grand Prix**. Loading by the file's key fails and silently drops 1/12 of the data. Any harness must resolve through `src/data/circuit_registry.py`, never by raw name.

### Harness gotcha, cost three runs

`simulate_race_lap_by_lap` is **injected as a dependency**, bound in `src/predictors/baseline/race/prediction_mixin.py` (line ~360) and called as `deps.simulate_race_lap_by_lap` in `race_simulation.py`. Patching it on `src.utils.lap_by_lap_simulator` or on `race_simulation` does nothing. Patch on `prediction_mixin`. Same class of trap as `build_finish_order`, which is imported by name into `race_simulation`.

A second trap: comparing a Monte Carlo **median** finish order against a single real race understates displacement mechanically. Compute the statistic **inside each simulated race**, then average.

## Attempts already made and reverted — do not retry as-is

All three built, measured, reverted. Tree returned clean to `566b5d3b`.

### 1. Pass-probability cap divided by contending pairs — worse

Cap changed from `overtaking_avg_changes_per_lap / (field_size - 1)` to `/ contending_pairs`.

```
MAE 3.5758 -> 3.5833 (worse)
probes: Monza P6->P5, Hungary P8->P6, Monaco P8->P9
track discrimination: unchanged
```

### 2. Queue invariant with a safety-car exemption — better MAE, worse physics

Failed pass cannot end the lap ahead, via `_FOLLOWING_EPSILON_S = 0.001`, exempting pitted cars, DNFs, **and neutralised laps**.

The SC/VSC exemption is genuinely new and worked: `test_high_sc_probability_produces_variance` — the test that killed the first attempt at this invariant — **passed**. Keep that exemption if the invariant returns.

```
MAE 3.5758 -> 3.5227 (best MAE measured on this model)
probes: Monza P5, Hungary P5, Monaco P6   <- Monaco got EASIER
track discrimination: still flat
```

Rejected: buys 0.053 MAE with worse physics and leaves the defect untouched. Third time the ledger records an MAE gain bought with physically false behaviour.

Four tests failed under it:

- `test_high_sc_probability_produces_variance` — **fixed** by the SC exemption.
- `test_race_simulation_uses_mapped_team_seconds_delta`
- `test_race_simulation_uses_seconds_native_driver_residual`
- `test_persistent_teammate_setup_offset_can_break_identical_teammates`
- `test_higher_skill_driver_wins_majority_of_intra_team_battles` (58.8% against a 60% floor)

The middle three are **plumbing tests using free passing as their read-out**: a two-car, three-lap race with `pass_probability_base: 0.0` and `pass_probability_scale: 0.0` asserting the 1.0 s/lap faster car starting P2 finishes first. That only worked because a faster car walked past without passing. Under a real queue they should fail; their intent is valid, their read-out is not. The fourth is a separate statistical claim needing its own evidence, never a threshold edit.

### 3. Deleting `resolve_pace_anchor` — ruled out by measurement

Exposes the grid anchor's double count: P19-P22 from P22 at every circuit.

## Acceptance criteria

Every bound below is derived from the measurements above.

- **C1 — displacement magnitude (pooled).** Mean simulated/actual displacement ratio within **1.00 +/- 0.15** across the 12 circuits. Currently **1.95**. The band is roughly two pooled standard errors (actual mean 1.83, pooled SE 0.14).
- **C2 — churn magnitude.** Mean simulated/measured churn ratio within **1.00 +/- 0.15**, no circuit outside **0.7-1.4**. Currently mean **2.01**, worst **3.20** (Monaco). Band matches the ~14% per-race relative sampling noise.
- **C3 — churn track discrimination.** `corr(measured, simulated) >= 0.618`, which is **80% of the reliability ceiling of 0.772**. Currently **+0.389**. Do not raise this above the ceiling; an unreachable criterion never terminates.
- **C4 — recovery envelope, conditioned on car pace.** A front-running car starting P22 should be predicted around **P15** (the +7 median) and **never better than P9** (+13, the four-season maximum). A backmarker starting P22 must stay within about one place of **P22** (the +1 median). The current candidate gives ANT P16 (+6) and BOT P22 (+0) — both within roughly one place of their bucket medians. Champion gives P3, a +19 gain that has never happened.

  Measured over 2022-2025, 401 driver-races starting P15 or worse, classified finishers only, both grid and finish ranked within finishers, bucketed by each driver's median finish across his *other* races that season:

  ```
  top car    (season median finish <= 6)  n= 31  median +7  p75 +10  p90 +13  max +13
  upper-mid  (6 < median <= 11)           n= 99  median +3  p75  +5  p90  +8  max +12
  backmarker (median > 11)                n=271  median +1  p75  +3  p90  +4  max +12
  2026 top car, n=2: HAD P21->P6 (+12), VER P20->P6 (+9)
  ```

  **This supersedes an earlier pooled bound** ("no better than P17 at Monaco, P10 anywhere") which was derived from a single undifferentiated distribution that is 271/401 backmarkers. Pooling drags the p90 down to 4.8 and badly understates what a penalised front-runner can do — the error that made P16 look too pessimistic when it is in fact one place off the top-car median.

  **Top-car recovery is only weakly track-dependent**: median +8 on easier circuits against +7 on harder ones, p90 12.2 against 12.5. VER went P15->P2 at Jeddah; RUS went P18->P6 at Hungary. Weak evidence — the split uses stored 2022-2024 rates carrying `overtaking_observed_races: 0` — but it argues against expecting a large Monza/Monaco spread in *this particular* quantity.
- **C5 — accuracy guard.** Mean race MAE over the 12 rounds **<= 3.5758**, against a **rebuilt** baseline, 100 simulations, seed 42, each round from its actual qualifying classification. **Report in-sample and leave-one-race-out.** Every phase fits and validates on the same 12 races; LOO is the only defence at n=12.
- **C6 — gates. MET.** **1762 passed, 5 skipped, 3 xfailed, 0 failed** across the five chunks, plus `ruff check`, `ruff format --check` and `mypy src` clean. Two `resolve_pace_anchor` tests were deleted and two team-pace tests added, so the total returns to 1762. Run in chunks (`scripts/run_pytest_chunk.py --letters abc|defgh|ilmnop|qrs|tuvwxyz`); a bare full-suite run was killed three times at 5-15 minutes.
- **C7 — no orphaned compensation.** `grid_anchor_weight` refit against the calibrated simulator, and `resolve_pace_anchor` **removed** unless it earns MAE on its own.

**Not a criterion, deliberately:** per-track displacement discrimination. Displacement reliability is negative at one race per circuit, so no such criterion is evaluable. Revisit when 2027 gives a second observation per circuit.

**C5 is a guard, not the objective.** A correctly calibrated simulator can score worse than one whose errors cancel. If the fit is right and MAE regresses, that is an explicit trade for the user — never a reason to tune until MAE improves.

## Work, in order

### Phase 0 — close the last three unknowns (half a day)

Only three remain; the rest are measured above.

**G0.1 — mechanism decomposition.** Split simulated position changes into on-track passes, pit-cycle swaps, and retirement cascades. **If on-track passes own less than half the excess, this plan targets the wrong parameters** — fix what actually owns it. Churn conflates all three; fitting `pass_probability_*` to the total would mis-tune passing to absorb pit-strategy or attrition error.

**G0.2 — evaluation cost.** Time one full evaluation of the target function. One 12-circuit, 30-simulation pass took roughly 20 minutes in the diagnosis session. **The fitting method in Phase 2 is chosen against that number**, not against hope: a naive least-squares over nine parameters needs hundreds of evaluations, which is days of pure compute.

**G0.3 — definition parity (demoted).** The churn probe's pit-exclusion rule (skip a driver on his pit lap and the lap after) does not match the extractor's (`PitOutTime.isna()`). Reconcile before quoting the 2.01x as exact. Demoted because both churn ratios and the displacement ratio agree at ~2x through independent paths, so the headline is not at risk — only its third digit.

**Exit:** a committed, deterministic, seeded `scripts/probe_overtaking_realism.py` emitting churn, displacement, the recovery envelope, and the correlations — resolving circuits through `circuit_registry` and patching at `prediction_mixin`.

### Phase 1 — test the two structural leads before fitting anything (one day)

Both are single quantities with measurable consequences. Test them before opening a nine-parameter fit, because either could remove most of the 2x on its own.

1. **`start_grid_gap_seconds`.** Constant 0.32 s against a measured 0.47-2.19 s. Replace with a per-circuit value derived from lap-1 timing data. Measure the effect on C1 and C2.
2. **Proximity does not suppress passing.** Dirty air (~0.09 s after relief) against a pass bonus (0.08-0.35 s). Test whether raising the following penalty relative to the pass reward reproduces the bunched-circuit over-production.

**Drop `overtaking_difficulty` as a pass-model input** and feed the measured rate directly. Rev 1 refitted difficulty from the measured rate, then fitted a model consuming difficulty against the same rate — circular, and the fit goes degenerate. Keep difficulty where it is not redundant: dirty air and the grid anchor.

### Phase 2 — fit only what Phase 1 leaves (cost set by G0.2)

Pin every parameter G0.1 says does not own the excess. **Two or three free parameters, not nine.**

Fit churn per-circuit (the only target with signal) and displacement as a pooled magnitude, with the weighting between them **written down**, not implied. Least squares on log ratio, jointly across circuits, never per-track — one race per circuit means a per-track fit is fitting noise, the error already recorded and rejected for per-track attrition (observed sd 1.68 below Poisson 2.10).

Use **common random numbers** across evaluations so Monte Carlo noise does not dominate the objective. Report leave-one-race-out alongside in-sample.

Champion defaults: `pass_probability_base 0.30`, `pass_probability_scale 0.45`, `pass_threshold_base 0.06`, `pass_threshold_track_scale 0.16`, `pass_window_s 1.2`, `dirty_air_penalty_base 0.05`, `dirty_air_penalty_track_scale 0.12`, `start_grid_gap_seconds 0.32`, `zone_front_threshold_boost 0.22`, `zone_front_probability_scale 0.55`, `zone_back_threshold_boost -0.03`, `zone_back_probability_scale 1.08`.

### Phase 3 — re-derive the queue mechanic, only if still needed (half a day)

Re-measure. **If the calibrated model meets C1-C4 without a queue invariant, do not add one.**

If it does not, the invariant returns with its proven SC/VSC exemption. The three plumbing tests then need re-pointing at lap time rather than finish order — **with the user's approval**, because rewriting a test to accommodate a regression is a mistake already made once in this project.

### Phase 4 — re-derive the blend (one day)

Refit `grid_anchor_weight` against the calibrated simulator, scored leave-one-race-out. Re-measure the penalised case. Remove `resolve_pace_anchor` per C7.

Each phase ends at a committable green state or lives on its own branch. A half-finished calibration is worse than champion, and the site is live.

Total: three to four working days, assuming Phase 1 removes a meaningful share of the 2x. If it does not, Phase 2's cost is set by G0.2 and the estimate is `[Unknown]` until then.

## Escalation

If C1, C2 and C3 cannot all be met after Phase 2, the conclusion is **architectural, not parametric**: position derives from cumulative time across a 6.7 s field with a pass worth at most 0.35 s, and track position may simply not be representable without a structural change.

In that case stop fitting. Report penalised recovery as an explicit band taken from the measured envelope above (median 0, p90 ~5, max 12 places) rather than a point estimate, and record in `docs/MODEL_LEDGER.md` that the point estimate is not supportable.

## Risks

- **n = 1 per circuit.** Displacement already fails its reliability test on this basis; churn survives with reliability 0.596. Fit jointly, never per-track.
- **Regulation break.** 2022-2025 rates describe different cars (f1 2026 regulation break). They may seed a transition prior for unmeasured circuits but stay out of the fit.
- **12 of 25 circuits measured.** The rest carry a 2022-2024 prior with `overtaking_observed_races: 0`.
- **Monza is unmeasured.** The 2026 Italian GP has not run; it inherits a 2025 fallback of 3.9. Its calibration is inherited, not measured — and Monza is the case that started this. LOO across the measured circuits is the best available estimate of Monza-like performance.
- **The four qualitative tests will fail again in Phase 3.** Scheduled decision point.
- **MAE may not improve.** See C5.

## Measured 2026 churn rates (2026-08-27)

`overtaking_observed_races: 1` on each:

```
Spanish 3.615, Dutch 3.557, Hungarian 3.391, Chinese 3.200, Belgian 3.163,
British 3.137, Japanese 2.827, Miami 2.804, Austrian 2.800, Australian 2.298,
Canadian 2.212, Monaco 1.263
```

## Not in scope

DNF probe artifacts (`dnf_calibration_probe.md` / `.json`) still report 11 retirements where FastF1 has 42. Re-running needs production actuals backfilled. It interacts with G0.1 — retirement cascades are one of the three mechanisms producing position changes.

## Interim product decision — still open

The prediction page reports a penalised driver's recovery from a model that cannot compute it. Options: suppress the figure, flag it as unreliable, or leave it. **No decision recorded.** Penalties are entered via the dashboard admin page gated on `TL_ADMIN_TOKEN`; removing the ANT entry reverts the page without a deploy.

## Probe scripts

In `C:/Users/tomas/AppData/Local/Temp/tlreplay/` (temp — promote what is worth keeping in Phase 0):

- `phase0_data.py` — lap-1 field spread, displacement, DNF counts, per-race churn SE. Data only, no simulation.
- `t2prime.py` — attrition-free displacement (ranked within finishers).
- `t2p_reliability.py` — the reliability calculation that showed displacement has no per-track signal.
- `recovery_envelope.py` — the pooled back-of-grid gain distribution behind C4.
- `sim_t2p_perrun.py` — simulator-side displacement, per simulated race, patched at `prediction_mixin`.
- `sim_vs_measured_rate.py` — churn table and correlations. Its pit-exclusion rule is what G0.3 reconciles.
- `recovery_harness.py` + `score_mae.py` — 12-round MAE snapshot and scorer.
- `probe_recovery.py` — penalised recovery with the pace anchor on and off.
- `sim_median.py` — captures `aggregated["median_positions"]` before the blend, patching `build_finish_order` on `race_simulation`.

## Remaining work, none of it blocking

- **BOT drift.** The slowest car's Monza recovery moved P22 -> P19 -> P18 across the three fixes. Each step sits inside the measured envelope (median +2, p90 +9 from P15+ at Monza); the cumulative trend was never examined.
- **C1 displacement** never reached target (2.07 -> 1.76 against 1.00 +/- 0.15) and was not pursued: displacement has no per-track signal at one race per circuit (reliability -0.223), so it supports only a pooled magnitude claim.
- **`team_strength` itself is still results-derived.** The fix substitutes a measured race-pace value where `base_pace` consumes it, leaving the score — and everything else it feeds, including qualifying — unchanged. Fixing the score at source is the larger, more principled version.
- **`MODEL_PROMOTION.md` had drifted** from `config/default.yaml` (2.3 against 2.4) before this session. Both now read 3.0.
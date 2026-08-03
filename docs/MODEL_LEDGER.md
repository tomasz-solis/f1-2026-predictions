# Model Ledger

What has been tried, how it was measured, and whether it helped.

`MODEL_PROMOTION.md` defines the gates a change must pass. This file is the
record of what actually went through them, kept so the current model's
reliability can be judged over time rather than re-argued from memory. Append to
it; do not rewrite past entries when a later result contradicts them — supersede
them and say so.

## How to read a verdict

| verdict | meaning |
|---|---|
| `adopted` | measured better, now part of champion |
| `worse` | measured, lost |
| `noise` | measured, difference indistinguishable from run-to-run variation |
| `never activated` | ran, but a runtime guard made it champion-identical — **untested, not neutral** |
| `refused` | could not produce a scored result at all |
| `open` | not yet measured |

`never activated` is the one that misleads. A variant that returns
champion-identical numbers looks harmless in a comparison table and is actually
a variant nobody has tested.

## The baseline problem

**Every result is only comparable to the champion it was measured against.**
On 2026-07-28 a bug fix moved champion qualifying MAE by 0.70 positions —
roughly ten times the largest challenger effect ever recorded here. Any
challenger scored before that date was scored against a champion that no longer
exists.

So every entry below records its baseline. When champion changes materially,
past challenger verdicts become stale rather than wrong, and re-baselining is
the only way to keep the table meaningful.

## Champion history

| date | change | measured effect | commit |
|---|---|---|---|
| 2026-07-28 | Centre `quali_rating_mu_s` within team when the qualifying driver list is built. It was carrying a team-level component on top of the team-strength term, so car pace was counted twice. | qualifying MAE **3.525 → 2.828**; mean per-driver \|bias\| **2.889 → 1.677**. HUL +6.11 → +1.67, ALB −5.78 → +0.11, GAS +4.33 → +0.33 | `93bfbeb0` |

Known residual after that change, measured the same way: SAI −5.67, LAW +6.44,
BOR +4.44, ALO −4.11, VER −4.11. These are team-strength errors (Williams
over-rated, RB under-rated), not driver-rating errors, and the centering fix
does not touch them.

## Measurement protocol

Results are only comparable if they were produced the same way. The walk-forward
entries below all used:

- **events** — `data/historical_replay/2026/event_catalog.json`, 9 events, of
  which 7 score (the two wet rounds are excluded by `dry_only`)
- **checkpoint** — `PRE`
- **seeds** — 3, from `DEFAULT_REPLAY_SEEDS`
- **simulations** — 20 qualifying, 20 race
- **command** —
  ```powershell
  uv run python scripts/run_challenger_research_walk_forward.py `
    --variants <variant> --qualifying-simulations 20 --race-simulations 20 --run-tag <tag>
  ```

Champion-side entries (the table above) were measured directly through
`predict_qualifying(..., practice_signal_mode="stored_profiles")` over all 9
catalog events, comparing predicted qualifying position against
`actual_qualifying_grid`.

**Before any re-run, move `data/historical_replay/2026/prediction_cache` aside.**
Its key (`_source_digest`) covers event data and simulation counts but *not*
code version, so a re-run will otherwise serve predictions computed by the old
model and silently score against a stale champion. Keep
`research_backend_state/` — that is model state built by the updater, unaffected
by prediction-side changes, and expensive to rebuild.

Use `--run-tag` so a follow-up never clobbers previous output.

## Challengers tested

All rows below were measured 2026-07-19/20 against the **pre-centering
champion**. Re-baselining against `93bfbeb0` has not been done — see "Blocked"
at the end for why.

| variant | thesis | verdict | evidence |
|---|---|---|---|
| `q0_driver_state` | richer driver-state term for qualifying | **worse** | quali grid MAE mean delta **+0.19**, 1 better / 5 worse |
| `q0_driver_state__baseline500` | q0, 500-sim baseline | **worse** | +0.24, 2 better / 13 worse |
| `q0_driver_state__fp_hisim2` | q0, higher FP sim count | **worse** | +0.23, 2 better / 7 worse |
| `q0_driver_state__pullcap025` | q0, cumulative pull cap 0.25 | **worse** | +0.24, 2 better / 13 worse |
| `q0_driver_state__pullcap035` | q0, cumulative pull cap 0.35 | **worse** | +0.24, 2 better / 13 worse |
| `r1_joint_grid` | sample race grid from the joint qualifying distribution instead of the marginal order | **noise** | end-to-end race MAE **−0.074**, but 4 better / 3 worse over 7 events |
| `r1_joint_grid__fp_hisim2` | as above, higher sim count | **noise** | 5 better / 4 worse, mean delta 0.000 |
| `q1_qualifying_practice` | practice one-lap pace → qualifying potential | **never activated** | manifest status `refused` ("no eligible scored events") because the research fit needs 4 prior dry same-class events and had 3; the `q1_retro` run that did produce output was champion-identical, disclosing `no_raw_practice_laps` |
| `r0_long_run` | practice long-run pace → race pace | **never activated** | 42 structural-identity flags, `missing_race_practice_evidence` |
| `r0_long_run__fp_hisim2` | as above | **never activated** | 60 structural-identity flags, `insufficient_field_evidence_coverage` |
| `r2_no_anchor`, `r1_r2_no_anchor` | grid-anchor variants | **refused** | `finish_order` invalid: a position fell outside its own p5–p95 interval |
| `r2_source_anchor`, `r1_r2_source_anchor` | grid-anchor variants | **refused** | no eligible scored events |

Across every run: 309 champion-vs-challenger metric pairs, 179 identical, 130
differing. **Nothing beat champion.**

Q0 is the clearest result — it lost under all four tunings, so it is not a knob
that needs turning.

### What this does and does not tell you

Q1 and R0 are the two variants that match the actual modeling thesis: one-lap
pace drives qualifying, long-run pace drives the race. Neither has ever run.
The replay harness feeds `practice_signal_mode="stored_profiles"`, so
`session_laps_by_type = {}` by construction and both variants hit a runtime
guard and return champion-identical output.

So the honest summary is not "practice-driven variants do not help". It is
**"the grid-plumbing variants were tested and lost; the practice-driven
variants have never been tested."**

`docs/RAW_LAPS_REPLAY_HANDOFF.md` is the fix and was already the stated priority
on 2026-07-19.

## Blocked

**Challenger work is shelved as of 2026-07-29.** ~~The tree stays untracked in
the worktree; no branch was made.~~ **Updated 2026-07-31: it is now on the branch
`shelved/challenger-research`** and no longer sits loose in the `master` working
tree.

> **The methodology is on `master`:** `docs/QUALIFYING_RACE_CHALLENGER.md` and
> `docs/RAW_LAPS_REPLAY_HANDOFF.md` were promoted here, since the reasoning
> outlives the code. Every path *they* cite resolves only on the branch.
>
> **The implementation is on `shelved/challenger-research`:**
> `scripts/run_challenger_research_walk_forward.py`, the
> `src/analysis/challenger_*` and `src/models/qualifying_practice_*` modules,
> their tests, and the human-readable reports under
> `data/model_diagnostics/2026/race_mae_investigation/`. Check the branch out to
> read or run any of it. Nothing in production imports them: a scan of all 409
> tracked Python files finds zero imports, and `master` passes its own suite
> without them.
>
> The generated `*_variant_comparison*.json` dumps behind those reports were
> **not** kept — 27,340 lines of machine output whose conclusions are already in
> this file. Regenerate them from the branch if a raw payload is ever needed.
>
> **The walk-forward artifacts under `data/historical_replay/2026/` are the one
> exception and are NOT on that branch.** They are 909 MB and gitignored
> (`.gitignore:41`), so they exist only on local disk, unversioned. They are the
> evidence behind every number above and there is no copy anywhere else: keep
> them, and do not assume a branch checkout restores them.

Re-running the three scoring variants against the fixed champion was attempted
and stopped. The challenger modules were written against production code that no
longer exists in this repo, and clearing one blocker only reveals the next:

| gap | outcome |
|---|---|
| `predict_qualifying(include_grid_scenarios=)` | implemented, then reverted with the shelving |
| `predict_qualifying(include_challenger_evidence=)`, `q1_retrospective_diagnostic=` | Q1-only, cannot work in this harness |
| `QualifyingGridEntry.start_type` dropped by `validate_qualifying_grid` | real bug, fixed in `3810c1ad` |
| `predict_race(grid_scenarios=)` | **stopped here** |

The last one is not plumbing. `race_view_replay.py` passes joint scenarios into
`predict_race` and validates a matching scenario count in the result, so the
race simulation has to sample its starting grid from those scenarios. Rebuilding
that means inventing how scenarios map to draws, how the marginal path stays
seed-comparable, and how the grid anchor is chosen — three decisions with no
surviving source. A wrong reconstruction produces numbers that look valid.

## Open — worth testing when there is time

Ranked by expected value, not by effort.

1. **Raw-laps replay** (`docs/RAW_LAPS_REPLAY_HANDOFF.md`) — the only thing that
   makes Q1 and R0 testable. Everything else in this list is smaller.
2. **Team-strength residual** — the largest known error in the current champion.
   SAI −5.67 and LAW +6.44 are team-level, and `overall_performance` already
   ranks Williams and Audi correctly while the blended strength does not.
3. **Gauge-fix the driver seconds at fit time** — centering currently happens at
   prediction time. `_update_pair_constraint` only applies difference
   constraints, so the per-team level is unidentified and the contamination will
   regrow at the next seeding. Proper fix: centre inside
   `attach_driver_rating_mus` before `team_target_s` is computed, then refit
   `data/processed/team_strength_seconds_mapping/latest.json`.
4. **Combination runs** — every variant so far was tested alone against
   champion. Nothing has tested two changes together, so an interaction that
   only appears when both are active has never been visible.

## 2026-07-30: learning-path fixes, measured by decomposition

Baseline for every number below: `3810c1ad`, rebuilt from the 2026-04-25
preseason driver artifact (`710fb551`) with seconds re-seeded, then all 11
completed rounds replayed offline. **Qualifying MAE 2.6195, mean per-driver
|bias| 1.5522**, scored as the champion protocol above over the 9-event catalog
x 3 seeds (594 driver-events).

Rebuilding matters: the same old code measured against the *stored* 6-round
artifact scores 2.8788. Production had been stuck at 6 rounds because practice
capture reset the season history every Friday. Comparing a fixed model against
that stale artifact credits the fixes with 0.164 MAE they did not earn.

| variant | MAE | mean \|bias\| | verdict |
|---|---|---|---|
| DB-first read + recency-weighted season mean + margin-scored fallback | **2.5993** | **1.4747** | `adopted` |
| the above, plus skipping unpaired drivers in the Bayesian update | 2.8148 | 1.6094 | `worse` |
| the above, with learn-time recency neutralised | 2.8013 | 1.6128 | `worse` |

The three-way split is the point. Measured as one change the package looks like
a 0.064 MAE improvement over the stale artifact and is really a 0.195
regression against a fair baseline. Neutralising the recency weighting moved it
0.014, so that was not the cause. Reverting only the Bayesian change recovered
0.215 and beat baseline.

**Skipping unpaired drivers is `worse`, and the reasoning behind it still
holds.** `update_teammate_relative` gives a driver whose teammate retired the
raw absolute 1..grid_size rating, mixing that scale into a model centred on the
field mean — 32 such observations across the replay set, including one driver
observed at the maximum 22.00 from the single race their teammate retired from.
Dropping those observations costs more than the contamination does. The next
attempt should rescale them, not discard them. Do not re-test discarding.

Related negative result the same day: disabling the `rating_mu` -> skill/pace
blend (`bayesian_quali_skill_blend_cap: 0.0`) scored 3.0505 against a 2.8788
baseline. `rating_mu` correlates only -0.068 with actual qualifying position,
but the raw characteristics it falls back to are worse still.

### Bayesian update confidence rebalance - `worse`

Same baseline and protocol as the entry above. `rating_mu` is a single
position-scale rating updated by both race and qualifying, and it feeds the
*qualifying* skill and pace blend. Race observations carry
`teammate_relative_confidence: 0.35` while qualifying carries
`qualifying_update_confidence: 0.15`, so the qualifying skill term is weighted
more by race results than by qualifying ones.

The distortion that predicts is visible in the 2026 data: backmarkers finish far
better than they qualify (ALO -4.09, STR -4.00, PER -4.16 positions) and
front-runners finish worse (ANT +2.78, RUS +0.94), matching the sign of the
residual bias on both groups.

| variant | MAE | mean abs bias | verdict |
|---|---|---|---|
| champion, quali 0.15 / race 0.35 | **2.5993** | **1.4747** | `adopted` |
| quali 0.35 / race 0.35 | 2.6970 | 1.5354 | `worse` |
| quali 0.35 / race 0.15 | 2.7811 | 1.6667 | `worse` |

Both directions lost, so the shipped 0.15/0.35 split is better than either. The
mechanism above is real but these weights are not the lever that fixes it.
HUL's residual bias sat between +4.5 and +5.3 in every arm including champion,
so nothing here moved the driver the hypothesis was aimed at. Do not re-test
either direction without a new mechanism.

### Margin-scored telemetry race pace - `worse`

Same baseline and protocol. `extract_team_performance_from_telemetry` computes a
per-team median race lap time and then discards it for
`1.0 - rank/(team_count-1)`, so team strength cannot express how large a gap
was. This variant kept the margin: delta against the field median as a fraction
of lap time (track-length invariant), mapped onto 0-1 by a configurable spread.

| variant | MAE | mean abs bias | verdict |
|---|---|---|---|
| champion, rank-collapsed | **2.5993** | **1.4747** | `adopted` |
| margin, spread 0.06 | 2.6700 | 1.7138 | `worse` |
| margin, spread 0.10 | 2.7744 | 1.8586 | `worse` |

It did what it was designed to do: Aston's season went from a flat
`[0.1, 0.0, 0.0, ...]` to `[0.028, 0.042, 0.04, 0.147, 0.0, 0.43, ..., 0.447]`,
so an upgrade that closes a deficit without changing rank is finally visible.
Qualifying accuracy still got worse, and monotonically - the wider the spread,
the worse the result, meaning the closer the scoring stays to rank the better.

The reason is the input, not the idea. A race median lap time carries strategy,
traffic, fuel load and safety cars: across three 2026 races the team spread was
4.6-4.8s, more than twice the 1.97 s/unit that
`team_strength_seconds_mapping` was fitted for. Rank is robust to that noise and
margin propagates it. Do not retry margin scoring on race medians.

The idea is not dead, but it needs a pace measure built for it. The matched-lap
same-session construct in `src/extractors/matched_laps.py` is what the seconds
mapping was actually fitted on; converting *that* through the calibrated slope
is the version worth testing. Converting race medians through it would be a
scale error, mixing two different definitions of "seconds".

### Still open

- `extract_team_performance_from_telemetry` computes real median lap times per
  team, then discards them for `1.0 - rank/(team_count-1)`. It is the primary
  path for most races, so team strength still cannot express margin; the
  2026-07-30 fallback fix only reached races where telemetry was missing.
  `team_strength_seconds_mapping/latest.json` already has the calibrated slope
  to convert lap-time deltas directly.
- HUL degrades **as the season is learned**, on unmodified code: +1.52 at 6
  rounds, +5.26 at 11. RUS moves +0.15 -> +2.19 the same way. Whatever causes
  that is in the learning path and predates all of the above.
  **Superseded 2026-07-31** — see the section below. The "learning path" framing
  is wrong, and treating HUL and RUS as one phenomenon is wrong.

## 2026-07-31: the HUL/RUS drift is not a driver-rating problem

**No prediction run was made for this entry.** Everything below is derived from
the shipped artifacts (`car_characteristics` v23 at 11 rounds,
`driver_characteristics` v26, `team_strength_seconds_mapping/latest.json`), the
9-event `event_catalog.json`, and code at `6771a8a5`. It carries no MAE because
it scores no variant. It is here to stop the next session spending runs on a
path that arithmetic already closes.

### The driver rating cannot produce an error this large

Centred `quali_rating_mu_s` for HUL is **-0.1225 s**
(`center_rating_mu_by_team`, `qualifying_preparation.py:807`). The qualifying
simulation projects it as `0.5 + seconds_delta / 1.9708`, clipped to [0, 1]
(`qualifying_simulation.py:359`). So HUL's entire driver term is **0.062 score
units**, and a 22-car grid spans 1.0 unit at roughly 0.045 units per position.

**Driver-term authority: about 1.4 grid positions. HUL's bias is +5.26.**

This is the explanation for the four failed hypotheses above. None of them was
obviously wrong in mechanism; all of them were pulling a lever with ~1.4
positions of authority against a ~5.3 position error. Do not test another
driver-rating variant against this bias without first showing the path has
enough authority to matter.

### Both teammates carry the same-signed bias, so it is a team offset

Recorded elsewhere in this file but never connected: HUL +5.26 / BOR +4.44
(champion-history note), RUS +2.19 / ANT +2.78 (confidence-rebalance section).
Same sign, comparable magnitude, same team, both pairs. That is the signature of
a team-strength offset, not a driver-rating error. **"The HUL/RUS bias" is a
misnomer for an Audi and Mercedes team-strength bias.**

### HUL and RUS do not share a mechanism

Model teammate ordering (centred `quali_rating_mu_s`, positive = faster) against
actual head-to-head over the 9 catalog events:

| team | model rates faster | actual H2H | agrees |
|---|---|---|---|
| Audi | BOR (+0.1225) | **HUL 5-4** | no |
| Red Bull | HAD (+0.0079) | **VER 6-3** | no |
| RB | LIN (+0.0955) | **LAW 6-3** | no |
| Cadillac | BOT (+0.1440) | **PER 6-3** | no |
| Mercedes | ANT (+0.1003) | ANT 5-4 | yes |
| McLaren | NOR (+0.0878) | NOR 5-4 | yes |
| Ferrari | LEC (+0.0099) | LEC 5-4 | yes |
| Aston Martin | ALO (+0.3537) | ALO 7-2 | yes |
| Alpine | GAS (+0.0221) | GAS 6-3 | yes |
| Haas | BEA (+0.1161) | BEA 7-2 | yes |
| Williams | SAI (+0.0442) | SAI 7-2 | yes |

Four of eleven pairs are ordered backwards. HUL is one of them. **RUS is not** —
Antonelli genuinely out-qualified him 5-4, so that rating direction is correct
and RUS's residual has to come from Mercedes team strength alone. Lumping the
two drivers into one phenomenon is what framed the whole search wrongly.

Sign convention verified rather than assumed: `sign_convention:
positive_seconds_means_faster_than_field` in the mapping artifact, consistent
with the `0.5 + delta/scale` projection where higher score is a better grid slot.

### Observation count does not compress the teammate gap - hypothesis killed

Worth stating because it looks plausible and costs a run to test. Centred
teammate gap against `quali_rating_observations` across all 11 teams shows no
relationship: Audi at 11 observations has the **largest** gap among
well-observed teams (0.245 s), Red Bull at 12 observations has the **smallest**
(0.016 s). Aston Martin's 0.708 s gap sits on 3 observations, so the two
extremes are the low-observation teams in both directions.

Do not test "more learning shrinks teammate gaps".

### Separate bug found while measuring: qualifying uses the race slope

`config/default.yaml:219` sets
`team_strength_seconds_score_scale: 1.9707717329051126`. That is the **race**
slope from `team_strength_seconds_mapping/latest.json`. It is applied in the
**qualifying** projection at `qualifying_simulation.py:359`, where the fitted
qualifying slope is **1.7741686893278807**.

Every qualifying deviation from 0.5 is therefore compressed by about 11%. This
is unrelated to the HUL residual and is a one-line fix, but it rescales the
whole qualifying grid, so measure it on its own before or after anything else —
never bundled.

### Team-vs-driver decomposition, measured - confirms the reframe

`identify_systematic_errors` (`src/analysis/model_evaluation.py:427`) already
computes `team_bias` alongside `driver_bias`. Every measurement before this one
read `driver_bias` only. Running both over cached champion predictions
decomposes each driver's residual into a team-shared component and a
within-pair component.

**Baseline caveat, and it is material.** The only cached champion predictions
are `source_digest` `3f07ca70`, written **2026-07-19 to 2026-07-22**, so they
predate the 2026-07-28 centering fix `93bfbeb0`. The prediction cache key does
not cover code version — the trap documented under "Measurement protocol" — so
these levels are *not* current-champion numbers. HUL reads +6.86 here against
+5.26 on rebuilt current code.

**The levels are stale; the decomposition is the claim.** The centering fix
removes the team-mean component of the driver rating, so on current code the
within-pair component can only be smaller than what is shown here. That
strengthens the conclusion rather than threatening it.

Protocol: champion variant, `qualifying` kind, `PRE` checkpoint, 7 events x 3
seeds, 462 driver-observations, actuals from `actual_qualifying_grid` in
`event_catalog.json`. Positive means too pessimistic.

| team | team bias | driver 1 | driver 2 | within-pair spread |
|---|---|---|---|---|
| **Audi** | **+6.71** | HUL **+6.86** | BOR **+6.57** | **0.29** |
| Williams | -7.10 | SAI -7.86 | ALB -6.33 | 1.53 |
| RB | +3.33 | LAW +2.48 | LIN +4.19 | 1.71 |
| McLaren | +1.74 | NOR +0.90 | PIA +2.57 | 1.67 |
| Mercedes | +1.14 | RUS +0.14 | ANT +2.14 | 2.00 |
| Alpine | +1.12 | GAS +1.48 | COL +0.76 | 0.72 |
| Cadillac F1 | +0.05 | PER +1.29 | BOT -1.19 | 2.48 |
| Haas F1 Team | -0.81 | OCO +0.62 | BEA -2.24 | 2.86 |
| Ferrari | -0.90 | LEC -0.76 | HAM -1.05 | 0.29 |
| Red Bull Racing | -2.43 | VER -5.52 | HAD +0.67 | **6.19** |
| Aston Martin | -2.86 | ALO -5.52 | STR -0.19 | **5.33** |

#### Correction, same day: the two extreme rows are the already-fixed bug

The first reading of this table treated Audi +6.71 and Williams -7.10 as the
largest *unexplained* errors in this file. That was wrong, and the table's own
baseline is why.

These cached predictions predate `93bfbeb0`, so they ran with **uncentered**
`quali_rating_mu_s` — the double-counted car pace that `93bfbeb0` fixed. The raw
team-mean driver rating still shows the size of it (values from the current
11-round artifact, so indicative of the state in force rather than exact):

| team | team-mean raw rating | implied position effect |
|---|---|---|
| Williams | +0.412 s | **+4.60** |
| Audi | -0.387 s | **-4.32** |

That is an 8.9 position spread between the two teams from the driver-rating term
alone. The observed gap in predicted mean position is **9.67** (Audi 18.86 vs
Williams 9.19) on team strengths that are nearly identical (0.366 vs 0.354) and
correctly ranked — Williams' strength rank 9 matches its actual rank 9.

So the team strength for both teams is approximately right, and the prediction
error is the uncentered driver term. This is the same defect the champion-history
entry already records collapsing: HUL +6.11 -> +1.67, ALB -5.78 -> +0.11.
**Audi and Williams are not open problems. They are the pre-`93bfbeb0` state.**

#### What this does to the HUL conclusion

The "HUL is closed as a driver problem" reading above is **withdrawn**, but the
stated reason for withdrawing it was also wrong, and the third pass settled both.
Recorded in full because two of the three readings here were mistakes.

**The within-pair column measures separation *error*, not driver-term magnitude.**
Bias is predicted minus actual per driver, so the spread between teammates is
`(predicted gap) - (actual gap)`. Audi's 0.29 does not mean the driver term is
inert; it means the pre-fix model got HUL-vs-BOR *separation* about right while
both drivers carried the same large shared offset. The column is readable — just
not as "how many positions of driver signal exist".

**The clip is not what compresses Audi.** Neither Audi driver is near a bound.
That hypothesis is dead.

### The clip is binding, at the front of the grid

Testing it turned up a different result. Deterministic score is
`clip(0.5 + 1.7742*(strength-0.5)/1.9708 + driver_mu/1.9708, 0, 1)`. Evaluated
over all 22 drivers on 6-round strengths:

| state | drivers hitting a bound |
|---|---|
| raw / pre-`93bfbeb0` | **ANT 1.136, LEC 1.060, HAM 1.050, RUS 1.035** (all clipped to 1.0), STR -0.049 (clipped to 0) |
| centred / post-`93bfbeb0` | STR only |

Before the fix, **four front-runners collapsed onto the identical score 1.0**, so
their relative order was simulation noise rather than signal. That is a second,
previously unrecorded consequence of the uncentered ratings, and `93bfbeb0`
incidentally fixed it. Worth knowing for any pre-fix number involving Mercedes or
Ferrari: their internal ordering was not being modelled at all.

The team term alone cannot reach a bound — it spans only ±0.42 against a ±0.5
threshold — so every clip event needs the driver term to push it over. Post-
centering, only Aston Martin still clips.

### Current-state driver ordering, and the one that is wrong

Ranking all 22 drivers by centred deterministic score — current 11-round
strengths and current ratings, so this *is* today's state — against actual
head-to-head over the 9 catalog events:

| pair | model order (centred rank) | actual H2H | |
|---|---|---|---|
| **Audi** | **BOR 13, HUL 18** | **HUL 5-4** | **inverted by 5 positions** |
| Red Bull | HAD 7, VER 8 | VER 6-3 | inverted by 1 |
| Mercedes | ANT 1, RUS 2 | ANT 5-4 | correct |
| Williams | SAI 14, ALB 16 | SAI 7-2 | correct |
| Aston Martin | ALO 19, STR 22 | ALO 7-2 | correct |

**HUL sits five places behind the teammate he actually outqualifies.** That is
the largest live driver-level error on the grid and it is a sign error, not a
magnitude error. It is unaffected by everything withdrawn above: it uses centred
ratings, current artifacts, and actual results — no cached prediction, no
pre-fix state.

So driver-level work does belong on HUL/BOR after all, but for the inversion, not
for the "degrades as the season is learned" framing that opened this
investigation. Caveat: deterministic score ordering is not simulated mean
position — Q1/Q2/Q3 structure and noise both intervene — so treat the rank gaps
as indicative and the sign as the finding.

### Mechanism of the inversion: rookie prior sigma sets the update gain

**There is no sign bug.** The path was checked end to end and is consistent:
`matched_gap_s = comparison_lap_time_s - reference_lap_time_s`, positive means
reference faster (`matched_laps.py:164`); `innovation = observed_gap -
(reference_mu - comparison_mu)` raises `reference_mu` on positive innovation
(`driver_seconds_state.py:267`); higher mu is faster. Do not re-hunt this.

**The prior had HUL and VER the right way round; 2026 flipped them.** Seed values
from `teammate_network_prior/latest.json` (2022-2025) against the current
11-round state:

| pair | prior mu (a / b) | prior sigma | var ratio b:a | now | flipped |
|---|---|---|---|---|---|
| Red Bull VER/HAD | +0.447 / -0.145 | 0.152 / 0.530 | **12.2** | +0.299 / +0.315 | **yes** |
| Audi HUL/BOR | -0.461 / -0.554 | 0.231 / 0.530 | **5.3** | -0.509 / -0.264 | **yes** |
| Mercedes RUS/ANT | +0.453 / +0.232 | 0.277 / 0.530 | **3.7** | +0.226 / +0.426 | **yes** |
| Aston ALO/STR | +0.143 / -0.111 | 0.152 / 0.155 | 1.1 | +0.364 / -0.343 | no |
| Ferrari LEC/HAM | +0.498 / +0.395 | 0.273 / 0.275 | 1.0 | +0.456 / +0.437 | no |
| Williams SAI/ALB | +0.421 / +0.405 | 0.272 / 0.260 | 0.9 | +0.456 / +0.368 | no |

**Every pair with a variance ratio above ~3 reordered. Every pair near 1.0 held.**
The pair update splits each innovation in proportion to variance
(`updated_reference_mu = reference_mu + (reference_var/denominator)*innovation`),
so the wider-prior driver absorbs the update. Audi: BOR moved +0.289 while HUL
moved -0.048. Red Bull: HAD moved +0.460 while VER moved -0.148.

`BOR`, `HAD` and `ANT` all carry sigma **0.5304206197376727** — bit-identical, so
it is a shared default for low-observation drivers, not a fitted value. Against
HUL's 0.231 (n=37) and VER's 0.152 (n=74) that is a 5x to 12x gain advantage on
every observation.

Cadillac PER/BOT has ratio 12.2 and did *not* flip, because the prior already had
BOT ahead and he simply moved further ahead. Consistent with the mechanism rather
than a counterexample.

**This is correct Bayesian behaviour given those priors, and it is not always
wrong** — the Mercedes flip agrees with actual head-to-head (ANT 5-4). The
question the mechanism raises is narrower: whether 0.5304 is the right default,
and whether qualifying evidence is strong enough to justify that gain.
`min_matched_pairs_quali` is **3**, most qualifying sessions produce exactly 3
matched pairs, and in 2025 eleven of twenty-one BOR/HUL qualifying sessions
produced no usable aggregate at all (`insufficient_matched_pairs`). Ratings are
being reordered on three-lap medians.

Two levers, neither tested: raise the low-observation prior sigma default, or
raise `min_matched_pairs_quali`. See the scoring note at the end of this section
for what testing them actually costs — it is not just a config edit.

### Measured on current code, and the mechanism holds

The blocked walk-forward was routed around. A champion-only scorer calling
production `Baseline2026Predictor.predict_qualifying(..., practice_signal_mode=
"stored_profiles")` plus the tracked `identify_systematic_errors` reproduces the
decomposition without touching the shelved tree. 7 dry events x 3 seeds x 20
simulations, 462 driver-observations, about 2 minutes.

**This is not a walk-forward.** Every event is predicted against the current
artifact state, which already contains that event's results. Absolute error is
optimistic and **not comparable to the walk-forward numbers above**. Both arms of
an A/B carry the same leakage, so deltas remain usable. Recorded level:
MAE 2.6494, mean per-driver |bias| 1.7186.

| team | team bias | driver 1 | driver 2 | spread |
|---|---|---|---|---|
| Audi | +2.26 | HUL **+4.62** | BOR -0.10 | 4.71 |
| RB | +1.45 | LAW **+3.38** | LIN -0.48 | 3.86 |
| Mercedes | +1.29 | RUS **+3.14** | ANT -0.57 | 3.71 |
| Cadillac F1 | +0.98 | PER +1.67 | BOT +0.29 | 1.38 |
| Ferrari | +0.55 | LEC +0.29 | HAM +0.81 | 0.52 |
| Haas F1 Team | +0.31 | OCO **+2.38** | BEA -1.76 | 4.14 |
| Alpine | -0.43 | GAS +0.24 | COL -1.10 | 1.33 |
| Williams | -1.38 | SAI -2.19 | ALB -0.57 | 1.62 |
| Red Bull Racing | -1.52 | VER -3.95 | HAD +0.90 | 4.86 |
| McLaren | -1.69 | NOR -4.05 | PIA +0.67 | 4.71 |
| Aston Martin | -1.81 | ALO -4.14 | STR +0.52 | 4.67 |

**The picture inverted relative to the pre-`93bfbeb0` table, exactly as
predicted.** Largest team offset fell from 7.10 to 2.26 — the centering fix
removed the shared component — and the within-pair spreads grew from a 0.29-2.00
band to 3.7-4.9. Pre-fix, teammates shared one large offset and the driver error
was hidden inside it. Post-fix the offset is gone and the driver error is what is
left. This also retroactively confirms the correction above: the pre-fix
within-pair column really was measuring separation error against a shared offset.

**Where a wide-prior rookie exists, the veteran carries the positive bias and the
rookie sits near zero**: Audi (HUL +4.62 / BOR -0.10), RB (LAW +3.38 / LIN
-0.48), Mercedes (RUS +3.14 / ANT -0.57), Haas (OCO +2.38 / BEA -1.76). Four of
the six such pairs. The rookie's rating has been pulled to fit and the veteran
absorbs the residual.

**Refinement: the mechanism is overshoot, not inversion.** Mercedes is the case
that shows it. The model's ANT-ahead ordering *agrees* with head-to-head, yet the
rating gap is 0.20 s (~2.3 positions) on a 5-4 split, and RUS still carries
+3.14. Rookie gain moves the rating too far whether or not it crosses over.
Crossing over (Audi, RB) is the visible symptom; the magnitude error is the
disease, and it is present in pairs that look correctly ordered.

**Red Bull is not this mechanism.** VER -3.95 with HAD +0.90 is the opposite
sign, and the VER/HAD rating gap is 0.016 s — about one position, far too small
to produce it. Red Bull's error is team strength, not driver rating. Same for
McLaren and Aston Martin, both of which have near-equal teammate sigmas and so no
gain asymmetry at all.

### The prior sigma is a clamp, not an estimate - both tuning levers rejected

Before scoring either lever, the quantity they tune was checked. It is not a
per-driver uncertainty for most of the grid.

`_driver_sigma` (`scripts/build_teammate_network_prior.py:841`) has three
branches:

```python
fallback_sigma = max(1.75 * population_sd_s, sigma_floor_s)
if not anchored:                                    return fallback_sigma
if n_observations < config.min_driver_observations: return fallback_sigma
trusted_sigma = max(bootstrap_sigmas[driver], 0.5 * population_sd_s, sigma_floor_s)
```

With `main_component_population_sd_s = 0.3030975`, both saturation values in the
artifact fall out exactly:

- `1.75 * 0.3030975 = 0.530421` — the fallback, **14 of 31 drivers**
- `0.5  * 0.3030975 = 0.151549` — the floor, **9 of 31 drivers**

**23 of 31 drivers carry a clamp. Only 8 have a sigma derived from their own
bootstrap.** The cliff is `min_driver_observations = 24`:

| driver | n_obs | sigma | source |
|---|---|---|---|
| DOO | 3 | 0.5304 | fallback |
| BOR | 10 | 0.5304 | fallback |
| ANT | 19 | 0.5304 | fallback |
| LAW | 23 | 0.5304 | fallback |
| RIC | 31 | 0.1515 | floor |
| HUL | 37 | 0.2309 | bootstrap |

DOO at 3 observations and LAW at 23 are assigned identical uncertainty. LAW and
RIC differ by eight observations and land 3.5x apart. **The quantity that sets
Bayesian update gain is a step function at an arbitrary threshold, with no
gradient across three quarters of the grid.**

**Both tuning levers are therefore rejected without being scored.** Raising the
`0.5304` default would mean overriding a fitted population quantity with a
hand-picked one to compensate for a threshold artefact, and it would move all 14
fallback drivers together regardless of whether they have 3 observations or 23.
Raising `min_matched_pairs_quali` starves an evidence stream that already fails
to produce an aggregate in half of all qualifying sessions. Neither survives at
20 rounds; both are counter-tuning.

**The structural fix, not attempted:** make the prior sigma continuous in
evidence — one shrinkage scaling with observation count and graph connectivity —
replacing the fallback / bootstrap / floor branches. Then BOR at 10 and LAW at 23
differ because their evidence differs, and the rookie-gain overshoot dissolves
rather than being counter-tuned.

Two caveats before anyone builds it. First, **nothing here has been shown to
convert into MAE** — the base rate on this bias is four mechanisms, four losses,
and a smaller |bias| is not automatically a smaller MAE. Run the cheap
prediction-side proxy (scale the centred teammate gap and re-score) to establish
the sign before spending a rebuild. Second, `population_sd_s` is itself a
main-component fit output, so a continuous scheme needs its own validation rather
than a drop-in swap.

### What scoring the two levers would have cost

Neither lever is testable by editing config and re-predicting. Both change how
state is *learned*:

- `min_matched_pairs_quali` gates aggregate-row production in the extractor.
- the 0.5304 prior sigma lives in `teammate_network_prior/latest.json`, built by
  `scripts/build_teammate_network_prior.py`.

So each arm needs: rebuild the prior (sigma arm only) -> re-seed driver seconds
-> replay all completed rounds -> then score. That is the rebuild procedure with
its two documented silent traps (the driver-baseline default that double-counts,
and `USE_DB_STORAGE` replaying onto stored state). Budget an hour per arm, not a
config edit. The scorer is the cheap half and it already exists.

### Still open after this

- Re-measure the decomposition on current code. The table above is structurally
  sound but its levels predate `93bfbeb0`, and no cached champion prediction
  exists after that commit.
  **Attempted 2026-07-31 via `run_challenger_research_walk_forward.py` and
  `refused`** — `TypeError: BaselineQualifyingMixin.predict_qualifying() got an
  unexpected keyword argument 'include_grid_scenarios'`. That is the first row
  of the "Blocked" table above: the challenger harness has been shelved since
  2026-07-29 and cannot run against current production code. **The walk-forward
  runner is not a route to a champion re-measurement.** Note the runner exits 0
  after printing the traceback, so a re-run that "completes" has still produced
  nothing.
  The remaining route is champion-only and avoids the shelved tree entirely:
  `predict_qualifying(year, race_name, practice_signal_mode="stored_profiles")`
  is production API, and `historical_replay.py`, `checkpoint_reconstruction.py`
  and `model_evaluation.py` are all tracked. It needs a small purpose-built
  driver loop over the 9 catalog events, which is a reconstruction — smaller
  and lower-risk than the race-scenario reconstruction that stopped the
  challenger work, but still capable of producing plausible wrong numbers if
  the checkpoint state is assembled incorrectly. Not attempted.
- ~~The Audi and Williams team offsets are the two largest single errors in this
  file and have no mechanism.~~ **Withdrawn the same day** — see the correction
  above. Both are the pre-`93bfbeb0` uncentered driver rating, already fixed.
  Neither is a team-strength question: both teams' strength values are about
  right and correctly ranked.
- ~~Why does a back-of-grid pair's predicted positions compress?~~ **Settled the
  same day** — nothing compresses. The within-pair column is a separation
  *error*, and no Audi driver is near the clip. See the correction above.
- **HUL/BOR is inverted by five positions in the current state** and is the
  largest live driver-level error found. Sign error, not magnitude. This is the
  one open item from this investigation that rests on nothing withdrawn.
  Mechanism identified and then confirmed on current code (HUL +4.62 against BOR
  -0.10): the shared low-observation prior sigma `0.5304` gives a rookie teammate
  5-12x the update gain of an established one, so thin qualifying evidence pulls
  the rookie's rating too far. **Overshoot, not inversion** — Mercedes is ordered
  correctly and still carries RUS +3.14, so pairs that look right are affected
  too. ~~Two levers: the prior sigma default and `min_matched_pairs_quali`.~~
  **Both rejected without scoring** — see "The prior sigma is a clamp, not an
  estimate". The open item is the structural fix (continuous sigma), gated on the
  cheap proxy showing the direction pays at all.
- **Red Bull, McLaren and Aston Martin are a separate problem.** Large
  within-pair spreads (4.9, 4.7, 4.7) with the wrong sign for rookie gain, and
  near-equal teammate sigmas, so no gain asymmetry exists to explain them. VER
  -3.95, NOR -4.05, ALO -4.14 are all the stronger driver predicted too well.
  Unexplained; likely team strength.
- **Four front-runners clipped to an identical 1.0 before `93bfbeb0`.** Any
  pre-fix result that depends on the internal ordering of Mercedes or Ferrari is
  reading noise. Applies retroactively to entries above measured on that state.
- The four inverted teammate pairs (Audi, Red Bull, RB, Cadillac) are measured
  against actual head-to-head, not against a model state, so they are unaffected
  by the baseline problem above.

## 2026-08-03: the low-observation prior sigma, finally scored — `noise`

Supersedes "both tuning levers rejected without being scored" above for the
sigma lever only. The rejection reasoning there still reads correctly; what it
could not know is how little the lever moves. `min_matched_pairs_quali` remains
unscored.

**What the variant changes.** One number: the low-observation prior sigma
multiplier in `_driver_sigma`, `1.75 * population_sd_s`. Exposed as
`--low-observation-sigma-multiplier` in `2aafbfc2` so an arm needs no source
edit. Nothing else differs between arms.

**Baseline.** Champion `b1381e06`. Arms ran on `71fa3615`, which adds only the
flag and the scorer; rebuilding the prior at the 1.75 default reproduces the
shipped artifact except at the 16th significant digit, so the flag is
behaviour-preserving.

**Protocol — differs from the one above.** Scored with
`scripts/champion_quali_bias.py`, 9 catalog events x 3 seeds x 20 simulations,
`--all-events` so wet rounds are included. **Leakage-inclusive**: every event is
predicted against a state containing its own result, so these MAEs are not
comparable to the walk-forward numbers elsewhere in this file. Valid only as
a delta between arms carrying identical leakage.

Each arm ran the full path — rebuild prior, rebuild rookie fallback, restore the
`710fb551` preseason driver artifact, re-seed, replay all 11 rounds, score. Both
documented traps were avoided (explicit `--driver-baseline-file` via the
preseason restore, `USE_DB_STORAGE` unset). Cost was **16s per race, about 6
minutes per arm**, not the hour budgeted above.

**Harness validation.** The 1.75 arm — full preseason reseed plus 11-round
replay — reproduces the shipped production artifact's score exactly: MAE 2.6734,
HUL +4.89, BOR -1.22, every team row identical. The rebuild path is faithful.

| arm | multiplier | rookie:established update gain | MAE | mean per-driver \|bias\| | HUL | Audi spread |
|---|---|---|---|---|---|---|
| baseline | 1.75 | 5.28x | 2.6734 | 1.4747 | +4.89 | 6.11 |
| arm | 1.00 | 1.72x | 2.6734 | 1.4579 | +4.85 | 6.00 |
| bound | 0.50 | 0.43x | 2.6599 | 1.4141 | +4.74 | 5.81 |

**Verdict: `noise`.** Cutting rookie update gain from 5.28x to 1.72x moves HUL
by 0.04 grid positions and leaves MAE bit-identical. The 0.5 bound is included
only to bracket the channel — it is not a defensible setting, because it gives a
10-observation rookie *less* update gain than a 37-observation veteran — and even
there HUL improves 0.15 against a +4.89 error, roughly 3% of the authority
required. The scorer is deterministic (verified: byte-identical artifacts and
exact MAE reproduction across runs), so these are real differences, not sampling
variation. They are simply immaterial.

Direction of every pair is consistent and correct — spreads shrink, |bias|
falls — so the mechanism described above is real. It is not load-bearing.

**Which pairs respond.** RB moves most (spread 4.00 -> 3.19 at the bound, -0.81),
consistent with LAW sitting at 23 observations, one short of the cliff. Mercedes
moves the wrong way (RUS +2.59 -> +2.67). Aston Martin does not move at all.

**What this closes.** The open item above gates the structural fix (continuous
sigma) on "the cheap proxy showing the direction pays at all". The proxy has now
been run at two settings including an extreme bound. **The direction pays, and
pays about 3% of what is needed.** Building a continuous-shrinkage scheme to
capture a 0.15-position effect on the largest live error is not worth it on this
evidence. The clamp remains poor engineering — a step function where a gradient
belongs — but it is not the cause of the HUL/RUS drift, and fixing it will not
fix that.

This is the fifth mechanism tested against this bias and the fifth to lose. The
authority-ceiling argument continues to hold: the driver-rating path cannot
produce a 5-position error, so the cause is not on that path.

## Adding an entry

Keep it to what a future reader needs to trust or discard the result:

- what the variant changes, in one line — the thesis, not the implementation
- the champion baseline it was measured against, by commit
- the protocol, if it differs from the one above
- the numbers, including how many events went each way — a mean delta alone
  hides a 4–3 split
- a verdict from the table at the top
- for `never activated` or `refused`, the disclosed reason verbatim

A result with no baseline recorded is not reusable later. That is the single
most common way this kind of log goes stale.

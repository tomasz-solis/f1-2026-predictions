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

**Challenger work is shelved as of 2026-07-29.** The tree stays untracked in the
worktree; no branch was made.

> Paths in this section — `docs/RAW_LAPS_REPLAY_HANDOFF.md`,
> `docs/QUALIFYING_RACE_CHALLENGER.md`,
> `scripts/run_challenger_research_walk_forward.py`, and the
> `src/analysis/challenger_*` modules — live only in that untracked tree. They
> are not in a fresh clone. The walk-forward artifacts under
> `data/historical_replay/2026/` and
> `data/model_diagnostics/2026/race_mae_investigation/` are also untracked, and
> they are the evidence behind every number above: keep them.

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

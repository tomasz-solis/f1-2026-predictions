# DNF calibration brief — scoping (no implementation, no backtest compute)

Purpose: reduce the systematic part of race finishing-position error caused by
DNFs. DNFs contribute ~1.2–1.4 MAE positions on incident weekends (Australia
+1.39, China +1.21, Monaco +1.20 — from the 2026 decomposition). This scopes a
calibration change; it does **not** implement anything or run the FastF1-heavy
backtest that has been stalling.

## How DNFs are modelled today (mapped in code)

1. **Per-driver rate, learned by a track-AGNOSTIC EMA.**
   `src/systems/updater.py:488` `_update_dnf_rate_ema`: for each driver,
   `updated = (1-w)*existing + w*(1 if retired else 0)`, clipped to
   [floor, cap]. It keys only on driver code — **a retirement at Monaco updates
   the driver's global rate identically to one at Monza.**
2. **Experience add-ons (flat).** `src/predictors/baseline/race/preparation_flow.py:18`
   `_EXPERIENCE_DNF_MODIFIERS`: rookie +0.05, second_year +0.03, developing
   +0.02, established +0.00. Added to the driver rate.
3. **Config bounds.** `src/utils/config_schema.py:996-1000` and
   `src/utils/constants.py:15-16,41`: base_rate 0.04, default 0.10, historical
   cap 0.20, final cap 0.35, floor 0.02.
4. **Simulator draw (single Bernoulli, race-long).**
   `src/predictors/baseline/race/params_mixin.py:110`:
   `dnf_occurred = rng.random() < info["dnf_probability"]`; a hit scores the
   driver `-10 + uniform(-1,0)` (dumped to the back). One draw per driver per
   race — no lap, no grid position, no first-lap notion.
5. **Scoring already exists.** `src/analysis/model_evaluation.py:347+` computes a
   DNF Brier score and `actual_dnf_rate` vs predicted — so we can measure
   calibration WITHOUT running the race Monte Carlo.

## DNF is TWO processes, not one (the load-bearing insight)

A DNF is **technical failure OR driver-caused crash**, and they have completely
different dynamics:

| | Technical / mechanical | Collision / driver crash |
|---|---|---|
| Level | Team / power-unit | Driver (× track) |
| Correlation | Both cars of a team correlated | Independent per driver |
| Track dependence | Weak | Strong (street/Monaco/Baku ≫ Monza) |
| **Time behaviour (2026)** | **Non-stationary — HIGH early due to the regulation reset, DECAYS as teams cure reliability, then levels off** | Roughly stationary |

**2026 is a regulation-reset year**, so early-season DNFs are inflated by
technical teething that will fall away. This breaks the current model twice over:

1. The learned rate is **per-driver**, but mechanical failure is **team/PU-level**
   and correlated across teammates (a Cadillac reliability problem hits both
   cars). A per-driver EMA can't see that.
2. The EMA blend is **stationary** (fixed weight), so it bakes in the high
   early-season technical rate and **keeps over-predicting DNFs as the cars
   actually get reliable**. A decaying process needs recency-weighting or an
   explicit downward trend, not a flat EMA.

## The gap (the "flat-rate assumption")

DNF probability = **one per-driver global rate + flat experience add-on**, with
**zero track dependence, zero team/mechanical split, zero time-trend, zero
grid/first-lap dependence, and a single race-long draw**. Beyond the two-process
problem above:
- Circuit *collision* attrition varies ~2-3x (street/Monaco/Baku ≫ Monza).
- Track files ALREADY carry `safety_car_prob` per circuit (an incident proxy)
  that is **not wired into DNF** — a ready-made prior sitting unused.
- Many collision DNFs are first-lap/first-corner incidents concentrated in the
  midfield pack — grid-position dependent — which the single race-long draw
  cannot express.

## Proposed change — ranked

The two-process insight promotes the split to the CORE of the design; the model
should compute `p_dnf = p_technical(team, time) + p_collision(driver, track, grid)`
(combined as independent risks: `1 - (1-p_tech)*(1-p_coll)`).

1. **Split DNF into technical + collision components.**
   - `p_technical`: estimated at **team/PU level** (shared by teammates), with a
     **time-decay toward a reliability floor** so the 2026 reg-reset teething
     falls off instead of being baked in. Simplest viable form: recency-weighted
     team retirement rate with a short half-life, or an explicit
     `base + amplitude * exp(-races/tau)` decay fit per team, floored at a mature
     reliability rate. A new team (Cadillac) starts at the high prior and is
     allowed to decay as data arrives.
   - `p_collision`: **driver × track**, roughly stationary — the per-driver
     crash tendency (current EMA is a reasonable starting point for this half)
     times a per-track collision multiplier.
2. **Per-track collision multiplier.**
   Add `collision_multiplier` (or `attrition_index`) per circuit to
   `data/processed/track_characteristics/*.json` from historical per-circuit
   retirement rates (FastF1 results carry classified/retired status), normalised
   to 1.0 at the field-average circuit; bootstrap from the unused
   `safety_car_prob` where circuit history is thin. Apply via the existing
   resolve-by-circuit-key path (`preparation_mixin.py:263-280`, as
   `overtaking_difficulty` already does). Keep existing cap/floor.
3. **First-lap / grid-position incident term.**
   A small first-lap collision component scaling with grid position (pack
   density), plus the race-long component. Ties collision risk to the predicted
   grid and captures first-corner chaos. Additive, small, capped.
4. **Lap-of-retirement for classification order.**
   Instead of every DNF → `-10`, order retirements by simulated lap reached so a
   late retirement classifies ahead of an early one. Reduces MAE *within* the
   DNF group at no modelling cost.

Note the interaction with the regulation reset: because `p_technical` decays,
the model must **not** freeze early-2026 DNF rates as permanent — the whole point
is that it should track reliability improving across the season. Any offline
validation must therefore be walk-forward (fit on races < N, predict N), or the
decay will look artificially good in hindsight.

## Leakage & validation (backtest-free)

- **Leakage:** per-track rates and any driver reliability must use only events
  strictly before the target event (same walk-forward cutoff discipline as the
  rest of the research). A circuit's own current-year race must never inform its
  own DNF prior.
- **Validate offline, not via the 500-sim Monte Carlo** (which is what stalls on
  FastF1). The DNF probability is a *calibration* problem: use the existing
  `compute_prediction_accuracy` DNF Brier / `actual_dnf_rate` machinery
  (`model_evaluation.py:347+`) to score predicted-vs-actual DNF on held-out
  events directly from **already-cached actuals** — no race simulation needed.
  Success = lower DNF Brier and better per-track calibration out-of-sample. Only
  after the calibration improves offline is a (single, sequential, stall-watched)
  full-sim confirmation worth attempting.
- **Honest ceiling:** all-driver MAE has an irreducible stochastic DNF floor;
  the target metric for skill is **finisher MAE / conditional-grid MAE** (already
  ~2.9 in 2026). This change trims the *systematic* DNF error (wrong rates at the
  wrong tracks), not the random part.

## Production boundaries (unchanged)

Same rules as the whole investigation: no production config/champion changes to
ship a research artifact; implement behind config defaults that reproduce current
behaviour until a change is explicitly approved; `config/production_config.json`
untouched. The per-track multiplier defaulting to 1.0 everywhere = exact current
behaviour, so it can land dormant and be enabled per-research-run first.

## Implementation status: item 1 (technical/collision split) landed, v2

Item 1 above (the two-process split) is implemented behind
`baseline_predictor.race.dnf_technical_collision_split_enabled` (default
`False` everywhere). Items 2 (per-track collision multiplier -- landed as a
dormant, default-1.0 optional field, see below), 3 (first-lap/grid term), and
4 (lap-of-retirement ordering) remain scoping only. Both split components
went through two rounds: v1 (grid-uniform technical, per-driver-EMA
collision) and v2 (real per-team technical evidence, driver-adaptive
collision), described below as the CURRENT state.

### Retirement-reason investigation (resolved)

The brief asked whether a mechanical-vs-collision retirement REASON is
available anywhere in the codebase before implementing the split for real.
Finding: **not in the stored actuals/backfill pipeline** --
`src/data/actual_results_fetcher.py::_result_row_is_dnf` collapses FastF1's
`Status` string to a plain boolean `dnf` flag immediately, and
`scripts/backfill_dnf_data.py` only ever re-fetches that same boolean. The
raw `Status` text (e.g. "Collision", "Accident", "Engine", "Gearbox",
"Hydraulics", "Retired", "Disqualified", ...) is never persisted past that
point.

It IS available one layer down: `src/systems/updater.py`'s
`_extract_dnf_drivers`/`_update_dnf_rate_ema` already receive the raw FastF1
`session_results` DataFrame (before any collapse) at season-state update
time, and its `Status` column still carries the real text. A keyword
classifier was built there (`_classify_dnf_status_reason`,
`_MECHANICAL_STATUS_KEYWORDS` / `_COLLISION_STATUS_KEYWORDS`) -- real FastF1
vocabulary, not fabricated. **Honest limitation:** a third bucket, "other"
(e.g. "Retired", "Disqualified", "Withdrawn", "Did not start" -- anything not
confidently matched to either keyword list), is never guessed and counts
toward NEITHER the technical nor the collision aggregation. This means both
v2 aggregations are undercounts relative to true mechanical/collision
incidence whenever a retirement's real cause doesn't surface in the Status
text clearly enough to keyword-match -- a real, uncorrected source of noise
in the per-team and per-driver evidence, on top of ordinary small-sample
variance.

### p_technical, v2: per-team, evidence-weighted

`_resolve_technical_dnf_probability` (`preparation_flow.py`) still starts
from the v1 grid-uniform decay prior (`floor + amplitude*exp(-races/tau)`,
uniform across every team by construction, races_completed-only). v2 adds a
Bayesian shrinkage toward the TEAM's own observed mechanical-retirement rate:
`(prior*k + observed*n)/(k+n)`, `k = dnf_technical_prior_strength` (default
3.0 -- "races-worth" of trust in the generic prior), `n =
team_technical_dnf_risk.races_observed`. A brand-new team (n=0, e.g. Cadillac
before its first race) always returns the pure prior regardless of `k` --
"starts at the high prior, decays/adapts as data arrives" holds exactly. This
is what lets Cadillac sit above a reliable team once evidence exists, which
v1's uniform curve could never express.

The observed rate is written by the new
`_update_team_technical_dnf_rate_ema` (`updater.py`, wired into both the
main-race and sprint-race update paths, alongside the existing per-driver
`_update_dnf_rate_ema` call): both of a team's observed cars pool into one
race-level MECHANICAL-only observed rate, EMA-blended into a stored rate with
its own (normally faster/shorter-half-life) blend weight
(`dnf_team_technical_update_blend`, default 0.35 vs. the driver EMA's 0.10) so
it follows reliability improving across a regulation-reset season. **Storage
decision:** rather than a new top-level "teams" section in
`driver_characteristics.json` (which would need new predictor-level
loading/wiring for a payload namespace that doesn't exist today), the value
is stored REDUNDANTLY under a new `team_technical_dnf_risk` key on EACH
teammate's own existing driver record, written identically for both
teammates every update. This reuses the exact plumbing
`_compute_driver_dnf_probability` already receives (`driver_data`) -- zero
new payload-loading path required -- and "team-level, shared by teammates" is
satisfied literally (both teammates always carry the byte-identical stored
value).

### p_collision, v2: driver-adaptive (the Verstappen arc)

`_resolve_driver_collision_crash_rate` (new, `preparation_flow.py`) replaces
v1's "raw per-driver EMA rate" input to `_resolve_collision_dnf_probability`
with a Bayesian shrinkage from an experience-tier crash PRIOR toward the
driver's OWN observed collision rate: `(prior*k + observed*n)/(k+n)`, `k =
dnf_collision_prior_strength` (default 5.0), `n =
collision_dnf_track_record.races_observed`. The tier priors reuse the
EXISTING `_EXPERIENCE_DNF_MODIFIERS` scale (rookie highest, established
lowest) layered on a new `dnf_collision_base_rate` (default 0.05) -- no new
per-tier config surface, per the "relate to the existing modifiers" mandate.
A rookie with no history (n=0) sits exactly at the rookie prior; a driver
with many collision-free observations converges toward their own near-zero
observed rate as n grows -- monotonically, by construction of the shrinkage
formula. This is the "young-crashy-Max becomes consistent-Max" arc: the
SAME driver's crash tendency adapts from a generic tier-based starting point
to their own demonstrated record, rather than being permanently anchored to
whatever the conflated all-cause EMA happened to say.

The per-driver observed rate is written by the new
`_update_driver_collision_track_record` (`updater.py`, same two call sites):
cumulative counts (`races_observed`, `collisions_observed`), not an EMA --
the shrinkage formula itself is the recency/evidence-weighting mechanism, and
cumulative counts are what make the blended crash rate converge cleanly
monotonically in `n`, which an EMA's own recency-decay would not guarantee in
the same simple sense.

### Combination and cap/floor (unchanged from v1)

`p_dnf = 1 - (1-p_technical)*(1-p_collision)` (`_combine_independent_dnf_risks`),
then the SAME final `[dnf_rate_floor, dnf_rate_final_cap]` clamp as the
pre-split model. `p_collision` still multiplies by the per-track
`collision_multiplier` (item 2, dormant/default-1.0 per-circuit optional
field in `data/processed/track_characteristics/*.json`, resolved via the
existing `overtaking_difficulty` circuit-key pattern) -- unchanged by v2.

### Byte-identical-when-off, still holds

Every new v2 field (`team_technical_dnf_risk`, `collision_dnf_track_record`)
is additive-only and absent from every driver record written before this
work and from every existing test fixture; both new resolvers fall through
to their v1 formulas whenever the field is missing or `races_observed <= 0`
-- no special-cased "off" value is needed for that guarantee, it falls out of
the shrinkage formula's own n=0 case. The master flag
(`dnf_technical_collision_split_enabled`) still defaults to `False`
everywhere and gates the entire split branch; with it off, the DNF path is
byte-identical to the pre-split model (proven by a 64-combo parametrized
equality test against a reference re-implementation of the old formula).

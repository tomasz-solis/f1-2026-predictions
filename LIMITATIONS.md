# Known Limitations

This document describes what the model does badly, what assumptions it relies
on, and what would need to change to address each gap. It is part of the
project because a model without an honest limits section is harder to trust,
not easier.

---

## 1. Weight schedule calibrated on a single regime change

What the model does: the signal-blending weight schedule determines how
fast the model shifts trust from pre-season baseline data toward current-season
race results. The current runtime uses `rapid_adaptive`; earlier reset-year
analysis also studied `extreme` against the 2021→2022 regulation reset - the
closest recent parallel to the 2026 reset.

The gap: that is one data point. The margin between `extreme` (0.809
Spearman correlation) and the next-best schedule `insane` (0.807) is 0.002
on a single season. Bootstrap resampling of the 2025 season shows the rank
ordering among the top-3 schedules is unstable across resample draws - the
"winner" changes depending on which races are in the sample.

What would fix it: a second major regulation change (post-2026 season)
would provide genuine out-of-sample evidence. Until then, treat the schedule
choice as a plausible prior, not a proven optimum. The schedule is fully
configurable - swap to `rapid_adaptive` or `moderate` in `config/default.yaml`
and compare using `scripts/backtest_2025_season.py`.

---

## 2. Pre-season constructor priors are assumptions, not measurements

What the model does: `src/data/data_generator.py` seeds the 2026 baseline
with constructor performance values derived from 2025 final standings
(e.g. McLaren 0.85, Mercedes 0.75). These are the starting point for race 1
predictions before any 2026 data exists.

The gap: the 2025 standings reflect a car that no longer exists after the
regulation reset. The priors could be substantially wrong - a team that was
third in 2025 might be first in 2026 (or last). The values carry explicit
uncertainty fields, but the uncertainty is uniform rather than tracking which
teams are likely to see the largest disruption.

What partially mitigates it: under the active `rapid_adaptive` schedule,
the baseline carries 35% weight at race 1, 8% at race 3, and 5% from race 4
onward. By mid-season it is effectively irrelevant. Races 1 - 2 predictions are
the most exposed.

What would fix it: a Bayesian prior over team disruption magnitude based
on regulation-change history (teams that led major rule changes have
historically held their advantage more often than not - but the variance is
high). This is a research direction, not a current implementation.

---

## 3. Pre-season testing sandbagging is not directly detected

What the model does: testing directionality signals (which teams improved
relative to last year, which look strong on high-stress configurations) are
extracted from pre-season running by `scripts/update_from_testing.py`. The
system treats these as directional, not absolute.

The gap: teams routinely run high fuel loads, restricted engine modes, or
entirely different suspension setups during testing to conceal pace. The model
has no way to distinguish "genuinely slow in testing" from "sandbagging". A
team running a 1.5s deficit in testing to hide a simulator-predicted
advantage will register as genuinely slower in the extracted signal.

What partially mitigates it: testing signals carry the lowest weight in
the blend (20% at race 1, 10% at race 2, 0% from race 3 onward). A badly
misleading testing signal self-corrects quickly once race data arrives.

What would fix it: explicit fuel-load correction using telemetry sector
times and pit-stop fuel data. FastF1 exposes some of this. It is a genuine
improvement that would require meaningful engineering work and domain
knowledge to implement correctly.

---

## 4. Compound performance needs minimum lap data before it applies

What the model does: per-team, per-circuit compound performance
adjustments are built from session lap data. They apply only when at least 8
laps per compound per team are available for a given circuit.

The gap: early in the season, before teams have visited circuits in 2026
conditions, compound adjustments fall back to neutral. This means predictions
for the first visit to each circuit rely entirely on overall team strength,
with no compound-specific tuning. High-deg circuits visited early in the
season (e.g. Bahrain at race 1) are most affected.

What would fix it: a compound-performance prior derived from 2025 data
that decays as 2026 circuit-specific data accumulates. This is more complex
than it sounds because compound characteristics change with the 2026 tyre
specifications - 2025 compound behaviour is a weakly informative prior at
best.

---

## 5. Rookie and mid-season driver substitution handling is conservative

What the model does: drivers with no 2026 race history fall back to team
baseline performance with a calibrated rookie uncertainty adjustment. The
uncertainty tier (`rookie`, `second_year`, `developing`) determines how much
the teammate gap is capped and how much the prediction shrinks toward team
mean.

The gap: the fallback is conservative by design - it avoids wild
predictions - but it will systematically underestimate a genuinely
exceptional rookie for the first few weekends. Similarly, a mid-season
substitution (injury replacement) starts from the same fallback regardless
of the substitute's historical track record, unless their characteristics
are manually updated in the driver characteristics file.

What would fix it: a career-level prior over driver performance that
initialises from historical results across all seasons, decayed by time. The
`scripts/extract_driver_characteristics.py` script could be extended to pull
this automatically rather than requiring manual maintenance.

---

## 6. Backtest validation uses a single historical season

What the model does: `scripts/backtest_2025_season.py` evaluates the
predictor against 2025 season data as a proxy for out-of-sample performance.
A temporal train/test split (first 70% of races = train, last 30% = test) is
used to measure generalization.

The gap: 2025 was a single, largely stable season without a major
regulation change. Performance on 2025 data may not predict performance on
2026 data, which is the regime the model was specifically designed for. The
model has not yet been evaluated on live 2026 race results - those will
accumulate across the season.

What would fix it: live evaluation as 2026 races complete. The
`make evaluation-report` command generates `docs/MODEL_CALIBRATION.md` from
saved predictions and actuals - re-running it after each race produces an
improving picture of true in-distribution performance.

---

## 7. Safety car and lap-1 incident priors are partly track-specific, but still heuristic

What the model does: race simulation now reads circuit metadata from
`data/processed/track_characteristics/2026_track_characteristics.json`,
including per-track `safety_car_prob`, track type, and overtaking difficulty.
Those values are used during backtesting and live race simulation instead of
falling back to one global safety-car probability for every circuit.

The gap: this is still a curated prior, not a learned incident model.
The values are not yet re-estimated automatically from historical safety-car
and lap-1 incident data, and the lap-1 chaos logic is still mostly driven by
global config scales. So the model is more circuit-aware than it used to be,
but it is not yet evidence-driven enough to claim principled incident
probabilities.

What would fix it: build a small extraction pipeline that derives
per-circuit safety-car and opening-lap incident rates from historical race
data, then compare those learned priors against the current manual settings
in season backtests before promoting them.

---

## 8. Monte Carlo intervals are measured and adaptively widened, but still under-calibrated

What the model does: each driver's predicted position carries a p5 - p95
interval from the Monte Carlo distribution, and the learning system now stores
interval residual history from completed predictions. Once enough samples
accumulate, qualifying and race outputs can widen those intervals with a
learned minimum radius rather than trusting the raw simulation spread alone.

The gap: the latest generated calibration report still shows the model is
too confident. On 2026-04-20, `docs/MODEL_CALIBRATION.md` reports 80.3%
empirical qualifying coverage against a 90.0% target across 66 driver-race
intervals from 3 completed events. So the measurement loop exists, and the
adaptive widening path exists, but the live evidence is still thin and the
saved predictions remain under-covered.

What would fix it: keep collecting races, replay the updated interval
learning path through historical backtests, and only then tighten the default
radius or target settings if the empirical coverage stays close to 90% across
a materially larger sample.

---

## 9. Experimental model components can look useful until stacked

What the model does: reset-year research can evaluate testing-derived team
seeds, residual models, and conformal interval calibration as separate
components. The evaluation script now applies an executable promotion gate and
movement diagnostics before a component is treated as stackable.

The gap: a component can improve one headline metric while making the
overall model worse. The clearest current risk is residual stacking: a
testing-derived team seed and residual correction can both encode similar
information, so applying both can overcorrect. Mean metrics can also hide broad
per-weekend damage.

What partially mitigates it: residual models are skipped by default when
the active team seed is `testing_model`, unless an ablation explicitly opts in.
Promotion gates require central-MAE improvement, no winner-accuracy drop, no
large top-3 drop, and no broad weekend-level degradation. Movement diagnostics
count whether challenger predictions moved drivers closer to or farther from
actual positions.

What would fix it: rerun guarded ablations after each component change and
promote only components that pass the gate across holdouts and live slices.
Qualifying residuals need direction diagnostics before clip tuning; if they
often move drivers the wrong way, smaller clips only hide a bad signal.

---

## 10. DNF probability is honest about existence, not yet about magnitude

What the model does: every race prediction carries a per-driver
`dnf_probability` built from historical retirement rates, driver experience,
and team uncertainty, then realised through Monte Carlo DNF sampling. The
dashboard shows it as a "DNF Risk %" column and the evaluation report scores
it with a Brier score against actual retirements.

The gap: against the low observed 2026 retirement rate, the raw output
overforecasts risk. The 2026 probe
(`data/model_diagnostics/2026/dnf_calibration_probe.md`, 13 race events,
286 driver observations, 11 DNFs) scores the emitted probabilities at
Brier 0.046 while shrinking them three-quarters of the way toward the season
base rate scores 0.037. The sample is small, so per-driver risk ranking may
still be informative even though the absolute magnitudes run high.

What partially mitigates it: an output-layer shrinkage knob
(`baseline_predictor.race.dnf_probability_shrinkage_lambda` with
`dnf_probability_base_rate`) can recalibrate the *reported* probability
without touching the simulation inputs. It defaults to 1.0 (no change) until
the probe evidence is confirmed on a larger sample.

What would fix it: keep scoring races, and if the overforecast persists,
recalibrate the simulation-input DNF rates themselves (historical caps,
floors, and experience modifiers) through the promotion-gate workflow, since
that changes finish-order predictions.

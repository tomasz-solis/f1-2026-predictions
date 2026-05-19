# Teammate-Network Prior Design

Date: 2026-05-09
Status: locked for Phase 3 extractor implementation

This document defines the historical teammate-network prior used by the
driver-rating de-carring work. It also defines the matched-lap observation
contract shared by the prior builder and the live updater.

The companion gates are filled and locked:

- `docs/fixes/teammate_network_prior_validation_evidence.md`
- `docs/fixes/matched_lap_extractor_smoke_sessions.md`

Current gate state as of 2026-05-12:

- Phase 1 source-backed magnitude rows are filled in the validation evidence
  doc, using the accepted Motorsport / PACETEQ source-family rule.
- Phase 2 smoke-session rows are locked.
- Phase 3 extractor code can start from this contract.

Direction-only checks are not validation evidence. Source-backed magnitude
checks are. The validation evidence file now gives the prior fit an acceptance
criterion, and the smoke-session file now gives the extractor smoke tests
locked expected behavior.

## 1. Purpose

The active-season orthogonality contract separates four signals:

- `team_strength`: shared car/team package pace.
- `race_rating_mu_s`: driver-specific dry race residual in seconds.
- `quali_rating_mu_s`: driver-specific dry qualifying residual in seconds.
- `wet_skill`: wet-vs-dry delta in teammate-relative pace.

Race and qualifying ratings are within-team residuals. A same-session teammate
residual can say "driver A was faster than driver B"; it cannot prove where
both drivers sit on a grid-wide driver scale. Cross-team identification comes
from the historical teammate-network prior defined here.

The prior is built offline, validated, versioned, and read by the live
system. It is not re-fit during a season.

## 1.5 Within-Session Update Ordering

The orthogonality contract depends on a procedural rule that this doc
must state explicitly. Driver residuals within a session are computed
against the **observed** same-session team median, not against any
version of `team_strength` (prior or posterior). The team_strength
state is updated from observed team-vs-field evidence and is then used
for **future** prediction and trace; it is not the subtraction anchor
for same-session driver residuals. Subtracting posterior team_strength
would let team-model error and any seconds-mapping error leak into
`race_rating_mu_s`.

Within a completed session:

1. build cleaned, comparable per-driver lap observations;
2. update `team_strength` from observed team-vs-field evidence; this
   record is for trace and for future-session prediction, not for the
   same-session driver subtraction;
3. compute driver residuals from the **observed same-session team
   median**:
   `driver_residual_s = observed_team_median_s - observed_driver_median_s`
   (or the equivalent paired teammate-gap formulation per Section 4.3);
   do not compute driver residuals against `predicted_team_strength` or
   against the posterior team_strength state;
4. update `race_rating_mu_s`, `quali_rating_mu_s`, or `wet_skill`
   according to session-type and weather routing rules.

The team_strength posterior is consumed at prediction time
(`predicted_team_seconds + driver_rating_mu_s`) and surfaced in trace.
It is never the same-session subtraction anchor. An implementation that
subtracts predicted or posterior team_strength from absolute driver
pace is disallowed by this contract.

## 1.6 Active-Season Learning Policy

The system should keep learning during the season, but the main in-season
learner is `team_strength`, not large driver-skill movement.

After each completed race weekend, the runtime update flow recomputes or
updates team strength from observed team-vs-field evidence. That updated
team state feeds the next prediction checkpoint. This is the mechanism that
lets the model learn that a car package has improved, regressed, or changed
track-type behavior during the season.

Driver skill in seconds is intentionally slower-moving:

- within-season race and qualifying driver ratings may update from clean
  teammate-relative evidence, but with conservative noise and shrinkage;
- one weekend should not cause a dramatic driver-skill jump unless the
  evidence is repeated and clean;
- full teammate-network rejudgment is an offline operation, normally after
  the season, during a season break, or on demand during a longer pause such
  as summer break.

This is why Section 1.5 updates `team_strength` before computing
same-session driver residuals. Team strength captures shared car movement for
future predictions, while same-session driver residuals stay anchored to the
observed team median so car error does not leak into driver skill.

Implementation note (2026-05-19): active-season learning is allowed only through
the replay/update path. The 2026 live artifacts are rebuilt from completed
session data, including FP/practice, sprint, race, qualifying, and explicit
`Status` rows for DNF-rate learning. The seed artifacts are not hand-edited to
make a driver or team "look right"; if the replay output conflicts with domain
expectation, the next step is to audit the input construct or updater rule.

## 1.7 Known Limits

These are not blockers; they are open risks that should sit visibly so
they are not rediscovered as bugs later.

**Shared driver improvement is entangled with car improvement at the
within-team level.** Within a single team-session or active-season
same-team update, shared driver improvement is structurally
indistinguishable from car improvement and will be absorbed by
`team_strength`. The historical teammate-network prior can anchor
cross-team driver scale through driver moves and multi-year teammate
links, so cross-era driver quality is partly identifiable. What the
live updater **cannot** do is infer from same-team residuals alone that
both current teammates improved together — that conclusion requires
evidence the within-team observation does not contain. Contract tests
asserting "team_strength moves and ratings stay neutral when only the
car changed" are therefore evaluable in synthetic data only. Real-data
validation is necessarily indirect: per-driver residual diagnostics,
replay stability across per-season folds, and sensitivity to the
declared extractor and calibration parameters
(`min_matched_pairs_race`, `min_matched_pairs_quali`,
`max_position_change_for_clean_lap`, `traffic_stint_sigma_threshold`,
`tire_age_fallback_window_laps`, observation-noise SE floors).

**Validation evidence may over-index on large gaps.** The current
validation candidate set (see validation evidence file Section 3) is
biased toward clear large-margin pairings. Tight teammate gaps are
where small bias matters most for race-outcome prediction. Phase 1 of
the master execution plan includes a deliberate search for small-gap
source-backed checks. If no defensible small-gap source surfaces,
the validation report records the gap as a validation limitation —
that is a statement about validation strength, not about model
uncertainty. Initial sigma on close pairings is widened only if
**internal diagnostics** support it: weak posterior evidence, low
effective sample size on the relevant teammate edge, fragile
component connectivity, high sensitivity in per-fold replay, or
unstable per-driver residuals across folds. The decision is documented
either way — widened with the diagnostic that justified it, or held at
the fitted value with the limitation logged.

## 2. Scope

In scope for this document:

- FastF1 historical session scope for the prior.
- Canonical matched-lap extractor specification.
- Race and qualifying network model specification.
- Connected-component handling.
- Robust weighting and uncertainty rules.
- Output artifact contract.
- Prior-validation gates.

Out of scope (owned by other docs/tasks):

- Schema migration for `race_rating_*` and `quali_rating_*` fields.
- Live updater rewrites.
- Wet-skill model implementation.
- Prediction-time blending.
- Replay harness implementation details.
- Confidence calibration of the live Bayesian update.
- The implementation of `team_strength_to_seconds()`.
- Test rewrite plan.

This doc records the blockers those downstream tasks must satisfy, but it
does not own them. The execution plan in `docs/fixes/master_execution_plan.md`
sequences them and assigns ownership.

## 3. Historical Scope

Use Formula 1 sessions from 2022 through 2025 for the first prior build.

Reasons:

- 2022 is the first season of the previous technical era.
- 2023 and 2024 provide settled-rules evidence.
- 2025 is the most recent pre-2026 evidence.
- Pre-2022 would mix in a different car generation and would require an
  explicit era covariate the v1 prior is not modelling.
- 2026 is a regulation reset; it is used as a transfer-risk check, not a
  prior-selection source.

Build separate historical observation sets for:

- race-pace network;
- qualifying-pace network.

Wet and mixed sessions are excluded from dry race/quali networks. They belong
to the separate `wet_skill` path.

Sprint and sprint-qualifying sessions are excluded from the v1 prior. They
can be added later as separate, lower-confidence evidence.

Race-weekend practice sessions are excluded from the offline
teammate-network prior fit because run programs are not comparable enough for
source-backed driver residuals. They are **not** excluded from the live
prediction system. FP1/FP2/FP3 remain valid runtime evidence for car features,
session pace blending, confidence, and team-strength updates when the local
feature logic marks them usable.

Pre-season testing is a separate case. It can provide car-feature hints and
uncertainty signals, but it must not directly impose a manufacturer order.

## 4. Matched-Lap Extractor Spec

The matched-lap extractor is the canonical observation pipeline for both the
historical prior builder and the live updater.

### 4.1 Function Contract

```python
def extract_matched_teammate_laps(
    session: fastf1.core.Session,
    *,
    session_kind: Literal["race", "qualifying"],
    weather_mode: Literal["dry", "wet", "mixed", "unknown"],
    config: MatchedLapConfig,
) -> pd.DataFrame:
    """Return canonical teammate matched-lap observations for one session."""
```

For valid evidence, the extractor returns one row per matched lap pair, not
one row per driver. Rows use a stable reference/comparison driver order so
the fitter cannot count both sides of the same evidence as independent
observations.

For a team/pair that is skipped before any matched rows are produced, the
extractor returns one diagnostic row with `row_type = "skipped_pair"`,
nullable lap fields, and a non-empty `skip_reason`. This keeps skip
accounting out of implementation guesswork and visible in the diagnostic
dump.

Required raw matched-pair columns:

- `row_type`
- `year`
- `race_name`
- `session_name`
- `session_kind`
- `team`
- `reference_driver_code`
- `comparison_driver_code`
- `reference_lap_number`
- `comparison_lap_number`
- `reference_lap_time_s`
- `comparison_lap_time_s`
- `matched_gap_s`
- `compound`
- `reference_stint`
- `comparison_stint`
- `stint_lap_index`
- `weather_bucket`
- `track_status_bucket`
- `reference_position_start`
- `reference_position_end`
- `comparison_position_start`
- `comparison_position_end`
- `skip_reason`

`row_type` is one of:

```text
matched_pair
skipped_pair
```

Sign convention:

```text
matched_gap_s = comparison_lap_time_s - reference_lap_time_s
```

Positive means the reference driver was faster than the comparison driver.

Reference driver ordering must be deterministic. Use driver code alphabetical
order unless the implementation has a stronger stable identifier. The chosen
ordering has no model meaning; it only prevents duplicate evidence.

### 4.2 Configuration

```yaml
matched_laps:
  min_matched_pairs_race: 8
  min_matched_pairs_quali: 3
  max_position_change_for_clean_lap: 2
  traffic_stint_sigma_threshold: 1.5
  tire_age_fallback_window_laps: 3
```

`min_matched_pairs_quali` is 3 in v1. The standard error of a median is not
meaningful with two points. If a future version decides to accept `n=2`, it
must define a separate SE rule (for example, half the absolute pair spread
plus a forced sigma floor) and document the change.

All thresholds live in config so they can be sensitivity-tested without code
changes.

### 4.3 Race Matching Rules

Race observations use per-pair matched-lap differences, not "median of driver
laps minus median of teammate laps". This avoids imbalance when the two
drivers have different comparable-lap coverage.

Valid race lap pairs must satisfy:

- same team;
- same compound;
- same stint-lap index;
- green-flag lap where track-status data is available;
- all weather samples inside each lap interval agree with the target weather
  bucket;
- neither lap is pit-in or pit-out;
- neither lap is lap 1;
- neither lap is the final classified lap for that driver;
- neither lap is deleted, inaccurate, missing `LapTime`, or a statistical
  outlier;
- neither driver changes more than `max_position_change_for_clean_lap`
  places during the lap;
- neither lap exceeds that driver's stint median by more than
  `traffic_stint_sigma_threshold * stint_sigma`.

Primary matching is same compound and same stint-lap index. The tire-age
fallback may be used only if strict matching does not produce enough valid
pairs **and** the smoke-session checks (Section 12.2) show that the fallback
is not adding strategic noise.

If the teammate has fewer than the required matched samples, both drivers
are skipped for driver-rating updates in that session. A surviving driver
does not receive a race-rating update when the teammate DNF leaves no valid
comparison sample.

### 4.4 Qualifying Matching Rules

Qualifying observations use comparable dry push laps from common qualifying
segments both teammates reached.

Rules:

- Identify all common segments both teammates reached.
- Process common segments from highest to lowest: Q3, then Q2, then Q1.
- Pair valid push laps within the same `(segment, compound)` by run order,
  not by lap-time rank. Do not pair a driver's best lap against the
  teammate's second run unless the run order actually matches.
- Aggregate matched pairs across common segments until the sample reaches
  `min_matched_pairs_quali`.
- Use only valid dry laps for dry `quali_rating_mu_s`.
- Exclude deleted, inaccurate, missing, pit, and non-green laps where
  available.
- Require at least `min_matched_pairs_quali` matched pairs.
- If common segments do not produce enough comparable laps, skip the pair.
- Wet or mixed qualifying does not update dry qualifying ratings.

### 4.5 Weather Routing

Lap-level weather classification uses FastF1 `session.weather_data`
timestamps mapped onto each lap interval.

Rules:

- If all weather samples inside the lap interval are dry, classify the lap
  as dry.
- If all samples indicate rainfall, classify the lap as wet.
- If samples are mixed, missing, or cannot be mapped reliably, classify the
  lap as `lap_level_mixed_unreliable`.
- Dry race/quali ratings use dry laps only.
- Wet-skill observations use wet laps only in v1.
- Lap-level mixed/unreliable laps feed neither dry race/quali rating nor
  `wet_skill` in v1.

Session-level mixed events may update dry signals only from reliable dry
laps. If lap-level routing is unreliable for the session, skip dry race/quali
updates and allow reduced-confidence wet-skill extraction only when enough
clean wet evidence exists.

### 4.6 Canonical Skip Reasons

The extractor must use stable skip strings. Trace and tests assert on these
strings rather than ad hoc text.

```text
single_car_session
team_driver_set_ambiguous
teammate_dnf_no_matched_laps
weather_routing_excludes_session
lap_level_weather_unreliable
insufficient_matched_pairs
no_compound_overlap
no_common_quali_segment
all_laps_filtered_out
missing_lap_time_data
track_status_excluded_all_laps
```

`lap_level_weather_unreliable` covers laps where weather samples are mixed,
missing, or cannot be mapped to the lap interval. It is distinct from
`weather_routing_excludes_session`, which is a session-level routing
decision.

## 5. Observation Aggregation

The prior fitter consumes one aggregated row per teammate pair per
team-session. It does not consume two mirrored driver rows. Mirrored driver
rows are not independent evidence; they are the same constraint with
opposite sign.

For each reference/comparison pair:

```text
matched_gap_median_s = median(matched_gap_s over matched lap pairs)
```

The fixed-effects model consumes:

```text
y_i = matched_gap_median_s
y_i = theta_reference - theta_comparison + epsilon_i
```

Positive `y_i` means the reference driver was faster than the comparison
driver.

Required aggregated columns:

- `reference_driver_code`
- `comparison_driver_code`
- `team`
- `year`
- `race_name`
- `session_name`
- `session_kind`
- `matched_gap_median_s`
- `matched_gap_se_s`
- `n_matched_pairs`
- `weather_bucket`
- `skip_reason`

The live updater may derive midpoint-relative driver observations from the
same aggregate:

```text
reference_driver_delta_s = matched_gap_median_s / 2
comparison_driver_delta_s = -matched_gap_median_s / 2
```

These derived rows are used only for live driver-rating updates; they are
**not** used as independent observations in the prior fit. The prior fitter
sees one row per teammate pair.

Estimate `matched_gap_se_s` by bootstrap over matched lap pairs. If the
bootstrap is unstable because the sample is small, mark the observation as
low confidence or skip it; do not silently emit a tiny SE.

## 6. Model Spec

Build separate models for race and qualifying observations.

For one aggregated teammate observation:

```text
y_i = theta_reference - theta_comparison + epsilon_i
```

Where:

- `y_i` is `matched_gap_median_s`;
- positive means the reference driver was faster;
- `theta_driver` is the driver's latent residual skill in seconds;
- `epsilon_i` is observation noise.

Fit all historical teammate constraints jointly with a sum-to-zero
constraint inside each connected component:

```text
sum(theta_d for d in component) = 0
```

The output rating is stored in seconds:

- `race_rating_mu_s`
- `race_rating_sigma_s`
- `quali_rating_mu_s`
- `quali_rating_sigma_s`

Positive `mu_s` means faster than the active-grid average driver residual.

Use weighted least squares with heteroskedasticity-aware errors and cluster
bootstrap by session/team-pair. A Bayesian hierarchical model is a future
upgrade path, not v1.

## 7. Connected Components

Before fitting, build the teammate graph:

- nodes: drivers;
- edges: drivers who have valid matched teammate observations;
- edge weights: valid aggregated observation count.

Run connected-components analysis before fitting.

Decision tree:

- If one component covers at least 90% of observations and at least 80% of
  active/relevant drivers, treat it as the main anchored component.
- Fit the main component with the normal sum-to-zero constraint.
- Small components are neutral-centered and assigned inflated uncertainty
  (Section 8).
- If multiple large components exist, stop. Do not relax thresholds just to
  force connectivity. Inspect whether the extractor is too strict, whether
  the historical scope is too narrow, or whether the graph is genuinely
  disconnected.
- Threshold relaxation is allowed only if extractor smoke checks show the
  strict rules are dropping valid comparable laps.

Recentring active-season driver ratings is housekeeping for asymmetric
missingness. It is not cross-team identification. Cross-team scale comes
from the historical teammate network.

## 8. Weighting And Uncertainty

Observation weights reflect both sample size and observation quality:

```text
weight_i = capped_effective_n_i / max(matched_gap_se_s_i^2, se_floor^2)
```

Where:

- `capped_effective_n_i` caps the influence of sessions with many matched
  laps;
- `matched_gap_se_s_i` comes from matched-pair bootstrap;
- `se_floor` prevents implausibly tiny SEs from dominating the fit.

Default sigma settings:

```yaml
prior:
  race_sigma_floor_s: 0.05
  quali_sigma_floor_s: 0.10
  min_driver_observations: 24
```

`min_driver_observations` defaults to 24, derived as
`3 * min_matched_pairs_race`. With the default race threshold of 8, that is
roughly three valid race-session equivalents per driver before the
main-component bootstrap sigma is trusted without escalation. The derivation
lives in prose; the YAML is a literal value so the config parses
unambiguously.

For main-component drivers:

```text
if n_driver_observations < min_driver_observations:
    sigma_d = max(1.75 * population_sd(theta_hat_main_component),
                  configured_floor)
else:
    sigma_d = max(
        bootstrap_sigma_d,
        0.5 * population_sd(theta_hat_main_component),
        configured_floor,
    )
```

For small components or unanchored drivers:

```text
small_component_sigma  = max(1.75 * population_sd(theta_hat_main_component),
                             configured_floor)
unanchored_sigma       = max(2.00 * population_sd(theta_hat_main_component),
                             configured_floor)
```

The reference for fallback uncertainty is the population spread of anchored
driver skills, not the posterior sigma tail of well-connected drivers.

## 9. Output Artifact

Path:

```text
data/processed/teammate_network_prior/{built_at}.json
data/processed/teammate_network_prior/latest.json
```

Required top-level structure:

```json
{
  "built_at": "2026-05-09T12:00:00Z",
  "config": {
    "historical_scope": {"start": 2022, "end": 2025},
    "matched_lap_config_race": {},
    "matched_lap_config_quali": {},
    "bootstrap_replicates": 1000,
    "race_sigma_floor_s": 0.05,
    "quali_sigma_floor_s": 0.10,
    "min_driver_observations": 24
  },
  "race_network": {
    "drivers": {},
    "components": [],
    "fit_diagnostics": {}
  },
  "quali_network": {
    "drivers": {},
    "components": [],
    "fit_diagnostics": {}
  },
  "validation": {
    "source_backed_checks": [],
    "all_hard_checks_passed": false
  }
}
```

Driver entries must include:

- `mu_s`
- `sigma_s`
- `n_observations`
- `n_teammate_partners`
- `component_id`
- `component_anchored`
- `first_session`
- `last_session`

`mu_s` is in seconds, signed so positive means faster than the component
mean. `sigma_s` is the bootstrap or fallback uncertainty, also in seconds.

## 10. Seconds Scale And Team Mapping

Driver ratings are stored in seconds natively. There is no
`driver_rating_to_seconds()` mapping in v1.

There should not be a default v2 plan to add `driver_rating_to_seconds()`.
Adding that mapping would make driver ratings less interpretable and could
hide model error inside a second conversion layer. Revisit this only if
held-out replay shows a stable, repeatable nonlinearity in driver residuals
that cannot be explained by team strength, weather, track type, or input
quality.

The forward model for dry sessions is:

```text
observed_driver_to_field_s =
    observed_field_median_s - observed_driver_median_s

predicted_driver_to_field_s =
    team_strength_to_seconds(session_kind, team_strength)
    + driver_rating_mu_s
```

Only team strength requires a mapping. The driver side is fixed by
construction.

Decision: use separate race and qualifying team-strength seconds mappings.
Race pace and qualifying pace are different constructs with different
variance, tire state, traffic, and fuel-load behavior. A shared mapping would
look simpler but would force two different signals through one slope.

Calibration equations:

```text
race_observed_driver_to_field_s - race_rating_mu_s
    ~ race_team_strength_centered

quali_observed_driver_to_field_s - quali_rating_mu_s
    ~ quali_team_strength_centered

race_team_strength_centered = race_team_strength - 0.5
quali_team_strength_centered = quali_team_strength - 0.5
```

If the stored team-strength artifact has only one `team_strength` value in
the first migration step, fit two mappings over that same scalar:

```text
race_observed_driver_to_field_s - race_rating_mu_s
    ~ team_strength_centered

quali_observed_driver_to_field_s - quali_rating_mu_s
    ~ team_strength_centered

team_strength_centered = team_strength - 0.5
```

The current v1 decision is to keep that one stored scalar rather than split it
into separate short-run and long-run team-strength states. A conventional-
weekend support probe over the currently cached 2022-2025 FP rows did not show
the required accuracy gain from the split:

- row-weighted combined MSE:
  - `shared_long_run = 0.5049`
  - `split_short_quali_long_race = 0.5077`
- the split beat the best shared policy in only two of four combined held-out
  folds;
- the split beat the best shared policy in zero qualifying folds.

That is evidence against adding state now, not evidence that qualifying and race
car behavior are literally the same. Keep the state simple until the split wins
on the actual prediction objective.

Fit the mappings once on historical data and freeze them for the model
version. Do not continuously refit them during the active season. In-season
learning updates team-strength state; it does not rewrite the mapping slope
unless a deliberate model-version recalibration is run.

## 10.5 Wet-Skill Calculation Method

Wet skill is a teammate-relative seconds delta that measures how much a driver
gains or loses in wet conditions compared with their dry race baseline. It is
not a finishing-position adjustment and it is not a global "good in rain"
rating detached from teammate evidence.

For a wet matched-lap aggregate:

```text
wet_gap_s = theta_reference_wet - theta_comparison_wet

dry_expected_gap_s =
    race_rating_mu_s(reference) - race_rating_mu_s(comparison)

wet_skill_observation_s = wet_gap_s - dry_expected_gap_s
```

Positive `wet_skill_observation_s` means the reference driver gained pace in
the wet relative to the dry expectation. The wet-skill updater consumes one
aggregate row per teammate pair, using the same reference/comparison ordering
as the dry extractor.

For prediction, the wet component is applied only when the race/weather
context says wet evidence should matter:

```text
predicted_driver_to_field_s =
    team_strength_to_seconds("race", team_strength)
    + race_rating_mu_s
    + wet_context_weight * wet_skill_mu_s
```

`wet_context_weight` is 0 for dry conditions, 1 for fully wet conditions, and
a documented fractional value for mixed forecasts or mixed live sessions. The
implementation must record the chosen weight and weather source in trace.

Fully wet sessions update `wet_skill` only. They must not update dry race or
qualifying ratings. Mixed sessions may update dry ratings from reliable dry
laps and wet skill from reliable wet laps; mixed, missing, or unreliable lap
intervals feed neither path.

## 11. Regulation-Reset Monitoring

Freezing the seconds scale assumes the historical relationship between
team-strength units and seconds is usable for 2026. This is a stability
assumption, not a guarantee.

The same rule applies to any future short-run/long-run state split. The
pre-2026 conventional-weekend probe is only support evidence because it sits
before the regulation reset and its historical per-season counts reflect local
FP cache coverage as well as sprint-weekend history. The decisive transfer-era
question is whether 2026 conventional weekends show a consistent MSE gain under
the new rules. Until that evidence exists, the v1 model keeps one stored
team-strength state and separate race/qualifying seconds mappings.

Trace diagnostics must monitor:

- rolling correlation between predicted and observed team deltas in seconds;
- slope of `observed_team_delta_s ~ predicted_team_delta_s`;
- R-squared versus the historical fit R-squared;
- per-driver residual means.

These diagnostics should be surfaced in the main Streamlit app on a separate
monitoring tab, not only written to offline reports. The tab should show the
current rolling window, historical reference band, latest slope/R-squared,
driver residual outliers, and whether the current model version is inside the
expected range. The dashboard reads these diagnostics from the same persisted
artifact path and Supabase tables used by background jobs; it must not compute
an incompatible version ad hoc.

If 2026 shows sustained scale drift, the remedy is a one-time
between-season or model-version refit, not continuous in-season refitting.

## 12. Validation Gates

Two validation gates must be filled and locked with real content before
extractor implementation starts.

### 12.1 Source-Backed Magnitude Checks

The hard validation table lives in:

```text
docs/fixes/teammate_network_prior_validation_evidence.md
```

A check is "hard" only when every required field is filled:

- a named driver comparison;
- session/range scope;
- source-backed threshold in seconds;
- source citation (URL or exact reference);
- source type;
- pass/fail rule;
- date accessed.

Gut-feel magnitude checks do not enter the validation report. Direction-only
checks (sign of teammate gap) are unit tests in `tests/test_prior_signs.py`,
not validation evidence.

Current state: locked on 2026-05-12, then amended on 2026-05-17 after the
construct audit. The validation doc contains 13 external PACETEQ context rows,
one supplemental near-zero row, no same-construct HARD rows yet, and documented
cuts.

### 12.2 Extractor Smoke Sessions

The extractor smoke-session lock lives in:

```text
docs/fixes/matched_lap_extractor_smoke_sessions.md
```

The smoke set must name the actual sessions before coding and must include:

- one clean dry race;
- one wet or mixed race;
- one teammate DNF or insufficient-sample case;
- one strategy-asymmetric race;
- one representative qualifying session.

For each session, record expected checks before coding:

- approximate matched-pair count range;
- expected sign and rough magnitude if an independent source exists;
- expected weather buckets;
- expected skip reasons;
- known compound-overlap or strategy complication.

If no independent source exists for approximate magnitude, the session can
test counts and skip reasons only. It must not become a magnitude
validation claim.

These are smoke checks for extractor correctness, not full validation of
the prior.

Current state: locked on 2026-05-12.

## 13. Bulk Extraction Diagnostic Dump

After bulk extraction and before fitting, write a diagnostic report with:

- `n_matched_pairs` distribution by season, session, session kind, and
  team;
- `matched_gap_median_s` distribution;
- `matched_gap_se_s` distribution;
- skip-reason counts;
- sessions with zero non-skipped observations;
- teammate-pair coverage counts;
- connected component summary;
- weather-bucket counts;
- compound-overlap counts.

If the SE distribution has implausibly tiny values, cap or winsorize before
WLS so a few sessions do not dominate the fit.

## 14. Calibration Splits

Use per-season folds for scale calibration and replay validation:

```text
train on three seasons, validate on the fourth;
rotate across 2022, 2023, 2024, and 2025.
```

Do not use random race holdout as the main split. Races inside a season
share lineups, car characteristics, tire behavior, and technical context.

Per-season folds expose season-to-season slope drift, which is useful for
regulation-reset risk.

## 15. Orthogonality Acceptance

After the new pipeline exists, held-out replay should show:

- shared team movement is absorbed by `team_strength`;
- teammate-specific movement is absorbed by `race_rating_mu_s` or
  `quali_rating_mu_s`;
- wet-only advantage is absorbed by `wet_skill`;
- dry race/quali ratings do not move in fully wet sessions.

For the dry leakage diagnostic:

```text
corr(delta_race_rating_mu_s, delta_team_strength_for_driver_team)
```

Measure this metric in v1, but do not treat it as a passed hard check until
a post-fix replay establishes a defensible threshold. Do not use current
`rating_mu` as a statistical baseline; it is a different object with
different units.

For wet leakage (hard invariant):

- fully wet sessions must produce zero dry race/quali rating updates;
- mixed sessions with dry-lap updates should have
  `abs(corr(wet_skill, delta_race_rating_mu_s)) <= 0.20`.

## 16. Replay Diagnostics

Replay must include per-driver residual diagnostics:

```text
driver_residual_mean_s =
    mean(observed_driver_to_field_s - predicted_driver_to_field_s)
```

A sustained non-zero mean for one driver is evidence that the driver's
prior rating is biased or weakly identified. This should flag the driver
rating or sigma for review. It should not trigger a v1 driver-side
shrinkage parameter.

The replay harness implementation lives in a downstream doc; this section
records the diagnostic the prior contract requires.

## 17. Migration Relationship

This prior design does not own the schema migration, but the migration
must happen in this order:

1. schema accepts old and new fields;
2. writers write new fields;
3. readers prefer new fields and fall back to old fields;
4. local and Supabase artifacts are migrated with rollback snapshots;
5. old fields and fallback paths are removed only after validation.

`_DRIVER_BAYESIAN_SCHEMA` currently has `additionalProperties: False`, so
the schema must accept the new fields before any writer persists them.

Implementation must update the whole project, not just the immediate writer.
Readers, validators, warmup jobs, dashboard rendering, evaluation reports,
checkpoint reconstruction, local artifact stores, and Supabase persistence
must all prefer the new artifacts once the migration phase reaches reader
cutover. Deprecated fields may exist only as temporary fallback during the
declared migration window. Supabase tables and local JSON artifacts must carry
the same field names, units, and version metadata.

## 18. Implementation Order Within This Doc

This doc owns steps 1 through 7. Steps 8 onward are downstream
dependencies; they live in the master execution plan
(`docs/fixes/master_execution_plan.md`).

1. Lock this design doc after the companion validation gates are filled.
2. Fill the source-backed magnitude evidence file. Done on 2026-05-12.
3. Fill the named extractor smoke-sessions file. Done on 2026-05-12.
4. Implement the matched-lap extractor from this spec. This is the first
   code-changing step.
5. Run extractor smoke checks on the locked validation sessions.
6. Run the 2022-2025 bulk extraction.
7. Review the extraction diagnostic dump and fit race/quali
   teammate-network priors.

Live updater changes are deliberately last.

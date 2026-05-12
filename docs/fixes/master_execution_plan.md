# Master Execution Plan: Driver-Rating De-Carring And Race/Quali Split

Date: 2026-05-09
Status: locked through Phase 2; Phase 3 extractor implementation can start

This plan sequences all the work from the orthogonality contract through
final rollout. It is the authoritative ordering. Individual design docs own
their phase; this plan owns the dependencies between phases.

Two non-negotiable rules:

- **Live updater changes are last.** Every prior phase exists so that the
  live updater is changed only when the new prior, mappings, schema, and
  diagnostics are in place. Reordering this surfaces bugs in production
  state that are then expensive to migrate out of.
- **No phase starts until its predecessor's acceptance criteria are met.**
  This includes Phase 0. A doc that still has TODO entries in lock-required
  fields is not closed.

Current local gate state as of 2026-05-12:

- Phase 1 source-backed evidence rows are filled using the accepted
  Motorsport / PACETEQ source-family rule.
- Phase 2 smoke sessions are locked.
- Phase 3 extractor implementation is the first model-path code change.

---

## Phase 0 — Documentation Lock Gates

Goal: make the design assumptions visible and reviewable before any code is
written.

Tasks:

- Revise the prior design doc with all agreed fixes
  (`docs/fixes/teammate_network_prior.md`).
- Create the validation evidence scaffold
  (`docs/fixes/teammate_network_prior_validation_evidence.md`).
- Create the smoke-session scaffold
  (`docs/fixes/matched_lap_extractor_smoke_sessions.md`).

Explicit blockers carried forward:

- Extractor code can start now that Phase 1 and Phase 2 are closed.
- No prior fit until Phase 5 bulk extraction closes.
- Use separate race and qualifying team-strength seconds mappings in Phase 7.

Acceptance criteria:

- Prior design doc lists all agreed conventions: `min_matched_pairs_quali =
  3`, prior fitter consumes one row per teammate pair (not mirrored driver
  rows), `lap_level_weather_unreliable` skip reason exists, scope reconciled
  against downstream docs, YAML config uses concrete values.
- Validation scaffold exists with required columns and lock rules.
- Smoke-session scaffold exists with named candidate sessions and lock
  rules.

Dependencies: none.

---

## Phase 1 — Source-Backed Validation Evidence

Goal: build the only thing that can grade the prior fit honestly.

Tasks:

- Research external race and qualifying teammate deltas. Candidate sources
  may include F1Metrics-style posts, broadcast race-pace summaries, F1TV
  pace charts, or other published analysis, but a source is accepted only
  if it satisfies the source acceptance criteria in Section 1 of
  `docs/fixes/teammate_network_prior_validation_evidence.md`. No
  blanket approval by outlet; the methodology is judged per artifact.
- F1Metrics-style sources are SUPPLEMENTAL only for v1. They may corroborate
  another timing source but never count toward HARD row totals.
- Motorsport.com / Motorsport-Total PACETEQ teammate pace articles are
  HARD-capable for v1 when they state the construct, sample scope, and
  numeric seconds delta. Treat them as one accepted source family, not
  independent corroboration of each other.
- Convert defensible numbers into hard checks with required fields filled
  (source URL, source type, threshold in seconds, pass rule, date accessed).
- Cut weak checks instead of soft-grading them.

Acceptance criteria:

- At least 3 race checks and 2 quali checks filled with real sources, OR
  the validation report explicitly labels quali coverage as provisional and
  documents the compensation (wider initial sigma, stricter replay
  diagnostics).
- Every filled row has all required fields.
- Cut rows are documented with cut reason.
- Conservative audit calls from Section 2 of the validation evidence file
  hold for v1: Tsunoda/de Vries stays cut from hard validation; Leclerc/
  Sainz multi-season quali stays out of hard validation. Either may return
  only if a source provides a clean single-season numeric threshold.
- File status changes to "locked".

Dependencies: Phase 0.

---

## Phase 2 — Smoke-Session Lock

Goal: lock the extractor smoke set so step 4 is not gut-graded.

Tasks:

- Pick exact sessions per category. Replace candidates if they do not fit.
- Fill expected matched-pair count ranges. Use ranges, not exact numbers.
- Fill expected weather buckets and expected skip reasons.
- Mark magnitude expectations only when externally sourced; otherwise
  leave magnitude blank and document that the session tests counts and
  routing only.

Allowed pre-extractor exception (read-only FastF1 inspection):

- Ad-hoc notebooks or scripts that **load** FastF1 sessions, **inspect**
  weather samples, lap counts, retirement laps, track status events,
  and team participation, and **emit** human-readable summaries used
  to fill expected behavior, are allowed under Phase 2.
- Saved exploration outputs (CSV/JSON/notebook) are retained as
  evidence for the locked expected behavior.
- Not allowed under this exception: writing matching logic (same
  compound, same stint-lap index, run-order pairing, common-segment
  processing), skip-reason emission, weather-routing logic, or
  anything resembling the extractor's output schema. Those are Phase
  3 work.
- Test for whether code is allowed: could the exploration output be deleted
  after Phase 2 closes without losing anything Phase 3 needs to implement?
  If yes, it's exploration. If no, it's the extractor and belongs in
  Phase 3.

Acceptance criteria:

- All five categories filled.
- Each row's status changes from `TODO_COUNTS` to `LOCKED`.
- Any read-only FastF1 inspection used to fill expected behavior is
  retained as evidence (notebook or summary CSV/JSON), not discarded.
- File status changes from its draft/pre-lock status to "locked".

Dependencies: Phase 0.

---

## Phase 3 — Matched-Lap Extractor

Goal: implement the canonical extractor exactly as specified.

Lap-level weather-routing rule (also referenced from Phase 8 and Phase
11):

- all FastF1 weather samples inside the lap interval indicate dry → lap
  may feed dry race/quali updates;
- all samples indicate rainfall → lap may feed `wet_skill`;
- samples are mixed, missing, or cannot be reliably mapped → lap feeds
  neither in v1; emit `lap_level_weather_unreliable`.

Session-level routing applies on top: fully wet sessions never update
dry race/quali ratings (hard invariant, Phase 8).

Tasks:

- Implement `extract_matched_teammate_laps()` per Section 4 of the prior
  design doc.
- One row per matched lap pair using the deterministic
  reference/comparison driver ordering.
- Skipped pairs emit one diagnostic row with `row_type = "skipped_pair"`
  and a non-empty `skip_reason`.
- Use only canonical skip strings.
- Race and quali matching rules implemented per spec; tire-age fallback
  gated behind smoke-session evidence.
- Bootstrap SE estimation for `matched_gap_se_s`.
- Docstrings on every public function, including the sign convention.
- Unit tests for sign convention, skip-reason emission, weather routing
  (per the rule above), and minimum-pair gating.
- Direction-only smoke tests live in `tests/test_prior_signs.py`, not in
  the validation report.

Acceptance criteria:

- All unit tests pass.
- Code review confirms one row per pair, not per driver.
- No magic skip-reason strings; all come from the canonical list.

Dependencies: Phases 0, 1, 2.

---

## Phase 4 — Extractor Validation

Goal: verify the extractor on the locked smoke sessions before bulk run.

Tasks:

- Run the extractor on each Phase 2 smoke session.
- Compare matched-pair counts against locked expected ranges.
- Verify weather routing and emitted skip reasons.
- For sessions with externally sourced magnitudes, verify the median
  matched gap sign and rough magnitude.
- Fix extractor bugs until smoke checks pass.

Hard rule: do not tune extractor rules to force desired prior outputs.
Tuning is allowed only when smoke evidence shows strict rules are dropping
valid comparable laps.

Acceptance criteria:

- All smoke sessions pass count and routing expectations.
- Any deviation has a documented explanation that survives review.

Dependencies: Phases 0, 1, 2, 3.

---

## Phase 5 — Bulk Historical Extraction

Goal: produce the canonical historical observation set.

Tasks:

- Run race and qualifying extraction across 2022-2025.
- Produce a parquet (or equivalent) cache.
- Produce the diagnostic dump per Section 13 of the prior design doc:
  `n_matched_pairs` distribution, SE distribution, skip-reason counts,
  zero-observation sessions, teammate-pair coverage, weather buckets,
  compound overlap, connected-component summary.

Acceptance criteria:

- Bulk run completes without crashes.
- Diagnostic dump is reviewed and any anomalies (silent FastF1 metadata
  gaps, unexpected zero-observation sessions, implausibly tiny SEs) are
  resolved before fitting.

Dependencies: Phase 4.

---

## Phase 6 — Teammate-Network Prior Builder

Goal: fit the race and qualifying priors.

Tasks:

- Run connected-components analysis first; apply the decision tree from
  Section 7 of the prior design doc.
- Fit race and qualifying networks separately with WLS and cluster
  bootstrap.
- Apply the population-SD fallback sigma rule for weak-evidence and
  unanchored drivers.
- Validate against the locked validation evidence table (Phase 1).
- Produce the output artifact per Section 9.

Acceptance criteria:

- Hard validation checks evaluated and reported. Failures are documented
  with source threshold and observed value.
- Connected-component output matches the decision tree expectations.
  Multiple-large-components scenario triggers stop-and-inspect, not
  threshold relaxation.
- Direction-only unit tests pass and are reported separately, not as
  validation evidence.

Dependencies: Phases 1, 5.

---

## Phase 7 — Team-Strength Seconds Mapping

Goal: calibrate the only fitted seconds conversion in the system.

Decision: use separate race and qualifying team-strength seconds mappings.
Race pace and qualifying pace are different constructs, so one shared slope
would mix different variance and fuel/traffic regimes.

The v1 calibration window (2022-2025 per-season folds) and the v1
calibration form are settled by this plan. If the first migration still has
one stored `team_strength` scalar, fit separate race and qualifying mappings
over that same scalar.

Tasks:

- Fit separate race and qualifying `team_strength_to_seconds()` mappings.
- Use 2022-2025 per-season folds (train on three of 2022-2025,
  validate on the fourth; Section 14 of prior design doc).
- Validate the combined sum prediction
  `predicted_team_seconds + driver_rating_mu_s` against observed
  driver-to-field seconds on held-out folds.
- Run a regulation-reset transfer check: 2026 leave-one-race-out
  cross-check against the calibrated mapping, with the 2024-2025
  portion emphasized as the nearest pre-reset transfer-risk window.
  The 2026 LOO output is a diagnostic, never a fit input. If the
  2026 estimate sits outside the one-standard-error band of the
  2024-2025 portion:
    - inflate early-2026 sigma on the affected drivers/teams;
    - document the transfer risk in the validation report;
    - decide between accepting wider intervals for 2026 and a one-time
      between-version refit. No continuous in-season refit.
- Freeze the mapping for the model version.

Acceptance criteria:

- Held-out per-season fold validation reports R-squared, slope, and
  per-driver residual means.
- Combined sum prediction RMSE is reported and reviewed.
- 2026 LOO transfer check executed and result documented; if material
  disagreement was found, the documented response (sigma inflation,
  risk note, refit-or-accept decision) is in the validation report.
- No continuous in-season refitting hooked up.

Dependencies: Phase 6.

---

## Phase 8 — Replay And Leakage Diagnostics

Goal: verify orthogonality before changing live state.

Tasks:

- Implement per-season-fold replay using the new prior and frozen
  mapping.
- Compute per-driver residual means (Section 16 of prior design doc).
- Compute the dry leakage diagnostic
  `corr(delta_race_rating_mu_s, delta_team_strength_for_driver_team)` and
  report it. Do not yet treat as a hard pass/fail; the post-fix replay
  establishes the threshold.
- Persist and surface regulation-reset monitoring in the main Streamlit app
  on a separate diagnostics tab:
  - rolling correlation between predicted and observed team deltas in
    seconds;
  - slope of `observed_team_delta_s ~ predicted_team_delta_s`;
  - R-squared versus the historical fit R-squared;
  - per-driver residual means.
- Hard wet-leakage invariant: fully wet sessions produce zero dry
  race/quali rating updates. Mixed sessions with dry-lap updates have
  `abs(corr(wet_skill, delta_race_rating_mu_s)) <= 0.20`. The
  underlying lap-level weather-routing rule (dry / wet / unreliable
  classification) is canonical in Phase 3 and is not redefined here.

Acceptance criteria:

- Per-driver residual means are reported. Drivers with sustained non-zero
  residuals are flagged for sigma review, not auto-corrected.
- Dry leakage diagnostic is measured.
- Dashboard diagnostics tab reads the persisted monitoring artifact or
  Supabase rows. It must not compute a separate ad hoc definition.
- Wet leakage hard invariant is satisfied.

Dependencies: Phase 7.

---

## Phase 9 — Schema Migration

Goal: add the new fields safely without breaking existing reads.

Tasks (in order):

- Schema accepts old and new fields (`additionalProperties: True` for the
  bayesian sub-block, or explicit allow-list).
- Writers begin writing `race_rating_*` and `quali_rating_*` fields.
- Readers prefer new fields and fall back to old fields.
- Snapshot rollback artifacts: write old artifact contents to a versioned
  backup before the first migration run.
- Migrate local artifacts and Supabase rows.
- Apply the migration across the whole project: readers, validators, warmup
  jobs, dashboard rendering, checkpoint reconstruction, evaluation reports,
  local artifact stores, and Supabase persistence must all prefer the new
  fields at reader cutover.
- **K=3 removal rule (canonical):** old fields and fallback reader paths
  are removed only after **3 consecutive completed race weekends** using
  the new path with no validation regression flagged in the per-weekend
  trace. The counter starts when the new path reaches production, not
  when Phase 14 begins. If a regression is flagged inside the 3-weekend
  window, the counter resets and the issue is investigated before any
  further removal attempt. Removal is a **separate change set / separate
  release step** (not bundled with any other refactor), executed only under
  this rule. Other phases reference this rule rather than restating it.

Acceptance criteria:

- Schema validation passes for both old-only, new-only, and mixed
  artifacts.
- Rollback artifact exists and is verified loadable.
- Reader fallback path covered by tests.

Dependencies: Phase 8.

---

## Phase 10 — Bayesian Race/Quali State Split

Goal: separate the previously-shared Bayesian state into race and
qualifying.

Sprint multipliers are the accepted v1 rule (not a deferral):

- `sprint_race_confidence = 0.5 * race_rating_confidence`;
- `sprint_quali_confidence = 0.5 * quali_rating_confidence`.

Rationale: fewer laps, compressed strategy, smaller sample. The 0.5
multiplier is a v1 default, not a calibrated value; revisit only if
sprint sample size is large enough to support a separate calibration.

The only branch is whether existing sprint code reaches production. The
April 2026 review flagged `update_from_sprint_race` as unwired dead
code; before Phase 10 implementation begins, verify the current state
of both the sprint race path and the sprint qualifying path:

- **Wired path (sprint update reaches production state and
  persistence):** implement the 0.5 confidence multipliers and add unit
  tests covering sprint update behavior, sprint-vs-main confidence
  ratio, and the race/quali state isolation requirement.
- **Unwired path (sprint code exists but isn't called by the live
  pipeline):** document explicitly that sprint updates remain
  production-unwired for this release, file a named follow-up task to
  wire them, and do not silently treat sprint as "handled." The 0.5
  rule still stands as the v1 design choice for whenever the wiring
  lands.

There is no third option (silent deferral without explanation).

Tasks:

- Two independent `BayesianDriverRanking` instances: one for race, one
  for quali.
- Separate restore, update, and persist paths.
- Race state is never clobbered by quali state and vice versa.
- Verify sprint path wiring; implement multipliers or document
  production-unwired status with follow-up task per the rule above.

Acceptance criteria:

- Unit tests confirm race and quali states evolve independently.
- A McLaren 1-2 race no longer moves both McLaren drivers' qualifying
  ratings.
- Sprint paths are verified. Wired sprint paths use the 0.5 confidence
  multipliers and are tested. Unwired sprint paths are documented as
  production-unwired with a named follow-up task. No third state.

Dependencies: Phase 9.

---

## Phase 11 — Wet-Skill Migration

Goal: move wet-skill to the same teammate-relative lap-time observation
basis as the dry ratings.

Tasks:

- Replace position-based wet-skill observations with lap-time
  teammate-relative observations using the canonical lap-level
  weather-routing rule from Phase 3 (dry samples → dry signals; rain
  samples → `wet_skill`; mixed/missing/unreliable → neither in v1).
- Store wet skill in seconds. Compute each wet observation as the observed
  wet teammate gap minus the dry race-rating expected teammate gap.
- Apply `wet_context_weight * wet_skill_mu_s` at prediction time. The weight
  is 0 for dry, 1 for fully wet, and documented as a fractional value for
  mixed forecasts or mixed live sessions.
- Wet and mixed sessions update `wet_skill`. Fully wet sessions never update
  dry race/quali ratings.
- Position-scale modifiers must not compose with seconds-scale ratings.

Acceptance criteria:

- Wet-skill update path uses matched-lap observations, not finishing
  position.
- Hard invariant from Phase 8 (wet sessions do not update dry ratings)
  remains satisfied.
- Composition of wet-skill with race/quali rating at prediction time is
  unit-tested, including trace output for `wet_context_weight`.

Dependencies: Phase 10.

---

## Phase 12 — Retire Or Demote Duplicate EMAs

Goal: stop double-counting through legacy `race_pace` and `quali_pace`
EMAs.

Tasks:

- Decide the role of `race_pace` and `quali_pace`: retire as prediction
  signals, or demote to diagnostics-only.
- Remove their effect from the prediction blend if retired.
- If demoted, document that they exist for trace/debugging only and have
  no influence on outputs.

Acceptance criteria:

- No prediction signal double-counts driver-team-conditioned pace through
  both the Bayesian rating and an EMA.
- Trace output continues to expose pace-EMA values for debugging if
  demoted.

Dependencies: Phase 10.

---

## Phase 13 — Test Rewrite

Goal: bring the test suite into alignment with the new contract.

Tasks:

- Per-file decisions: mechanical rename only, semantic rewrite, or
  deletion.
- The McLaren 1-2 test (and equivalents) splits into two assertions:
  shared movement is absorbed by `team_strength`; the driver-rating
  delta reflects only the teammate-relative gap.
- Tests previously asserting both McLaren drivers' rating goes up after
  a 1-2 finish are rewritten or deleted; they encoded the old absolute-
  form interpretation and are now wrong.
- Sign-flip and direction tests live in `tests/test_prior_signs.py` and
  similar; they are smoke tests, not validation evidence.

Acceptance criteria:

- Full test suite passes against the new contract.
- No test silently relies on the old interpretation.
- Direction-only tests are clearly tagged and not counted in validation
  reports.

Dependencies: Phase 12.

---

## Phase 14 — Final Rollout

Goal: validate end-to-end before promoting to production.

Tasks:

- Run the full replay against the latest pipeline.
- Compare calibration metrics against the pre-change baseline (using
  comparable post-fix quantities, not legacy `rating_mu`).
- Inspect trace residuals for leftover leakage.
- Promote only after hard validation checks pass and smoke checks pass.
- Old fields and fallback reader paths are removed only under the
  Phase 9 K=3 removal rule. Phase 14 does not restate or override that
  rule.

Acceptance criteria:

- All hard validation checks from Phase 1 pass, or failures are
  explicitly accepted with documented reasoning.
- Wet-leakage hard invariants from Phase 8 hold.
- Per-driver residual diagnostics are reviewed; flagged drivers have
  documented sigma adjustments or accepted bias notes.

Dependencies: Phases 1 through 13.

---

## Cross-Phase Notes

**Things this plan deliberately does not include:**

- A "soft validation" tier. Magnitude checks are either source-backed
  hard checks or unit-test smoke checks. Nothing in between.
- A driver-side shrinkage parameter in the seconds calibration. If
  per-driver residuals show systematic bias (Phase 8), the response is
  to flag the rating, not introduce another fitted parameter that
  obscures it.
- An in-season prior refit. The prior is built once per model version.
  Regulation-reset drift triggers a between-version refit, not a live
  one.

**Phase boundaries that are easy to violate:**

- Phase 3 (extractor implementation) without Phase 1 and 2 closed:
  produces an extractor with nothing to validate against.
- Phase 6 (prior fit) without Phase 1 closed: produces a fit graded only
  by direction-only smoke tests, which is the failure mode this plan
  exists to prevent.
- Phase 10 (state split) without Phase 9 (schema migration) closed:
  writes new fields with no schema support.
- Phase 14 final removal of old fields without Phases 1-13 acceptance:
  removes the rollback path before the new path is verified.

**Ownership notes:**

- The prior design doc owns Phases 0-7 design content.
- The team-strength mapping doc owns the Phase 7 pre-phase blocker
  decision.
- The schema migration doc (to be created if not yet) owns Phase 9
  details.
- The replay harness doc owns Phase 8 implementation; Phase 8 acceptance
  criteria here are the contract.

If a phase's owner doc does not exist yet, that doc must be created
before its phase starts. The lack of a doc is itself a blocker.

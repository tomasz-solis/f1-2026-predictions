# Teammate-Network Prior Validation Evidence

Date: 2026-05-09
Status: locked

This file records hard, source-backed validation checks and non-gating
external evidence for the teammate-network prior. It exists to prevent
post-fit rationalisation.

A row is not a hard validation check until every required field is filled:

- `source_url` or exact source reference;
- source type;
- source-backed threshold in seconds;
- pass/fail rule;
- date accessed;

Direction-only checks do not belong here. Put sign-flip smoke tests in code,
for example `tests/test_prior_signs.py`.

## 1. Source Acceptance Criteria

A source is judged on methodology, not reputation. The same outlet can
publish a defensible number in one article and a marketing number in
another. Apply the rules below per artifact, not per outlet.

### 1.1 Accepted

A source is accepted for hard validation only if it satisfies all of:

- it publishes a numeric teammate gap in seconds (or a per-lap delta that
  can be converted to seconds without an undocumented coefficient);
- it states the construct: race pace, qualifying pace, single-lap pace,
  long-run pace, or session-best;
- it states the sample basis: which sessions, how laps were filtered,
  whether outliers or pit laps were removed;
- it is teammate-relative or can be made teammate-relative without
  re-running the source's model inside this project;
- the construct matches what the prior estimates (see Section 1.4 on
  construct mismatch).

Examples of source classes that *can* clear this bar, evaluated case by
case, not blanket-approved:

- broadcast race-pace charts that quote teammate seconds-per-lap deltas
  with the lap window stated (e.g. "laps 15-30 of stint 2");
- independent analytical posts that publish their methodology and a
  numeric teammate seconds delta; model-derived driver-rating projects are
  governed by Section 1.3.5;
- academic or technical writeups that publish teammate-relative pace
  deltas with the cleaning rules described.

### 1.2 Conditionally Accepted

A source is conditionally accepted when the construct or scope partly
matches but requires translation. Use the row only if the translation is
mechanical and documented.

Examples:

- a published per-race teammate qualifying delta aggregated into a season
  median: acceptable if the aggregation method is written into the row's
  notes;
- a season-summary chart with seconds deltas but no explicit lap-filtering
  description: acceptable if the prior-side construct can be matched to the
  source's window (e.g. compare against the prior fitted on green-flag
  long-run laps if that is what the source approximates);
- a source that reports the gap as a percentage (e.g. "0.42% slower"):
  acceptable if converted using a documented reference lap time.

### 1.3 Rejected

A source is rejected for hard validation if any of the following hold:

- the methodology is opaque or unstated;
- the source reports a derived rating on its own scale rather than a
  seconds delta (e.g. an Elo-style number, a 0-100 rating);
- the source is itself a model that estimates the same quantity the prior
  estimates, with no independent observational basis;
- the source is a marketing artifact (the AWS "Fastest Driver" rankings,
  team PR, sponsor-branded analytics) where the construct is undocumented;
- the source is memory, "common F1 knowledge", or social media commentary;
- the source is this project's own matched-lap extractor or any future
  output of the prior being validated;
- the source is an AI assistant's synthesis or summary rather than the
  underlying artifact.

### 1.3.5 F1Metrics-Style Sources Rule

F1Metrics and similar independent driver-rating projects publish numbers
that look like validation evidence but partly overlap with what this prior
estimates. The rule for v1:

- F1Metrics-style sources are conditionally acceptable when they
  publish a numeric teammate seconds delta with enough methodology to
  understand the construct (which sessions, which laps, what filters).
- They are not acceptable as the sole hard threshold when the
  published value is a model-derived driver rating on its own scale rather
  than a numeric teammate seconds delta with documented methodology.
- They may corroborate another timing source (broadcast charts, F1TV
  graphics with named lap windows) or support rough magnitude when paired
  with a clearly independent observation.
- If F1Metrics is the only source for a row, mark the row conditionally
  accepted in the notes, record the circularity risk explicitly, and do
  not treat the threshold as load-bearing in the validation report.

This rule applies to any independent driver-rating project that estimates
the same quantity this prior estimates. F1Metrics is named because it is a
likely candidate source; the rule is not F1Metrics-specific.

Decision (2026-05-12) - F1Metrics tightening. F1Metrics-style
sources are SUPPLEMENTAL only, never HARD. They are reported
separately per Section 6 and do not count toward the Section 7 lock
rule's "at least 3 race + 2 quali HARD checks" requirement. A row
whose only available source is F1Metrics remains CUT unless an
independent acceptable source surfaces. This decision overrides the
"conditionally acceptable as sole source" framing earlier in this
section: F1Metrics may *corroborate* a HARD source, but it is never
itself a HARD source.

### 1.3.6 Motorsport / PACETEQ Source Family Rule

Decision (2026-05-12, amended 2026-05-17). Motorsport.com and
Motorsport-Total articles that publish PACETEQ teammate pace deltas are useful
external timing-analysis evidence, but they are not HARD-capable for the
current v1 extractor unless the article proves a same-construct match.

The amendment follows the construct audit:

- the current qualifying extractor measures a multi-run matched-lap median,
  while the best-documented PACETEQ qualifying rows use best qualifying times;
- the current race extractor measures a tightly paired same-compound,
  same-stint-lap residual, while the PACETEQ race articles do not document an
  equivalent pairing rule.

PACETEQ rows therefore remain visible as `EXTERNAL_CONTEXT`, not acceptance
gates, until genuinely aligned evidence is found.

Limits:

- multiple PACETEQ-backed articles are the same source family, not
  independent corroboration of each other;
- rows must record the specific article and value used, not just cite
  "Motorsport";
- very small gaps, especially around 0.01s/lap, are not load-bearing HARD
  checks even if the source family is accepted;
- conflicting non-PACETEQ sources should be noted when they materially
  affect threshold choice.

### 1.3.7 Construct Audit

Decision (2026-05-17). The available PACETEQ race and qualifying rows are
no longer HARD acceptance gates for the current v1 extractor.

The qualifying mismatch is direct: the current local qualifying construct is a
matched multi-run median over comparable quick laps, while the best-documented
2024 PACETEQ rows use best qualifying times.

The race mismatch is less explicit but still not proven safe: the current local
race construct is a paired residual under same-compound and same-stint-lap
controls, while the available PACETEQ articles do not document an equivalent
lap-control contract.

Those rows remain useful external context, but they are now classified as
`EXTERNAL_CONTEXT` until the project either:

- changes `quali_rating_mu_s` to mean peak qualifying skill; or
- sources same-construct evidence for the current race and qualifying
  extractors.

### 1.4 Construct Mismatch Categories

Most cuts come from construct mismatch, not source quality. The five
patterns to watch for:

- race pace vs qualifying pace. A driver may be 0.15s/lap faster than
  the teammate over long runs but 0.30s/lap faster on a single push lap.
  The prior estimates these separately. A row that uses qualifying
  evidence to validate the race prior is mismatched.
- teammate-relative vs global. The prior is a teammate-relative
  residual. A source that reports "VER is the fastest driver in F1 by
  0.4s" is a global statement. It does not pin the VER-PER teammate gap.
- single-season vs multi-season aggregate. A row that says
  `verstappen_perez_race_2022_2024` aggregates three seasons of car
  development, Pérez's documented 2024 form drop, and 2022 vs 2023 vs 2024
  reliability. Most published deltas are season-by-season. Aggregating
  them across seasons requires assumptions the source did not make.
- model-derived rating vs published timing delta. A source that runs
  its own driver-rating model and publishes the rating outputs is partly
  circular as validation: it is doing what the prior is doing. Prefer
  sources that publish raw timing deltas.
- opaque broadcast number vs documented method. Sky/F1TV graphics
  often quote teammate seconds-per-lap during a stint, but the underlying
  filter (which laps, what tire age) is rarely visible. Acceptable only
  if the broadcast also names the lap window and the prior-side construct can
  match it.

### 1.5 Promotion Discipline

Candidate links may be collected in the scratch file, but only this document
defines the validation set. Once a row is promoted here, the source URL,
source type, threshold, pass rule, date accessed, and cut/supplemental status
must be explicit. Scratch notes are not validation evidence.

## 2. Candidate Check Audit

Each candidate from the original scaffold is reviewed against the
criteria in Section 1. Action codes:

- `KEEP` - row is well-scoped; proceed to source research.
- `SPLIT` - scope is too broad; split into per-season rows or narrower
  comparisons before sourcing.
- `NARROW` - scope is too aggressive; tighten the relationship or the
  sample window.
- `QUALITATIVE` - magnitude unlikely to be sourceable; convert to a
  unit-test smoke check (sign-only) and remove from the validation set.
- `CUT` - sample too thin or construct mismatch unlikely to be fixable;
  remove the row entirely.

### 2.1 Race Candidates

`verstappen_perez_race_2022_2024`

- Action: SPLIT.
- Reasoning: 2022-2024 aggregates three different cars, Perez's documented
  2024 form drop, and changing reliability. Most race-pace sources are
  per-season.
- Replacement rows: `verstappen_perez_race_2022`,
  `verstappen_perez_race_2023`, `verstappen_perez_race_2024`.

`alonso_stroll_race_2023_2024`

- Action: SPLIT.
- Reasoning: same multi-season aggregation issue. The 2023 and 2024 Aston
  Martin contexts differ enough that the gap should be sourced by season.
- Replacement rows: `alonso_stroll_race_2023`,
  `alonso_stroll_race_2024`.

`albon_sargeant_race_2023_2024`

- Action: SPLIT after research.
- Reasoning: no clean two-season aggregate source was found. The per-season
  numbers are sourceable and cleaner.
- Replacement rows: `albon_sargeant_race_2023`,
  `albon_sargeant_race_2024`.

`russell_latifi_race_2022`

- Action: CUT.
- Reasoning: the row is factually wrong. Russell drove for Mercedes in 2022,
  while Latifi's Williams teammate was Albon.
- Replacement rows: none.

`bottas_zhou_race_2022_2024`

- Action: SPLIT, then mostly cut.
- Reasoning: same multi-season problem. Phase 1 research found no defensible
  2022 race source, a 2023 direction conflict, and only a near-zero 2024
  Bottas advantage.
- Replacement rows: cut `bottas_zhou_race_2022`; cut or reframe
  `bottas_zhou_race_2023`; keep `bottas_zhou_race_2024` as supplemental
  only.

`tsunoda_devries_race_2023`

- Action: CUT.
- Reasoning: De Vries ran about 10 races. Even season-scoped, the sample is
  too thin for a hard magnitude check.
- Replacement rows: move any sign-only check to `tests/test_prior_signs.py`.

### 2.2 Qualifying Candidates

`verstappen_perez_quali_2022_2024`

- Action: SPLIT.
- Reasoning: same multi-season concern as the race version. Qualifying deltas
  are usually published per season.
- Replacement rows: `verstappen_perez_quali_2022`,
  `verstappen_perez_quali_2023`, `verstappen_perez_quali_2024`.

`leclerc_sainz_quali_2022_2024`

- Action: NARROW or QUALITATIVE.
- Reasoning: direction is contested in some seasons. The original hedged
  relationship is not a hard magnitude check.
- Replacement rows: use a single clear season if sourced, or move a
  direction-only check to `tests/test_prior_signs.py`.

`russell_hamilton_quali_2024`

- Action: KEEP.
- Reasoning: single season, clear construct, and sourceable published
  qualifying deltas.
- Replacement rows: keep `russell_hamilton_quali_2024`.

`albon_sargeant_quali_2023_2024`

- Action: SPLIT after research.
- Reasoning: Albon had a clear qualifying advantage, but clean numeric
  sources are per-season rather than a two-season aggregate.
- Replacement rows: `albon_sargeant_quali_2023`,
  `albon_sargeant_quali_2024`.

### 2.3 Audit Summary

After applying the audit and the 2026-05-12 Phase 1 source research pass:

- Race: 7 PACETEQ rows remain as `EXTERNAL_CONTEXT` after the 2026-05-17
  construct audit; none currently counts as a HARD gate for the paired race
  extractor. Three Bottas-Zhou rows are cut or supplemental; the
  Russell-Latifi 2022 row is cut as an impossible pairing/year; Tsunoda-De
  Vries remains cut.
- Quali: 6 PACETEQ rows remain as `EXTERNAL_CONTEXT`; none currently
  counts as a HARD gate for the multi-run qualifying extractor. Leclerc-Sainz
  remains smoke-only because the original relationship was hedged and
  contested.

The filled Section 3 tables are now the source of truth for Phase 1 status.
The scratch file at `docs/fixes/phase_1_source_research.md` remains research
history only, not validation evidence.

## 3. Revised Candidate Tables

These tables replace the original Candidate Race Checks and Candidate
Qualifying Checks tables. They reflect the audit and the 2026-05-12
Phase 1 source research pass.

Each row also acquires an `evidence_tier` value during Phase 1, per
Section 6:

- HARD - counts toward pass/fail in the validation report;
- EXTERNAL_CONTEXT - useful external evidence that does not yet match the
  fitted construct closely enough to gate the fit;
- SUPPLEMENTAL - F1Metrics-style or partly model-derived; reported
  separately, does not count toward pass/fail;
- SMOKE_ONLY - direction-only smoke check; does not appear in the
  validation report;
- CUT - researched and rejected; cut reason recorded.

The evidence_tier is recorded in the Status column. HARD rows count toward
the validation report; EXTERNAL_CONTEXT, SUPPLEMENTAL, and CUT rows do not.

### 3.1 Race Candidates (Post-Audit)

`verstappen_perez_race_2022`

- Status: EXTERNAL_CONTEXT.
- Scope: Red Bull race pace, 2022 only.
- Expected relationship: VER faster than PER.
- Threshold: `0.234s/lap`.
- Source: Motorsport.com / PACETEQ Perez trend.
- Source type: teammate race-pace delta.
- Pass rule: `VER_mu_s - PER_mu_s >= 0.234`.
- Date accessed: 2026-05-12.
- Notes: replaces the 2022-2024 aggregate.

`verstappen_perez_race_2023`

- Status: EXTERNAL_CONTEXT.
- Scope: Red Bull race pace, 2023 only.
- Expected relationship: VER faster than PER.
- Threshold: `0.451s/lap`.
- Source: Motorsport.com / PACETEQ 2023 review.
- Source type: teammate race-pace delta.
- Pass rule: `VER_mu_s - PER_mu_s >= 0.451`.
- Date accessed: 2026-05-12.

`verstappen_perez_race_2024`

- Status: EXTERNAL_CONTEXT.
- Scope: Red Bull race pace, 2024 only.
- Expected relationship: VER faster than PER.
- Threshold: `0.56s/lap`.
- Source: Motorsport-Total / PACETEQ Red Bull duel.
- Source type: teammate race-pace delta.
- Pass rule: `VER_mu_s - PER_mu_s >= 0.56`.
- Date accessed: 2026-05-12.
- Notes: exact full-season value appears in article text.

`alonso_stroll_race_2023`

- Status: EXTERNAL_CONTEXT.
- Scope: Aston Martin race pace, 2023 only.
- Expected relationship: ALO faster than STR.
- Threshold: `0.486s/lap`.
- Source: Motorsport.com / PACETEQ 2023 review.
- Source type: teammate race-pace delta.
- Pass rule: `ALO_mu_s - STR_mu_s >= 0.486`.
- Date accessed: 2026-05-12.
- Notes: replaces the 2023-2024 aggregate.

`alonso_stroll_race_2024`

- Status: EXTERNAL_CONTEXT.
- Scope: Aston Martin race pace, 2024 only.
- Expected relationship: ALO faster than STR.
- Threshold: `0.25s/lap`.
- Source: Motorsport-Total / PACETEQ Aston Martin duel.
- Source type: teammate race-pace delta.
- Pass rule: `ALO_mu_s - STR_mu_s >= 0.25`.
- Date accessed: 2026-05-12.
- Notes: exact full-season value appears in article text.

`albon_sargeant_race_2023`

- Status: EXTERNAL_CONTEXT.
- Scope: Williams race pace, 2023 only.
- Expected relationship: ALB faster than SAR.
- Threshold: `0.293s/lap`.
- Source: Motorsport.com / PACETEQ 2023 review.
- Source type: teammate race-pace delta.
- Pass rule: `ALB_mu_s - SAR_mu_s >= 0.293`.
- Date accessed: 2026-05-12.
- Notes: split from the unsourced 2023-2024 aggregate.

`albon_sargeant_race_2024`

- Status: EXTERNAL_CONTEXT.
- Scope: Williams race pace, 2024 Sargeant sample.
- Expected relationship: ALB faster than SAR.
- Threshold: `0.38s/lap`.
- Source: Motorsport-Total / PACETEQ Williams duel.
- Source type: teammate race-pace delta.
- Pass rule: `ALB_mu_s - SAR_mu_s >= 0.38`.
- Date accessed: 2026-05-12.
- Notes: covers Sargeant's 2024 Williams starts before the driver change.

`russell_latifi_race_2022`

- Status: CUT_IMPOSSIBLE_PAIRING_YEAR.
- Reason: Russell drove for Mercedes in 2022; Latifi's Williams teammate was
  Albon.

`bottas_zhou_race_2022`

- Status: CUT_NO_NUMERIC_RACE_SOURCE.
- Reason: targeted Phase 1 search found head-to-head and qualifying context,
  but no defensible numeric race-pace delta.

`bottas_zhou_race_2023`

- Status: CUT_DIRECTION_CONFLICT.
- Source checked: Motorsport.com / PACETEQ 2023 review.
- Reason: the source reports ZHO ahead by `0.013s/lap`, not BOT.

`bottas_zhou_race_2024`

- Status: SUPPLEMENTAL_NEAR_ZERO.
- Scope: Stake/Sauber race pace, 2024 only.
- Expected relationship: BOT faster than ZHO.
- Threshold: `0.01s/lap`.
- Source: Motorsport-Total / PACETEQ Sauber duel.
- Source type: teammate race-pace delta.
- Pass rule if reported: `BOT_mu_s - ZHO_mu_s >= 0.01`.
- Date accessed: 2026-05-12.
- Notes: report separately only. Direction is supported, but the gap is too
  close to noise for a load-bearing HARD check.

### 3.2 Qualifying Candidates (Post-Audit)

These rows remain visible because they are useful external comparisons, but
the 2026-05-17 construct audit demoted them from HARD to `EXTERNAL_CONTEXT`.

`verstappen_perez_quali_2022`

- Status: EXTERNAL_CONTEXT.
- Scope: Red Bull qualifying, 2022.
- Expected relationship: VER faster than PER.
- Threshold: `0.290s`.
- Source: Motorsport.com / PACETEQ Perez trend.
- Source type: teammate qualifying delta.
- Pass rule: `VER_mu_s - PER_mu_s >= 0.290`.
- Date accessed: 2026-05-12.

`verstappen_perez_quali_2023`

- Status: EXTERNAL_CONTEXT.
- Scope: Red Bull qualifying, 2023.
- Expected relationship: VER faster than PER.
- Threshold: `0.621s`.
- Source: Motorsport.com / PACETEQ 2023 review.
- Source type: teammate qualifying delta.
- Pass rule: `VER_mu_s - PER_mu_s >= 0.621`.
- Date accessed: 2026-05-12.
- Notes: RacingNews365 reports a lower corroborating value of `0.495s`;
  threshold follows the accepted PACETEQ source.

`verstappen_perez_quali_2024`

- Status: EXTERNAL_CONTEXT.
- Scope: Red Bull qualifying, 2024.
- Expected relationship: VER faster than PER.
- Threshold: `0.66s`.
- Source: Motorsport-Total / PACETEQ Red Bull duel.
- Source type: teammate qualifying delta.
- Pass rule: `VER_mu_s - PER_mu_s >= 0.66`.
- Date accessed: 2026-05-12.
- Notes: Motor Sport Magazine gives a partial-season corroborating value of
  `0.486s`.

`russell_hamilton_quali_2024`

- Status: EXTERNAL_CONTEXT.
- Scope: Mercedes qualifying, 2024 only.
- Expected relationship: RUS faster than HAM.
- Threshold: `0.23s`.
- Source: Motorsport-Total / PACETEQ Mercedes duel.
- Source type: teammate qualifying delta.
- Pass rule: `RUS_mu_s - HAM_mu_s >= 0.23`.
- Date accessed: 2026-05-12.
- Notes: Motor Sport Magazine gives a partial-season corroborating value of
  `0.098s`.

`albon_sargeant_quali_2023`

- Status: EXTERNAL_CONTEXT.
- Scope: Williams qualifying, 2023 only.
- Expected relationship: ALB faster than SAR.
- Threshold: `0.522s`.
- Source: Motorsport.com / PACETEQ 2023 review.
- Source type: teammate qualifying delta.
- Pass rule: `ALB_mu_s - SAR_mu_s >= 0.522`.
- Date accessed: 2026-05-12.
- Notes: split from the unsourced 2023-2024 aggregate.

`albon_sargeant_quali_2024`

- Status: EXTERNAL_CONTEXT.
- Scope: Williams qualifying, 2024 Sargeant sample.
- Expected relationship: ALB faster than SAR.
- Threshold: `0.66s`.
- Source: Motorsport-Total / PACETEQ Williams duel.
- Source type: teammate qualifying delta.
- Pass rule: `ALB_mu_s - SAR_mu_s >= 0.66`.
- Date accessed: 2026-05-12.
- Notes: covers Sargeant's 2024 Williams starts before the driver change.

### 3.3 Source URLs

- Motorsport.com / PACETEQ Perez trend:
  https://lat.motorsport.com/f1/news/checo-perez-diferencia-verstappen-f1-2024/10627633/
- Motorsport.com / PACETEQ 2023 review:
  https://lat.motorsport.com/f1/news/verstappen-checo-perez-diferencia-f1-2023/10561671/
- Motorsport-Total / PACETEQ Red Bull duel:
  https://www.motorsport-total.com/formel-1/news/maximal-ueberlegen-wie-verstappen-perez-2024-in-grund-und-boden-fuhr-24122902
- Motorsport-Total / PACETEQ Aston Martin duel:
  https://www.motorsport-total.com/formel-1/news/analyse-ist-lance-stroll-wirklich-zu-langsam-24122701
- Motorsport-Total / PACETEQ Williams duel:
  https://www.motorsport-total.com/formel-1/news/nach-sargeant-rauswurf-so-viel-schneller-war-franco-colapinto-wirklich-24122307
- Motorsport-Total / PACETEQ Sauber duel:
  https://www.motorsport-total.com/formel-1/news/sauber-duell-das-war-2024-die-ganz-grosse-schwaeche-von-valtteri-bottas-24122202
- Motorsport-Total / PACETEQ Mercedes duel:
  https://www.motorsport-total.com/formel-1/news/mercedes-fahrer-analysiert-hat-lewis-hamilton-seine-qualifyingpace-verloren-24122802

Removed from validation set (now smoke tests in `tests/test_prior_signs.py`,
not validation evidence):

- `tsunoda_devries_race_2023`: sample too thin.
- `leclerc_sainz_quali_2022_2024`: relationship contested; hedged check is
  a smoke test in disguise.

Cut or downgraded during Phase 1 source research:

- `russell_latifi_race_2022`: impossible pairing/year.
- `bottas_zhou_race_2022`: no accepted numeric race-pace source found.
- `bottas_zhou_race_2023`: accepted source reports Zhou slightly faster,
  conflicting with the row's expected direction.
- `bottas_zhou_race_2024`: kept as SUPPLEMENTAL only because the accepted
  source reports a near-zero 0.01s/lap Bottas advantage.
- `albon_sargeant_race_2023_2024` and
  `albon_sargeant_quali_2023_2024`: replaced by season-specific rows.

## 4. Phase 1 Research Protocol

Phase 1 research is closed for this validation set. Future rows should follow
the same discipline:

1. Read Section 1 in full before opening any source. Source acceptance
   is a methodology call, not a reputation call.
2. Pick a target shape for the validation set before sourcing. A
   reasonable target: 4-6 race rows filled, 2-3 quali rows filled. If
   sources do not support that, reduce the count rather than soften the
   criteria.
3. Scratch-file protocol (Decision 2026-05-12). Candidate links live in
   `docs/fixes/phase_1_source_research.md` until promoted here. That file is
   research notes only and is not validation evidence. It must not be
   referenced as validation evidence by any downstream doc, fit artifact, or
   replay output.
4. For each candidate row, in audit order:
   - identify candidate sources (assistant may help by surfacing links
     into the scratch file from the previous validation pass);
   - read each candidate source against Section 1.1-1.4;
   - if accepted, record `source_url`, `source_type`, `threshold_s`,
     `pass_rule`, and `date_accessed` in Section 3 of this doc;
   - if conditionally accepted, record the translation (e.g. percentage
     to seconds, season aggregation method) in the notes column;
   - if rejected, record the rejection reason in the notes column and
     change the row's status to `CUT_<reason>`;
   - never leave a worked row unresolved; either
     promote it to filled, or cut it.
5. Do not chase a target row count. If a row cannot be sourced
   defensibly, cut it. The validation report acknowledges undersourced
   areas (especially quali) by widening initial sigma and tightening
   replay diagnostics, not by inventing thresholds.
6. Record cut reasons. A cut row carries information: it tells future
   readers why a check was not made. Never silently delete a candidate.

After Phase 1 closes, write a one-page "Validation Set Provenance" note
that summarizes:

- how many rows were filled, per category;
- which rows were cut and why;
- whether quali coverage is provisional and what compensates for it
  (wider sigma, stricter replay diagnostics);
- the source-acceptance criteria applied, including any project-specific
  exceptions.

## 5. Cut Criteria

Cut a candidate if:

- no independent numeric source meeting Section 1 criteria is found;
- the source reports a different construct than the model estimates and
  cannot be translated cleanly;
- the sample is too thin for a hard magnitude check;
- the threshold would have to be chosen from this project's own fitted
  output;
- the relationship is contested across seasons and a hedged version would
  collapse into a direction-only check.

## 6. Validation Report Format

Every row in Section 3 carries an `evidence_tier` value, written into
the row's notes column or as a separate column once the table is
edited:

```text
HARD - independent numeric seconds delta, methodology
                   stated or accepted under the Motorsport / PACETEQ
                   source-family rule. Counts toward pass/fail.
EXTERNAL_CONTEXT - independent numeric seconds delta with useful context,
                   but not a proven same-construct match for the current
                   fitted target. Reported separately.
SUPPLEMENTAL - F1Metrics-style or partly model-derived; methodology
                   stated; or accepted-source evidence too near zero to
                   carry a hard threshold. Reported separately.
SMOKE_ONLY - direction-only check, lives in
                   tests/test_prior_signs.py, not in the validation
                   report.
CUT - researched and rejected; cut reason recorded.
```

The validation report counts only HARD rows toward "passed/failed".
EXTERNAL_CONTEXT and SUPPLEMENTAL rows appear in separate sections.
SMOKE_ONLY rows do not appear in the validation report at all.

The prior validation report must state:

- HARD race checks passed and failed, with thresholds and sources;
- HARD quali checks passed and failed, with thresholds and sources;
- EXTERNAL_CONTEXT rows reported separately (not counted toward pass/fail);
- SUPPLEMENTAL rows reported separately (not counted toward pass/fail);
- which candidate checks were CUT before fitting and why;
- whether HARD validation is provisional, and what internal diagnostics carry
  the load while same-construct external evidence is absent;
- that SMOKE_ONLY direction tests are excluded from the pass count.

## 6.5 Phase 1 Validation Set Provenance

Phase 1 source research closed on 2026-05-12 with the Motorsport / PACETEQ
source-family rule in Section 1.3.6 accepted.

Filled HARD rows:

- Race pace: 0 rows after the 2026-05-17 construct audit.
- Qualifying pace: 0 rows after the 2026-05-17 construct audit.

External context rows:

- Race pace: 7 rows.
  `verstappen_perez_race_2022`, `verstappen_perez_race_2023`,
  `verstappen_perez_race_2024`, `alonso_stroll_race_2023`,
  `alonso_stroll_race_2024`, `albon_sargeant_race_2023`,
  `albon_sargeant_race_2024`.
- Qualifying pace: 6 rows.
  `verstappen_perez_quali_2022`, `verstappen_perez_quali_2023`,
  `verstappen_perez_quali_2024`, `russell_hamilton_quali_2024`,
  `albon_sargeant_quali_2023`, `albon_sargeant_quali_2024`.

Supplemental rows:

- `bottas_zhou_race_2024`: accepted source reports BOT +0.01s/lap, which
  is useful context but too near zero for a hard threshold.

Cut rows and replacements:

- `russell_latifi_race_2022`: impossible pairing/year.
- `bottas_zhou_race_2022`: no accepted numeric race-pace source found.
- `bottas_zhou_race_2023`: accepted source reports Zhou slightly faster,
  conflicting with the original expected relationship.
- `tsunoda_devries_race_2023`: sample too thin; smoke-only if retained.
- `leclerc_sainz_quali_2022_2024`: contested and hedged; smoke-only if
  retained.
- `albon_sargeant_race_2023_2024` and
  `albon_sargeant_quali_2023_2024`: replaced by season-specific rows.

Coverage note: HARD validation is now explicitly provisional for both race and
qualifying because the available external rows are useful but not proven
same-construct evidence for the current extractors. That is preferable to
counting incompatible checks as validation coverage.

## 7. Lock Rules

This file is lockable when:

- Section 1 (source acceptance criteria) has been reviewed and accepted by
  the project owner;
- every row in Section 3 is either filled with a real source, threshold,
  pass rule, and date accessed, or has been cut with a documented reason;
- at least 3 race checks and 2 quali checks are filled, OR the report
  explicitly labels quali coverage as provisional and documents the
  compensation;
- the validation report format (Section 6) is followed in the prior fit
  output;
- the file's status header is updated to "locked".

The validation set is locked and can grade the Phase 6 prior fit.

# Teammate-Network Prior Validation Evidence

Date: 2026-05-09
Status: scaffold; not lockable until source acceptance criteria are approved
and source-backed thresholds are filled

This file records hard, source-backed validation checks for the
teammate-network prior. It exists to prevent post-fit rationalisation.

A row is not a hard validation check until every required field is filled:

- `source_url` or exact source reference;
- source type;
- source-backed threshold in seconds;
- pass/fail rule;
- date accessed;
- analyst sign-off (the project owner explicitly accepted this row, not just
  an AI surfacing a candidate).

Direction-only checks do not belong here. Put sign-flip smoke tests in code,
for example `tests/test_prior_signs.py`.

## 1. Source Acceptance Criteria

A source is judged on methodology, not reputation. The same outlet can
publish a defensible number in one article and a marketing number in
another. Apply the rules below per artifact, not per outlet.

### 1.1 Accepted

A source is **accepted** for hard validation only if it satisfies all of:

- it publishes a numeric teammate gap in seconds (or a per-lap delta that
  can be converted to seconds without an undocumented coefficient);
- it states the construct: race pace, qualifying pace, single-lap pace,
  long-run pace, or session-best;
- it states the sample basis: which sessions, how laps were filtered,
  whether outliers or pit laps were removed;
- it is teammate-relative or can be made teammate-relative without
  re-running the source's model on the analyst's side;
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

A source is **conditionally accepted** when the construct or scope partly
matches but requires translation. Use the row only if the translation is
mechanical and the analyst documents it.

Examples:

- a published per-race teammate qualifying delta that the analyst
  aggregates into a season median: acceptable if the aggregation method is
  written into the row's notes;
- a season-summary chart with seconds deltas but no explicit lap-filtering
  description: acceptable if the prior-side construct can be matched to the
  source's window (e.g. compare against the prior fitted on green-flag
  long-run laps if that is what the source approximates);
- a source that reports the gap as a percentage (e.g. "0.42% slower"):
  acceptable if the analyst converts using a documented reference lap time
  and records the conversion.

### 1.3 Rejected

A source is **rejected** for hard validation if any of the following hold:

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
- the source is an AI assistant's synthesis or summary that the analyst
  has not personally verified against the underlying artifact.

### 1.3.5 F1Metrics-Style Sources Rule

F1Metrics and similar independent driver-rating projects publish numbers
that look like validation evidence but partly overlap with what this prior
estimates. The rule for v1:

- F1Metrics-style sources are **conditionally acceptable** when they
  publish a numeric teammate seconds delta with enough methodology to
  understand the construct (which sessions, which laps, what filters).
- They are **not acceptable as the sole hard threshold** when the
  published value is a model-derived driver rating on its own scale rather
  than a numeric teammate seconds delta with documented methodology.
- They may corroborate another timing source (broadcast charts, F1TV
  graphics with named lap windows) or support rough magnitude when paired
  with a clearly independent observation.
- If F1Metrics is the only source for a row, mark the row conditionally
  accepted in the notes, record the circularity risk explicitly, and do
  not treat the threshold as load-bearing in the validation report.

This rule applies to any independent driver-rating project that estimates
the same quantity this prior estimates. F1Metrics is named because it is
the most likely candidate the analyst will encounter; the rule is not
F1Metrics-specific.

### 1.4 Construct Mismatch Categories

Most cuts come from construct mismatch, not source quality. The five
patterns to watch for:

- **race pace vs qualifying pace.** A driver may be 0.15s/lap faster than
  the teammate over long runs but 0.30s/lap faster on a single push lap.
  The prior estimates these separately. A row that uses qualifying
  evidence to validate the race prior is mismatched.
- **teammate-relative vs global.** The prior is a teammate-relative
  residual. A source that reports "VER is the fastest driver in F1 by
  0.4s" is a global statement. It does not pin the VER-PER teammate gap.
- **single-season vs multi-season aggregate.** A row that says
  `verstappen_perez_race_2022_2024` aggregates three seasons of car
  development, Pérez's documented 2024 form drop, and 2022 vs 2023 vs 2024
  reliability. Most published deltas are season-by-season. Aggregating
  them across seasons requires assumptions the source did not make.
- **model-derived rating vs published timing delta.** A source that runs
  its own driver-rating model and publishes the rating outputs is partly
  circular as validation: it is doing what the prior is doing. Prefer
  sources that publish raw timing deltas.
- **opaque broadcast number vs documented method.** Sky/F1TV graphics
  often quote teammate seconds-per-lap during a stint, but the underlying
  filter (which laps, what tire age) is rarely visible. Acceptable only
  if the broadcast also names the lap window and the analyst can match
  it.

### 1.5 Methodology Ownership Rule

An assistant may surface candidate links and summarize what they appear
to contain. The assistant **must not** turn those into accepted
validation thresholds. Every row that becomes hard evidence must carry an
explicit analyst sign-off recorded in the row's notes column. Without
sign-off, the row stays at `TODO_SOURCE` regardless of how complete the
other fields look.

This rule exists to prevent the validation set from drifting into "Claude
found numbers on the internet that seemed authoritative." The whole point
of the validation set is that the analyst stands behind the thresholds.

## 2. Candidate Check Audit

Each candidate from the original scaffold is reviewed against the
criteria in Section 1. Action codes:

- `KEEP` — row is well-scoped; proceed to source research.
- `SPLIT` — scope is too broad; split into per-season rows or narrower
  comparisons before sourcing.
- `NARROW` — scope is too aggressive; tighten the relationship or the
  sample window.
- `QUALITATIVE` — magnitude unlikely to be sourceable; convert to a
  unit-test smoke check (sign-only) and remove from the validation set.
- `CUT` — sample too thin or construct mismatch unlikely to be fixable;
  remove the row entirely.

### 2.1 Race Candidates

| Original Check ID | Action | Reasoning | Recommended replacement(s) |
|---|---|---|---|
| `verstappen_perez_race_2022_2024` | SPLIT | 2022-2024 aggregates three different cars, Pérez's well-documented 2024 form drop, and changing reliability. Most race-pace sources are per-season. Aggregating across seasons hides the very signal the prior should reflect. | `verstappen_perez_race_2022`, `verstappen_perez_race_2023`, `verstappen_perez_race_2024` — three rows, each sourced separately if available. Even one of these would be a defensible hard check. |
| `alonso_stroll_race_2023_2024` | SPLIT | Same multi-season aggregation issue. 2023 Aston Martin pace was very different from 2024. ALO-STR gap likely larger in 2023 than 2024 but should be sourced per season. | `alonso_stroll_race_2023`, `alonso_stroll_race_2024`. |
| `albon_sargeant_race_2023_2024` | KEEP (with caution) | Two-season scope is borderline but ALB-SAR gap was reportedly large and reasonably stable. Try as written first; split if a clean two-season number cannot be sourced. | Keep `albon_sargeant_race_2023_2024`; fall back to `albon_sargeant_race_2024` (when SAR was full-time) if multi-season aggregation is unsourceable. |
| `russell_latifi_race_2022` | NARROW or CUT | 2022 Williams sample is small relative to other rows. The pair only ran together one season. Construct itself is fine but findings depend heavily on source availability for a comparatively obscure pairing. | If sourceable, keep as written. If not, cut rather than soft-source. |
| `bottas_zhou_race_2022_2024` | SPLIT | Same multi-season problem. ZHO improved across his stint; 2022 BOT-ZHO gap likely larger than 2024. Use season-by-season. | `bottas_zhou_race_2022`, `bottas_zhou_race_2023`, `bottas_zhou_race_2024`. |
| `tsunoda_devries_race_2023` | CUT | DEV ran ~10 races. Even season-scoped, the sample is too thin for a hard magnitude check, and the published deltas (where they exist) carry wide error bars. Direction is obvious; magnitude is not. | Move sign-only check to `tests/test_prior_signs.py`. Remove from validation set. |

### 2.2 Qualifying Candidates

| Original Check ID | Action | Reasoning | Recommended replacement(s) |
|---|---|---|---|
| `verstappen_perez_quali_2022_2024` | SPLIT | Same multi-season concern as the race version. Quali deltas are usually published per season; aggregating across three seasons hides VER's improvement vs PER's decline. | `verstappen_perez_quali_2022`, `verstappen_perez_quali_2023`, `verstappen_perez_quali_2024`. |
| `leclerc_sainz_quali_2022_2024` | NARROW or QUALITATIVE | Direction is contested in some seasons. The relationship as originally written ("LEC at least close to or faster than SAI") is hedged because the truth is in fact contested year-by-year. A hedged relationship is not a hard check; it is a smoke test in disguise. | Either pick a single season where the source is clear-cut, or move to `tests/test_prior_signs.py` as a "no large unexpected reversal" smoke check. Do not try to make the multi-season version a hard check. |
| `russell_hamilton_quali_2024` | KEEP | Single season, clear construct, multiple independent published quali deltas exist for 2024. Most defensible quali candidate in the set. | Keep as `russell_hamilton_quali_2024`. |
| `albon_sargeant_quali_2023_2024` | KEEP (with caution) | ALB had a clear quali advantage; published head-to-heads are common. Two-season aggregation is borderline; split if sourceable per season. | Keep, with `albon_sargeant_quali_2024` as the fallback narrower scope. |

### 2.3 Audit Summary

After applying the audit:

- **Race**: 0 KEEP-as-written, 3 SPLIT (Verstappen-Pérez, Alonso-Stroll,
  Bottas-Zhou — each becoming up to 3 per-season rows), 1 KEEP-with-caution
  (Albon-Sargeant), 1 NARROW-or-CUT (Russell-Latifi), 1 CUT
  (Tsunoda-De Vries).
- **Quali**: 1 KEEP-as-written (Russell-Hamilton 2024), 1 KEEP-with-caution
  (Albon-Sargeant), 1 SPLIT (Verstappen-Pérez), 1 NARROW-or-QUALITATIVE
  (Leclerc-Sainz).

The candidate pool the analyst should research is closer to 7-9 race
checks (mostly per-season) and 3-5 quali checks, *before* applying the
"can I actually source this defensibly" filter. Expect roughly half of
those to survive Phase 1. That is the realistic shape of the validation
set, not the inflated row count of the original scaffold.

## 3. Revised Candidate Tables

These tables replace the original Candidate Race Checks and Candidate
Qualifying Checks tables. They reflect the audit. All numeric and
source fields remain `TODO`; the analyst fills them during Phase 1.

Each row also acquires an `evidence_tier` value during Phase 1, per
Section 6:

- HARD — counts toward pass/fail in the validation report;
- SUPPLEMENTAL — F1Metrics-style or partly model-derived; reported
  separately, does not count toward pass/fail;
- SMOKE_ONLY — direction-only smoke check; does not appear in the
  validation report;
- CUT — researched and rejected; cut reason recorded.

The evidence_tier may be recorded in the Notes column or added as a
separate column when the analyst edits the tables. For now, all rows
default to TODO_SOURCE/TODO_TIER until Phase 1 work begins.

### 3.1 Race Candidates (Post-Audit)

| Status | Check ID | Scope | Expected relationship | Threshold s | Source | Source type | Pass rule | Date accessed | Analyst sign-off | Notes |
|---|---|---|---|---:|---|---|---|---|---|---|
| TODO_SOURCE | `verstappen_perez_race_2022` | Red Bull race pace, 2022 only | VER faster than PER | TODO | TODO | TODO | `VER_mu_s - PER_mu_s >= threshold_s` | TODO | NO | Replaces the 2022-2024 aggregate. |
| TODO_SOURCE | `verstappen_perez_race_2023` | Red Bull race pace, 2023 only | VER faster than PER | TODO | TODO | TODO | `VER_mu_s - PER_mu_s >= threshold_s` | TODO | NO | Per-season split. |
| TODO_SOURCE | `verstappen_perez_race_2024` | Red Bull race pace, 2024 only | VER faster than PER | TODO | TODO | TODO | `VER_mu_s - PER_mu_s >= threshold_s` | TODO | NO | PER form drop in second half complicates choice of sample window. |
| TODO_SOURCE | `alonso_stroll_race_2023` | Aston Martin race pace, 2023 only | ALO faster than STR | TODO | TODO | TODO | `ALO_mu_s - STR_mu_s >= threshold_s` | TODO | NO | Replaces 2023-2024 aggregate. |
| TODO_SOURCE | `alonso_stroll_race_2024` | Aston Martin race pace, 2024 only | ALO faster than STR | TODO | TODO | TODO | `ALO_mu_s - STR_mu_s >= threshold_s` | TODO | NO | Per-season split. |
| TODO_SOURCE | `albon_sargeant_race_2023_2024` | Williams race pace, 2023-2024 | ALB faster than SAR | TODO | TODO | TODO | `ALB_mu_s - SAR_mu_s >= threshold_s` | TODO | NO | Try multi-season first; fall back to 2024-only if unsourceable. |
| TODO_SOURCE | `russell_latifi_race_2022` | Williams race pace, 2022 | RUS faster than LAT | TODO | TODO | TODO | `RUS_mu_s - LAT_mu_s >= threshold_s` | TODO | NO | Cut if no defensible source surfaces. |
| TODO_SOURCE | `bottas_zhou_race_2022` | Alfa Romeo race pace, 2022 only | BOT faster than ZHO | TODO | TODO | TODO | `BOT_mu_s - ZHO_mu_s >= threshold_s` | TODO | NO | Per-season split. |
| TODO_SOURCE | `bottas_zhou_race_2023` | Alfa Romeo race pace, 2023 only | BOT faster than ZHO | TODO | TODO | TODO | `BOT_mu_s - ZHO_mu_s >= threshold_s` | TODO | NO | Per-season split. |
| TODO_SOURCE | `bottas_zhou_race_2024` | Stake/Sauber race pace, 2024 only | BOT faster than ZHO (smaller margin) | TODO | TODO | TODO | `BOT_mu_s - ZHO_mu_s >= threshold_s` | TODO | NO | Margin closed in 2024; threshold likely smaller. |

### 3.2 Qualifying Candidates (Post-Audit)

| Status | Check ID | Scope | Expected relationship | Threshold s | Source | Source type | Pass rule | Date accessed | Analyst sign-off | Notes |
|---|---|---|---|---:|---|---|---|---|---|---|
| TODO_SOURCE | `verstappen_perez_quali_2022` | Red Bull qualifying, 2022 | VER faster than PER | TODO | TODO | TODO | `VER_mu_s - PER_mu_s >= threshold_s` | TODO | NO | Per-season split. |
| TODO_SOURCE | `verstappen_perez_quali_2023` | Red Bull qualifying, 2023 | VER faster than PER | TODO | TODO | TODO | `VER_mu_s - PER_mu_s >= threshold_s` | TODO | NO | Per-season split. |
| TODO_SOURCE | `verstappen_perez_quali_2024` | Red Bull qualifying, 2024 | VER faster than PER | TODO | TODO | TODO | `VER_mu_s - PER_mu_s >= threshold_s` | TODO | NO | Per-season split. |
| TODO_SOURCE | `russell_hamilton_quali_2024` | Mercedes qualifying, 2024 only | RUS faster than HAM if source supports it | TODO | TODO | TODO | `RUS_mu_s - HAM_mu_s >= threshold_s` | TODO | NO | Most defensible single-season quali candidate. |
| TODO_SOURCE | `albon_sargeant_quali_2023_2024` | Williams qualifying, 2023-2024 | ALB faster than SAR | TODO | TODO | TODO | `ALB_mu_s - SAR_mu_s >= threshold_s` | TODO | NO | Fall back to 2024-only if multi-season unsourceable. |

Removed from validation set (now smoke tests in `tests/test_prior_signs.py`,
not validation evidence):

- `tsunoda_devries_race_2023`: sample too thin.
- `leclerc_sainz_quali_2022_2024`: relationship contested; hedged check is
  a smoke test in disguise.

## 4. Phase 1 Research Protocol

These steps belong to the analyst, not to an AI assistant. The assistant
may surface candidate links per Section 1.5; turning a link into a hard
check requires the analyst's explicit sign-off in the row's notes column.

1. **Read Section 1 in full** before opening any source. Source acceptance
   is a methodology call, not a reputation call.
2. **Pick a target shape for the validation set** before sourcing. A
   reasonable target: 4-6 race rows filled, 2-3 quali rows filled. If
   sources do not support that, reduce the count rather than soften the
   criteria.
3. **For each candidate row, in audit order:**
   - identify candidate sources (assistant may help by surfacing links);
   - read each candidate source against Section 1.1-1.4;
   - if accepted, record `source_url`, `source_type`, `threshold_s`,
     `pass_rule`, `date_accessed`, and tick `analyst_sign_off` to `YES`;
   - if conditionally accepted, record the translation (e.g. percentage
     to seconds, season aggregation method) in the notes column;
   - if rejected, record the rejection reason in the notes column and
     change the row's status to `CUT_<reason>`;
   - never leave a row at `TODO_SOURCE` after working on it; either
     promote it to filled, or cut it.
4. **Do not chase a target row count.** If a row cannot be sourced
   defensibly, cut it. The validation report acknowledges undersourced
   areas (especially quali) by widening initial sigma and tightening
   replay diagnostics, not by inventing thresholds.
5. **Record cut reasons.** A cut row carries information: it tells future
   readers (and the analyst) why a check was not made. Never silently
   delete a candidate.
6. **Sign-off discipline.** A row's `analyst_sign_off` field flips to
   `YES` only when the analyst has personally read the source and accepts
   the threshold. AI-surfaced candidate sources stay at `NO` until that
   read happens.

After Phase 1 closes, write a one-page "Validation Set Provenance" note
that summarizes:

- how many rows were filled, per category;
- which rows were cut and why;
- whether quali coverage is provisional and what compensates for it
  (wider sigma, stricter replay diagnostics);
- the source-acceptance criteria the analyst applied (a reference to
  Section 1 above is fine, plus any project-specific exceptions).

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
HARD             — independent numeric seconds delta, methodology
                   stated, analyst sign-off. Counts toward pass/fail.
SUPPLEMENTAL     — F1Metrics-style or partly model-derived; methodology
                   stated; corroborates HARD rows. Reported separately.
SMOKE_ONLY       — direction-only check, lives in
                   tests/test_prior_signs.py, not in the validation
                   report.
CUT              — researched and rejected; cut reason recorded.
```

The validation report counts only HARD rows toward "passed/failed".
SUPPLEMENTAL rows appear in a separate section. SMOKE_ONLY rows do not
appear in the validation report at all.

The prior validation report must state:

- HARD race checks passed and failed, with thresholds and sources;
- HARD quali checks passed and failed, with thresholds and sources;
- SUPPLEMENTAL rows reported separately (not counted toward pass/fail);
- which candidate checks were CUT before fitting and why;
- whether quali validation is weaker than race validation, and how the
  prior compensates (wider sigma, stricter replay);
- that SMOKE_ONLY direction tests are excluded from the pass count.

## 7. Lock Rules

This file is lockable when:

- Section 1 (source acceptance criteria) has been reviewed and accepted by
  the analyst;
- every row in Section 3 is either filled with a real source, threshold,
  pass rule, date accessed, and `analyst_sign_off = YES`, or has been cut
  with a documented reason;
- at least 3 race checks and 2 quali checks are filled, OR the report
  explicitly labels quali coverage as provisional and documents the
  compensation;
- the validation report format (Section 6) is followed in the prior fit
  output;
- the file's status header is updated from "scaffold" to "locked" by the
  analyst, not by an assistant.

Until then, the prior fit has nothing to be graded against, and Phase 6 of
the master execution plan is blocked.

# Teammate-Network Prior Construct Audit

Date: 2026-05-17
Status: Phase 6 audit note

## Purpose

This note audits the failed direct validation rows from the first Phase 6
prior fit. The question is narrow:

> Are the failed source-backed rows measuring the same object as the local
> matched-lap extractor?

The answer is mostly no for qualifying and not proven for race pace.
That does not automatically make the extractor wrong. It means the current
HARD validation set is grading several non-identical constructs as though
they were one construct.

## Summary

1. The local qualifying extractor does not estimate "best qualifying lap
   gap." It estimates the median of run-order-matched quick laps, starting
   from the highest common segment and then pulling in lower segments until it
   reaches three paired laps.
2. The strongest 2024 PACETEQ wording found during the audit says the
   qualifying comparison uses the drivers' best qualifying times. That is
   a different target from the local multi-lap median.
3. The current prior fit then adds a second difference: session observations
   are weighted by effective pair count and inverse SE, while the published
   source rows are season summary averages with no evidence that they use the
   same weighting rule.
4. The direct qualifying failures are therefore not clean evidence that the
   extractor is underestimating driver skill. They are evidence that the
   validation contract is mixing:
   - peak qualifying pace;
   - comparable-stage qualifying pace;
   - local multi-run median qualifying pace;
   - equal-session summaries;
   - WLS-weighted summaries.
5. Race pace is less clear. The PACETEQ race rows and the local race extractor
   both aim at "race pace," but the local extractor is much more tightly
   controlled: same compound, same weather bucket, same stint-lap index,
   green-track filtering, and stint-outlier removal. The PACETEQ articles do
   not document an equivalent conditioning rule, so those race thresholds are
   not yet proven same-construct HARD checks either.

## What The Local Extractor Actually Measures

### Qualifying

For one team in one qualifying session, the extractor:

1. keeps only accurate, non-deleted, non-pit, weather-routable qualifying
   laps;
2. keeps "quick" laps within `1.07 * best_lap_in_segment_weather_bucket`;
3. finds common segments;
4. visits `Q3`, then `Q2`, then `Q1`;
5. pairs push laps by:
   - qualifying segment;
   - compound;
   - weather bucket;
   - run order within that segment;
6. keeps collecting segments until at least three matched lap pairs exist;
7. writes the session observation as the median matched gap.

Relevant implementation points:

- `src/extractors/matched_laps.py:604-646`
- `src/extractors/matched_laps.py:836-888`
- `src/extractors/matched_laps.py:1052-1067`

That is a reasonable repeatable qualifying execution statistic. It is not a
single-lap peak statistic.

Two details matter:

- A driver pairing that both reaches Q3 often contributes one Q3 lap plus Q2
  support laps, because the extractor needs at least three matched pairs.
- The quick-lap filter is permissive. It removes clear cooldown laps, but it
  intentionally keeps more than the single best lap.

### Race Pace

For race sessions, the extractor:

1. removes lap 1, final driver laps, pit laps, deleted laps, inaccurate laps,
   non-green laps, large position-change laps, stint outliers, and unroutable
   weather laps;
2. pairs only laps with the same:
   - compound;
   - weather bucket;
   - stint-lap index;
   - repeated match order;
3. writes the session observation as the median matched gap.

Relevant implementation points:

- `src/extractors/matched_laps.py:809-833`
- `src/extractors/matched_laps.py:1037-1050`

That is a controlled teammate-relative race residual. It is narrower than a
generic season-summary "race pace" number.

## What The Sources Actually Measure

### 2023 PACETEQ review

The 2023 Motorsport/PACETEQ review says it compiled teammate:

- qualifying pace;
- race pace;
- only representative laps;
- with obvious invalid cases removed.

It does not publish enough detail to show:

- whether qualifying uses best laps or multiple push laps;
- whether lower qualifying segments are mixed with later segments;
- whether season rows are equally weighted by event;
- whether race laps are conditioned on same compound and same tire age.

This makes it useful external context, but weaker as a same-construct HARD
validator than the current lock implies.

### 2024 PACETEQ duel articles

The clearest 2024 wording found in the Red Bull duel article says the method
compares the drivers' best qualifying times and race pace. The same article
reports:

- qualifying pace: Verstappen faster than Perez by `0.66s`;
- race pace: Verstappen faster by `0.56s/lap`.

The Mercedes, Williams, Aston Martin, and Sauber duel articles use the same
PACETEQ article family. The articles report qualifying pace, race pace, and
often tire degradation as separate season-summary outputs.

That creates a direct mismatch for qualifying:

- PACETEQ 2024 qualifying row: best qualifying times;
- local qualifying extractor: median matched push-lap gap across one or more
  common segments.

### Motor Sport Magazine comparator

The mid-2024 Motor Sport Magazine article is useful because it states a much
clearer qualifying comparison rule:

- compare only sessions where teammates reached the same qualifying phase;
- exclude wet sessions;
- exclude sessions with different car specs or other non-comparable cases.

Its mid-season qualifying values are:

- VER over PER: `0.486s`;
- RUS over HAM: `0.098s`;
- ALB over SAR: `0.257s`.

Those values are not drop-in HARD thresholds for full-season rows, but the
method is materially closer to the local comparable-stage idea than the
PACETEQ best-lap wording is.

## Exact Artifact Evidence

The table below uses only the versioned Phase 5 aggregate artifact and the
current Phase 6 validation report.

| Row | Locked threshold | Direct WLS delta | Equal-session mean | Artifact read |
| --- | ---: | ---: | ---: | --- |
| `verstappen_perez_quali_2023` | `0.621s` | `0.363s` | `0.543s` | weighting and construct both matter |
| `verstappen_perez_quali_2024` | `0.660s` | `0.462s` | `0.507s` | source target is wider than local target |
| `russell_hamilton_quali_2024` | `0.230s` | `0.113s` | `0.083s` | source target strongly disagrees with local target |
| `albon_sargeant_quali_2023` | `0.522s` | `0.412s` | `0.418s` | likely target mismatch plus sample loss |
| `albon_sargeant_quali_2024` | `0.660s` | `0.222s` | `0.402s` | severe sample and weighting mismatch |

The WLS weights are not a small detail:

- `VER-PER 2023 qualifying`: Las Vegas, Italy, Bahrain, and Azerbaijan make up
  about `51.8%` of total WLS weight.
- `ALB-SAR 2024 qualifying`: Austria, Monaco, and Canada make up about `95.1%`
  of total WLS weight.
- `ALB-SAR 2024 qualifying` has only five valid dry aggregate rows in the
  Phase 5 artifact, while the locked source row covers Sargeant's 2024 Williams
  starts as a season summary.

Those are not equivalent sample summaries.

## Reproducible Construct Probe

On 2026-05-17 I added a dedicated offline diagnostic runner:

- `scripts/probe_teammate_network_constructs.py`

It reloads the cached qualifying sessions, recomputes the current extractor
construct, and reports two alternative statistics beside it:

- `highest_common_best`: best lap gap in the highest common segment;
- `any_valid_best`: best valid lap gap anywhere in the session.

The fresh offline recomputation now reproduces every stored Phase 5 qualifying
delta to within `1ms` for the audited HARD rows: `0` artifact/cache mismatch
rows across all six checks. That matters because it removes the earlier concern
that the alternative-statistic comparison was resting on a drifting cache read.

Generated evidence:

- `data/diagnostics/teammate_network_construct_probe/qualifying_construct_probe.md`
- `data/diagnostics/teammate_network_construct_probe/qualifying_construct_probe.json`
- `data/diagnostics/teammate_network_construct_probe/qualifying_session_rows.csv`

| Pair-season | HARD threshold | Phase 5 WLS | Phase 5 equal mean | Highest common best | Any valid best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `VER-PER 2022` | `0.290s` | `0.361s` | `0.157s` | `0.189s` | `0.210s` |
| `VER-PER 2023` | `0.621s` | `0.363s` | `0.543s` | `0.672s` | `0.839s` |
| `VER-PER 2024` | `0.660s` | `0.462s` | `0.507s` | `0.467s` | `0.610s` |
| `RUS-HAM 2024` | `0.230s` | `0.113s` | `0.083s` | `0.345s` | `0.353s` |
| `ALB-SAR 2023` | `0.522s` | `0.412s` | `0.418s` | `0.554s` | `1.236s` |
| `ALB-SAR 2024` | `0.660s` | `0.222s` | `0.402s` | `0.380s` | `0.636s` |

Interpretation:

- changing only the qualifying statistic can move a season summary by several
  tenths while the underlying cached session set stays fixed;
- `RUS-HAM 2024` is the clearest example: the local multi-run median and a
  single-best-lap statistic tell materially different stories;
- `ALB-SAR 2024` shows why the current Phase 5 sample is too thin to grade a
  full-period external season row cleanly.

## Network-Wide Peak Probe

On 2026-05-17 I extended the diagnostic from the six audited rows to every
stored dry qualifying pair-season:

- `scripts/probe_teammate_network_peak_constructs.py`

Generated evidence:

- `data/diagnostics/teammate_network_construct_probe/qualifying_network_peak_probe.md`
- `data/diagnostics/teammate_network_construct_probe/qualifying_network_peak_probe.json`
- `data/diagnostics/teammate_network_construct_probe/qualifying_network_pair_seasons.csv`
- `data/diagnostics/teammate_network_construct_probe/qualifying_network_session_rows.csv`

The wider result is mixed, not a simple vote for replacing the current
extractor:

| Measure | Current construct | Highest-common best | Any-valid best |
| --- | ---: | ---: | ---: |
| Session rows | `627` | `880` | `881` |
| Pair-seasons with at least one row | `49` | `52` | `52` |
| Median within-pair-season SD | `0.317s` | `0.460s` | `0.854s` |
| P75 within-pair-season SD | `0.412s` | `0.648s` | `1.307s` |

`highest_common_best` recovers `253` extra session rows and even supplies rows
for three pair-seasons where the current construct has none:

- `2022:Williams:DEV-LAT`;
- `2024:Alpine:DOO-GAS`;
- `2025:Red Bull Racing:LAW-VER`.

That coverage gain is real. So is the noise cost:

- `60` highest-common-best session rows have an absolute delta above `1s`;
- `18` have an absolute delta above `2s`;
- the largest session outliers include `2024:Alpine:GAS-OCO` at `-5.247s`
  in Britain and `2023:Red Bull Racing:PER-VER` at `-4.299s` in Canada.

Those are not believable skill measurements. They occur because a single
best-lap statistic can still be driven by one compromised "valid" lap once the
pair is reduced to a thin common segment. The current `1.07` quick-lap filter
is wide enough that "best valid lap" is still not the same thing as "clean peak
lap."

The segment mix also matters:

| Highest common segment | Rows |
| --- | ---: |
| `Q1` | `357` |
| `Q2` | `248` |
| `Q3` | `275` |

More than two fifths of the recovered peak rows are Q1 rows. So even the
stronger comparable-stage peak construct is not usually a Q3-only "when it
matters" statistic. It is a cross-grid comparable-stage best-lap statistic.

At pair-season level, the peak-minus-current equal-mean shift has:

- median `0.001s`;
- P75 `0.097s`;
- max `0.490s`.

That shape matters. Moving to `highest_common_best` does not apply one stable
correction to the graph. It leaves the typical pair-season near where it was,
while moving a minority of pair-seasons a lot. The Phase 5 WLS-versus-equal
summary shift is itself non-trivial: median `0.045s`, P75 `0.094s`, max
`0.238s`.

The broader probe supports one conclusion strongly and another only weakly:

- strongly supported: the current external qualifying rows should not be HARD
  checks against the local multi-run estimator;
- not yet supported: replacing the extractor wholesale with
  `highest_common_best` is automatically better.

If the project chooses a peak qualifying target, it still needs a tighter
single-lap validity rule and a revised uncertainty model before that target is
safe to fit as though it were just a higher-coverage version of the current
construct.

## Row-By-Row Audit

### `verstappen_perez_quali_2023`

- PACETEQ locked threshold: `0.621s`.
- Alternative published corroboration already recorded in the validation doc:
  `0.495s`.
- Local direct artifact:
  - WLS: `0.363s`;
  - equal-session mean: `0.543s`.

This is not a clean extractor failure. There are already two published external
values for the same apparent season target, and the local result changes by
`0.180s` depending on whether the season is summarized by WLS or equal-session
mean. The lock is overconfident relative to the documented source ambiguity.

### `verstappen_perez_quali_2024`

- PACETEQ locked threshold: `0.660s`.
- Motor Sport Magazine mid-season comparable-session value: `0.486s`.
- Local direct artifact:
  - WLS: `0.462s`;
  - equal-session mean: `0.507s`.

This is strong construct-mismatch evidence. The local estimator sits near the
clearer comparable-session source and below the PACETEQ best-time source. It is
not reasonable to treat the `0.660s` PACETEQ number as a same-construct HARD
threshold for the current extractor.

### `russell_hamilton_quali_2024`

- PACETEQ locked threshold: `0.230s`.
- Motor Sport Magazine mid-season comparable-session value: `0.098s`.
- Local direct artifact:
  - WLS: `0.113s`;
  - equal-session mean: `0.083s`.

This is the strongest qualifying mismatch in the set. The local extractor and
the comparator with an explicit same-phase rule agree on a value around one
tenth. The locked PACETEQ row grades a different qualifying object.

### `albon_sargeant_quali_2023`

- PACETEQ locked threshold: `0.522s`.
- Local direct artifact:
  - WLS: `0.412s`;
  - equal-session mean: `0.418s`.

The exploratory highest-common-best probe moves to `0.554s`, which is close to
the source row. That is exactly the pattern expected when the external row is
closer to a best-lap statistic and the local row is a multi-run median.

### `albon_sargeant_quali_2024`

- PACETEQ locked threshold: `0.660s`.
- Motor Sport Magazine mid-season comparable-session value: `0.257s`.
- Local direct artifact:
  - WLS: `0.222s`;
  - equal-session mean: `0.402s`;
  - valid dry aggregate rows: `5`.

This row is not fit to be a HARD check for the current extractor. It has:

- best-lap versus multi-run-median mismatch;
- full Sargeant-start source scope versus only five local dry rows;
- extreme WLS concentration in low-gap sessions.

The row is useful context. It is not a clean acceptance gate.

## Race Rows

The failed race rows are less decisive:

| Row | Locked threshold | Direct WLS delta | Equal-session mean |
| --- | ---: | ---: | ---: |
| `verstappen_perez_race_2023` | `0.451s/lap` | `0.295s/lap` | `0.259s/lap` |
| `alonso_stroll_race_2023` | `0.486s/lap` | `0.413s/lap` | `0.364s/lap` |
| `albon_sargeant_race_2023` | `0.293s/lap` | `0.257s/lap` | `0.214s/lap` |
| `albon_sargeant_race_2024` | `0.380s/lap` | `0.228s/lap` | `0.264s/lap` |

These rows all sit below the locked PACETEQ thresholds, but not by the same
kind of unmistakable construct jump seen in qualifying. The likely mismatch is
conditioning:

- PACETEQ says "race pace" over representative laps;
- the local extractor conditions much more tightly on compound, weather, and
  stint-lap index, then takes paired medians.

That narrower local target can legitimately shrink season gaps relative to a
broader season-summary race-pace number. The current source text is not detailed
enough to prove equivalence, so the correct conclusion is:

> the race rows are not yet proven same-construct HARD checks.

One warning: even `alonso_stroll_race_2024`, which passes through the pooled
prior, has a direct source-scope local delta below its threshold
(`0.223s/lap` versus `0.250s/lap`). A pooled pass can hide direct construct
mismatch.

### Reproducible Race Probe

On 2026-05-17 I added a race-side offline probe:

- `scripts/probe_teammate_network_race_constructs.py`

Generated evidence:

- `data/diagnostics/teammate_network_construct_probe/race_construct_probe.md`
- `data/diagnostics/teammate_network_construct_probe/race_construct_probe.json`
- `data/diagnostics/teammate_network_construct_probe/race_session_rows.csv`

It recomputes the current paired extractor view and compares it with broader
valid-lap summaries that keep the local lap-quality filters but remove
same-compound and same-stint-lap pairing.

| Row | HARD threshold | Current equal mean | Broad valid-lap median | Broad valid-lap mean |
| --- | ---: | ---: | ---: | ---: |
| `verstappen_perez_race_2022` | `0.234s/lap` | `0.250s/lap` | `0.322s/lap` | `0.330s/lap` |
| `verstappen_perez_race_2023` | `0.451s/lap` | `0.259s/lap` | `0.298s/lap` | `0.462s/lap` |
| `verstappen_perez_race_2024` | `0.560s/lap` | `0.667s/lap` | `0.488s/lap` | `0.636s/lap` |
| `alonso_stroll_race_2023` | `0.486s/lap` | `0.364s/lap` | `0.325s/lap` | `0.372s/lap` |
| `alonso_stroll_race_2024` | `0.250s/lap` | `0.249s/lap` | `0.359s/lap` | `0.462s/lap` |
| `albon_sargeant_race_2023` | `0.293s/lap` | `0.214s/lap` | `0.749s/lap` | `0.729s/lap` |
| `albon_sargeant_race_2024` | `0.380s/lap` | `0.264s/lap` | `0.482s/lap` | `0.472s/lap` |

Interpretation:

- removing the strict pairing rules can move the season summary materially;
- `VER-PER 2023`, `ALB-SAR 2023`, and `ALB-SAR 2024` move toward or beyond the
  published source rows under the broad valid-lap mean;
- `ALO-STR 2023` does not, so "the local extractor is simply too narrow" is
  not a complete explanation for every failed race row.

The broad probe is useful as a sensitivity test, but it is not a replacement
construct. It produces obviously polluted session numbers when the two drivers
do not have comparable retained lap sets:

- `ALB-SAR 2023` at Zandvoort: broad valid-lap mean `11.082s/lap`, with only
  six Sargeant laps left;
- `ALO-STR 2024` in Qatar: broad valid-lap mean `4.935s/lap`, with only one
  Stroll lap left;
- `VER-PER 2023` in Japan: broad valid-lap mean `3.176s/lap`, with only seven
  Perez laps left.

Those rows show why the current paired extractor exists. They also show why a
generic published "race pace" summary is not automatically the same object as a
strict paired residual.

The race-side conclusion is therefore narrower than the qualifying conclusion:

- the current race rows still carry useful external information;
- the current evidence does not prove they are same-construct acceptance gates
  for the paired extractor;
- the broad probe supports construct sensitivity, but not a clean alternate
  estimator that should replace the current race residual.

## What Should Change Next

### 1. Stop treating the current PACETEQ rows as acceptance gates

Implemented on 2026-05-17: the current PACETEQ race and qualifying rows are
now reported as `EXTERNAL_CONTEXT`, not HARD gates.

Do not relax the numbers just to turn the Phase 6 boolean green.

### 2. Decide what `quali_rating_mu_s` is supposed to mean

There are two defensible options:

#### Option A: peak qualifying skill

If prediction wants "best lap when it matters," change the qualifying
extractor toward a single best comparable lap per session. Then find or build
validation rows that measure that same object.

#### Option B: repeatable qualifying execution

If the current multi-run median is intentional, keep the extractor and stop
validating it against best-lap thresholds. Use same-phase comparable-session
sources, locally reproducible replay checks, or a dedicated same-construct
source set instead.

Decision recorded on 2026-05-17: keep repeatable qualifying execution for v1.
Peak qualifying should return later as a separate candidate model, not as a
patch to make mismatched source rows pass.

### 3. Tighten the validation evidence schema

Every future source-backed row should record:

- lap selection unit: `best_lap`, `representative_laps`, `paired_laps`, or
  `unknown`;
- qualifying segment policy;
- weather policy;
- season aggregation rule;
- weighting rule;
- exact sample count;
- whether the row is same-construct with the local extractor.

Without those fields, "HARD" only means "the source has a number," not "the
source can grade this model."

### 4. Use the new construct probes before changing extractor semantics

Implemented on 2026-05-17:

- the locked-row probe compares current, highest-common-best, and any-valid-best
  views for the audited external rows;
- the network-wide probe measures coverage, segment mix, dispersion, weighting
  shifts, and peak-session outliers across every stored qualifying pair-season.

The broader probe now makes the tradeoff explicit: the peak comparable view adds
coverage, but it is also noisier and still admits implausible single-session
outliers. If the project chooses a peak qualifying target, the next extractor
work should tighten single-lap validity and redesign uncertainty before refit.

### 5. Re-audit the race rows before calling them HARD

Implemented in part on 2026-05-17: the local race probe now shows that broader
valid-lap summaries can move several rows materially, but also that those broad
summaries become unreliable when retained lap counts are badly imbalanced.

For race pace, the remaining decision still needs evidence that the source
either:

- uses effectively the same lap controls as the local paired extractor; or
- is intentionally accepted as a broader contextual target, in which case it
  should not be a pass/fail gate for the narrow extractor.

### 6. Summer-break revisit

Reminder: revisit peak qualifying as a separate candidate-model track on
`2026-07-27`, the first day of the 2026 F1 summer break after the Hungarian
Grand Prix.

The review question is not "how do we make the old PACETEQ thresholds pass?"
It is:

> does a peak-qualifying candidate with stricter single-lap validity, its own
> uncertainty model, replay stability, and held-out-season evidence outperform
> the v1 repeatable-execution model enough to justify a second model path?

## Bottom Line

The Phase 6 failure is real, but the first explanation is not "the extractor is
too conservative." The stronger explanation is:

> the current validation set is grading a WLS-weighted, multi-run matched-lap
> estimator against season-summary external rows that often measure best-lap or
> broader pace constructs.

Decision recorded on 2026-05-17:

- keep the current race and qualifying extractor semantics for v1;
- treat the current Phase 6 boolean failure as a validation-contract failure,
  not as proof that the extractor is wrong;
- continue the pipeline with internal reproducibility checks, replay stability,
  held-out-season diagnostics, and source-backed external rows only when the
  construct match is genuinely aligned;
- revisit peak qualifying later as a separate candidate model.

That gap is now fixed in the contract before Phase 7 work continues.

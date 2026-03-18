# Progress Report

## 2026-03-18

### Completed

- Fixed `scripts/validate_characteristics.py` so it now resolves the season-scoped repo layout first instead of hardcoding the legacy flat driver path.
- Kept a safe fallback for `data/processed/driver_characteristics.json` so older flows still validate.
- Added `--season-year` support and latest-season auto-detection for season-scoped characteristics files.
- Added focused tests for the season-scoped layout and the legacy driver fallback.
- Fixed weekend-type resolution to fail closed instead of silently defaulting unknown races to conventional weekends.
- Updated warmup to skip targets with unknown weekend format and record an explicit error instead of precomputing against the wrong branch.
- Updated qualifying requests to raise a clear weekend-format error instead of silently forcing the normal-weekend path.
- Taught the schedule helper to supplement incomplete FastF1 schedules with local track fallback rows, which was needed because the current 2026 FastF1 snapshot here does not include Bahrain.
- Added focused tests for fail-closed weekend lookups, warmup skipping, predictor error handling, and incomplete-schedule fallback supplementation.
- Switched the dashboard prediction path to persisted-only serving. Clicking `Predict weekend` now loads warmed artifacts instead of mutating state or rerunning simulations.
- Disabled request-path force refresh and made the error explicit so manual recovery stays in Render cron / warmup scripts, not the UI.
- Changed the boundary-lag behavior to serve the latest warmed checkpoint instead of generating an inline current-boundary prediction.
- Updated dashboard copy and docs to match the actual product contract: warmup/cron owns freshness, the request path stays read-only.
- Removed the most obvious AI-style UI fingerprints from the dashboard: emoji-heavy warning chrome, the `Prediction lab` label, and the over-polished hero summary copy.
- Kept the layout and information density intact while making runtime notices and forecast sections sound more direct and less templated.
- Reviewed the persisted-only and UI-tone changes again before calling them stable, focusing on regression risk and operator trust.
- Fixed a same-boundary cache staleness bug by rechecking persisted storage on each request and only treating the RAM entry as a cache hit when the persisted write timestamp still matches.
- Removed the dead `dashboard.prediction_precompute.inline_enabled` story from config and docs so the code, configuration, and product contract all say the same thing.
- Rewrote the persisted-prediction failure hint so it points operators to warmup first and keeps `--require-db` as an optional deployment constraint, not the default story.
- Added focused tests for same-boundary persisted rewrites, read-only request-path behavior, and the updated failure guidance.
- Updated the warmup-pending state to distinguish "older warmup exists for a different artifact set" from "nothing has been warmed yet," which is the real reason the dashboard can relock after local artifact or config changes.
- Updated the Development Over Time chart to include sprint checkpoints in the time series while keeping the separate latest-snapshot comparison logic conservative.
- Fixed Development Over Time cache staleness so out-of-band snapshot backfills invalidate the cached history on the next rerun instead of waiting for the full Streamlit TTL window.
- Rebuilt the 2026 local warmup horizon for the current artifact hash and verified the persisted Japanese/Miami/Canada checkpoint set is present again.

### Verification

- `uv run pytest tests/test_validate_characteristics_script.py -q`
- `uv run python scripts/validate_characteristics.py --data-dir data/processed`
- `uv run pytest tests/test_weekend_module.py tests/test_warmup_precompute.py tests/test_baseline_2026_integration.py -q`
- `uv run pytest tests/test_dashboard_pipeline.py tests/test_dashboard_pages.py -q`
- `uv run pytest tests/test_validate_characteristics_script.py tests/test_weekend_module.py tests/test_warmup_precompute.py tests/test_baseline_2026_integration.py tests/test_dashboard_pipeline.py tests/test_dashboard_pages.py -q`
- `uv run pytest tests/test_dashboard_rendering.py tests/test_dashboard_pages.py tests/test_dashboard_pipeline.py -q`
- `uv run python -m py_compile src/dashboard/live_prediction_flow.py src/dashboard/precomputed_predictions.py src/dashboard/pages.py tests/test_dashboard_pipeline.py tests/test_dashboard_pages.py`
- `uv run pytest tests/test_dashboard_pipeline.py tests/test_dashboard_pages.py tests/test_dashboard_rendering.py -q`
- `uv run pytest tests/test_validate_characteristics_script.py tests/test_weekend_module.py tests/test_warmup_precompute.py tests/test_baseline_2026_integration.py tests/test_dashboard_rendering.py tests/test_dashboard_pages.py tests/test_dashboard_pipeline.py -q`
- `uv run pytest tests/test_dashboard_pages.py tests/test_team_comparison_module.py -q`
- `uv run pytest tests/test_dashboard_pipeline.py tests/test_dashboard_pages.py tests/test_dashboard_rendering.py tests/test_team_comparison_module.py -q`
- `uv run python scripts/warmup_precompute.py --year 2026 --verbose`
- `uv run pytest tests/test_team_comparison_module.py -q`
- Current repo result: validation now passes against the shipped `data/processed` layout. The run still reports driver expectation warnings, but they are warnings, not blocking errors.
- Current repo result: the weekend-resolution slice passes with 33 tests, including warmup and predictor coverage.
- Current repo result: the combined dashboard + validator + warmup regression slice passes with 91 tests.
- Current repo result: the dashboard rendering/page slice passes with 67 tests after the copy and icon cleanup.
- Current repo result: the dashboard read-only slice passes with 68 tests after the cache-validation follow-up.
- Current repo result: the combined dashboard + validator + warmup regression slice now passes with 106 tests after the review-driven fixes.
- Current repo result: the pages + team-comparison slice passes with 74 tests after the warmup-state and sprint-history follow-up.
- Current repo result: the broader dashboard slice passes with 97 tests after the same follow-up.
- Current repo result: the team-comparison slice passes with 28 tests after adding cache invalidation coverage for out-of-band snapshot writes.
- Current repo result: local warmup now reports `status=success` for the current 2026 artifact hash, with ready races `Japanese Grand Prix`, `Miami Grand Prix`, and `Canadian Grand Prix`.

### Learnings

- The validator failure was a file-resolution contract bug, not a predictor bug.
- There are still other legacy `driver_characteristics.json` references in the repo. Those need separate passes so we do not mix a safe validator fix with broader behavior changes.
- FIXME item 1 is not a safe code-only patch. It needs a product decision about the button contract first, so I started with item 2 to avoid breaking the app.
- The weekend fallback issue was real, but the first pass exposed a second bug too: FastF1 was returning an incomplete 2026 schedule in this environment, so fail-closed behavior had to be paired with explicit fallback supplementation for missing races.
- Silent fallback is only safe when the fallback itself is trustworthy. In this repo, the schedule-window data in automation is explicit enough to reuse, but guessing "conventional" was not.
- The product contract matters more than any one refactor. The repo had drifted into a hidden hybrid where the UI claimed persisted predictions while the request path still behaved like a refresh path.
- Persisted-only serving removes one class of instability, but not every live dependency. The dashboard still performs lightweight boundary checks, so the docs now say that plainly instead of pretending the click path is fully offline.
- The fastest visible quality win was not a redesign. It was deleting the little things that make software sound autogenerated: emoji prefixes, inflated labels, and copy that narrates the layout instead of the product.
- Persisted-only serving still needs an explicit freshness rule inside the in-memory cache. A boundary key alone is not enough when persisted artifacts can be rewritten under the same checkpoint.
- Config drift is a credibility bug. Leaving a dead toggle in `default.yaml` and docs makes the repo look less intentional even when runtime behavior is correct.
- Focused reruns right after cleanup pay off. This review caught a small renderer import regression before it turned into a user-visible break.
- A "should already be warmed" complaint can still be true from the user's point of view and false from the system's point of view. In this repo, changing artifact inputs changes the warmup hash, so old warmed predictions no longer count for the current state.
- The development chart and the latest-comparison snapshot should not share the exact same session filter. Sprint checkpoints are useful in the time series, but they are not automatically the right default for the headline comparison payload.
- The remaining missing `Chinese Grand Prix R` point was not missing source data. The stored snapshot existed, but the dashboard kept a stale cached snapshot list after an out-of-band backfill.
- When local warmup succeeds but the page still shows the old lock state, the right first question is whether the dashboard reran against the same storage backend. The local read path now resolves to an unlocked warmed state for Japan.

### Next Candidate

- Continue through the remaining FIXME items one at a time, now that the prediction-path contract is explicit.

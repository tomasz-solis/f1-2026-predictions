# Warmup Precompute Worker

Use this worker to precompute prediction payloads outside Streamlit request handling.

## Why it exists

- Keeps the dashboard race dropdown focused on warmed races.
- Avoids first-click compute delays by pre-filling checkpoint-aware prediction rows.
- Runs safely from cron/worker processes without blocking app requests.

## Command

```bash
python scripts/warmup_precompute.py --year 2026
```

Useful flags:

```bash
python scripts/warmup_precompute.py --year 2026 --dry-run --verbose
python scripts/warmup_precompute.py --year 2026 --verbose --no-verify-writes
python scripts/warmup_precompute.py --year 2026 --require-db
```

`--dry-run` computes a plan only (no writes).
`--verbose` prints per-target checkpoint/boundary context and reuse/compute counts.
`--require-db` fails fast when storage mode is not DB-backed, or if write verification warns.
By default, writes are verified with immediate DB read-back when DB mode supports it.

Exit codes:

- `0`: success, nothing-to-do, or checkpoint-not-ready
- non-zero: unexpected runtime failure, or `--require-db` validation failure

## Warmup behavior

Each run:

1. Loads the current calendar and selects the next upcoming race as anchor.
2. Builds a 3-race horizon (`anchor + 2`).
3. Resolves checkpoint readiness from real session data:
   - conventional: `PRE`, `FP1`, `FP2`, `FP3`, `Q`
   - sprint: `PRE`, `FP1`, `SQ`, `Sprint`, `Q`
4. If expected checkpoint data is not ready, writes a throttled lightweight status and exits.
5. If ready, for each target race:
   - computes/stores base features once per key
   - computes/stores missing weather scenarios (`dry`, `mixed`, `rain`) only
   - uses each target race's own boundary signature/checkpoint key (no anchor-boundary reuse)
6. Uses a distributed DB lock (when DB writes are enabled) to avoid overlapping workers.
7. Updates horizon index (`ready_races`) for dropdown filtering.

PRE behavior:

- Missing PRE scenarios are computed immediately when the script runs.
- This includes Thursday runs before the first race weekend starts.

## Scheduling

For production preheat, run it every 5 minutes so completed `Q`/`SQ`/`Sprint`
sessions are warmed quickly after the boundary flips:

```bash
*/5 * * * * cd /path/to/formula1-2026 && /path/to/formula1-2026/.venv/bin/python scripts/warmup_precompute.py --year 2026 --require-db >> /tmp/f1_warmup.log 2>&1
```

If you use multiple workers, keep the same cadence; warmup writes are idempotent.

## Supabase/Render note

For multi-instance deployments (Render web + worker), set `USE_DB_STORAGE` to a DB-backed mode:

- `fallback`
- `dual_write`
- `db_only`

`file_only` does not share warmup state across instances.

To keep user requests fast, disable inline warmup in the Streamlit path (`dashboard.prediction_precompute.inline_enabled: false`) and rely on this worker.

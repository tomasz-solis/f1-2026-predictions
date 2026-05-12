#!/usr/bin/env bash
# Local Phase 2 smoke-session inspector run.
# Run this from the repo root after activating .venv.
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-data/raw/.fastf1_cache}"
OUT_DIR="${OUT_DIR:-data/diagnostics/smoke_session_inspections}"

mkdir -p "$CACHE_DIR" "$OUT_DIR"

echo "FastF1 cache: $CACHE_DIR"
echo "Inspections out: $OUT_DIR"
echo

.venv/bin/python scripts/inspect_smoke_sessions.py \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/run.log"

echo
echo "Listing output files:"
ls -la "$OUT_DIR"

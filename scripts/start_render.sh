#!/usr/bin/env bash
set -euo pipefail

# Force root-path serving on Render; stale env overrides can make `/` return 404.
unset STREAMLIT_SERVER_BASE_URL_PATH

exec streamlit run app.py \
  --server.port "${PORT:-10000}" \
  --server.address 0.0.0.0 \
  --server.headless true

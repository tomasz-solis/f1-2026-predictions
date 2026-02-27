#!/usr/bin/env bash
set -euo pipefail

# Force root-path serving on Render; stale env overrides can make `/` return 404.
unset STREAMLIT_SERVER_BASE_URL_PATH

python3 - <<'PY'
import sys

if sys.version_info >= (3, 14):
    raise SystemExit(
        "Unsupported runtime detected: Python "
        f"{sys.version.split()[0]}. This project targets Python 3.11.x (requires-python <3.14)."
    )
PY

exec streamlit run app.py \
  --server.port "${PORT:-10000}" \
  --server.address 0.0.0.0 \
  --server.headless true

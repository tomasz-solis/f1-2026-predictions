"""
Trackside Labs Streamlit Dashboard for F1 2026 Predictions

Live race predictions with historical accuracy tracking.
"""

import logging
import os
from pathlib import Path

import streamlit as st


def _load_local_env(path: str = ".env.local") -> None:
    """Load KEY=VALUE lines from a local env file into the process environment.

    `streamlit run` does not read .env files, so without this the app starts in
    file_only persistence mode (USE_DB_STORAGE unset) and never reads the
    Supabase-backed forecast store. Real environment variables always win — we only
    fill values that aren't already set. No dependency: the file is trivial KEY=VALUE.
    """
    env_file = Path(path)
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Must run before importing anything under `src`: the persistence layer validates
# USE_DB_STORAGE / SUPABASE_* at import time, so the env has to be populated first.
_load_local_env()

from src.dashboard import (  # noqa: E402
    BRAND_NAME,
    configure_page,
    render_analytics_scripts,
    render_global_styles,
    render_header,
    render_page,
    render_sidebar,
)
from src.dashboard.analytics import track_page_view  # noqa: E402

logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1.api").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

configure_page()
render_global_styles()
render_analytics_scripts()
render_header()

page, enable_logging = render_sidebar()
track_page_view(page)
render_page(page, enable_logging)

st.markdown(
    (f'<div class="brand-footer">{BRAND_NAME} | independent motorsport forecasting project</div>'),
    unsafe_allow_html=True,
)

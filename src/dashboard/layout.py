"""Dashboard layout, global styling, and navigation sidebar."""

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

# Brand asset config: update these filenames when you want to swap branding.
BRAND_NAME = "Trackside Labs"
BRAND_PAGE_TITLE = f"{BRAND_NAME} | Motorsport Forecasting"
BRAND_ASSET_DIRS = (Path("assets/logis"), Path("assets/logos"))
BRAND_WORDMARK_FILE = "trackside-labs_wordmark_w800.png"
BRAND_FAVICON_FILE = "trackside-labs_mark_32.png"
BRAND_WORDMARK_ALT = "Trackside Labs wordmark"
BRAND_TAGLINE = "Motorsport data forecasting and telemetry insights"
BRAND_DISCLAIMER = "Independent analytics project • not affiliated with any racing series, teams, or governing bodies"
# Header alignment toggle. Options: "left" or "center".
BRAND_HEADER_ALIGNMENT = "left"
BRAND_MODEL_VERSION = "v1.2"
BRAND_LAST_UPDATED = "2026-02-23"
NAVIGATION_PAGES = [
    "Live Prediction",
    "Model & Learning",
    "Prediction Accuracy",
    "About",
]

_CUSTOM_CSS = """
<style>
/* ---- Brand tokens (assets/BRAND_GUIDE.md) ---- */
:root {
  --ts-graphite: #0B0F14;
  --ts-soft-light: #E8EDF2;
  --ts-steel: #8B949E;
  --ts-heat: #FF4D2D;
  --ts-panel: #111826;
  --ts-panel-alt: #0f1623;
  --ts-border: rgba(232,237,242,0.12);
}

/* ---- App background ---- */
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(120% 90% at 10% 0%, #101727 0%, #0B111D 38%, var(--ts-graphite) 72%);
    color: var(--ts-soft-light);
}
[data-testid="stHeader"] {
    background: rgba(11,15,20,0.0);
}
[data-testid="stSidebar"] {
    background: #0E1521;
    border-right: 1px solid rgba(255,255,255,0.09);
}

/* ---- Shared page rail ---- */
[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 1240px;
  padding-top: 1.8rem;
  padding-right: 2.6rem;
  padding-left: 2.6rem;
  padding-bottom: 2.4rem;
}

@media (max-width: 1080px) {
  [data-testid="stAppViewContainer"] .main .block-container {
    padding-right: 1.6rem;
    padding-left: 1.6rem;
  }
}

@media (max-width: 760px) {
  [data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 1rem;
    padding-right: 1rem;
    padding-left: 1rem;
  }
}

/* ---- Typography ---- */
html, body, [class*="css"] {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.main-header {
  font-size: 2.1rem;
  font-weight: 760;
  letter-spacing: 0.02em;
  text-align: left;
  margin: 0 0 0.15rem 0;
  color: var(--ts-soft-light);
}
.sub-header {
  font-size: 1.03rem;
  line-height: 1.45;
  max-width: 62ch;
  text-align: left;
  color: rgba(232,237,242,0.94);
  margin: 0 0 0.35rem 0;
  letter-spacing: 0.01em;
}
.micro-disclaimer {
  font-size: 0.86rem;
  line-height: 1.5;
  max-width: 90ch;
  text-align: left;
  color: rgba(139,148,158,0.95);
  margin: 0 0 1.55rem 0;
}
.brand-shell {
  margin: 0 0 0.75rem 0;
  padding-top: 0.2rem;
}
.brand-row {
  display: flex;
  align-items: flex-end;
  justify-content: flex-start;
  margin: 0 0 0.95rem 0;
}
.brand-logo {
  width: clamp(340px, 44vw, 700px);
  max-width: 100%;
  height: auto;
  display: block;
}

.brand-shell--center .brand-row {
  justify-content: center;
}
.brand-shell--center .sub-header,
.brand-shell--center .micro-disclaimer,
.brand-shell--center .main-header {
  text-align: center;
  margin-left: auto;
  margin-right: auto;
}
.brand-shell--center .brand-logo {
  width: min(560px, 90vw);
}

@media (max-width: 960px) {
  .brand-logo {
    width: min(520px, 92vw);
  }
}

/* Stronger global text defaults (Streamlit DOM-safe) */
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"] *,
[data-testid="stCaptionContainer"] *,
[data-testid="stHeader"] *,
[data-testid="stToolbar"] * {
  color: rgba(232,237,242,0.9);
}

/* Headings */
h1, h2, h3, h4 {
  color: var(--ts-soft-light) !important;
  letter-spacing: 0.01em;
}
h2 {
  margin-top: 0.85rem;
  margin-bottom: 1rem;
}

[data-testid="stRadio"] {
  margin-bottom: 0.65rem;
}
[data-testid="stExpander"] {
  margin-bottom: 1.7rem;
}

/* ---- Sidebar controls contrast ---- */
[data-testid="stSidebar"] * {
  color: rgba(232,237,242,0.86) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  color: var(--ts-soft-light) !important;
  font-weight: 700 !important;
}

/* ---- Inputs (readable values + placeholders) ---- */
[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div {
  background: var(--ts-panel-alt) !important;
  border-radius: 12px !important;
  border: 1px solid var(--ts-border) !important;
}
[data-baseweb="select"] * {
  color: rgba(232,237,242,0.94) !important;
}
[data-baseweb="select"] input::placeholder {
  color: rgba(232,237,242,0.45) !important;
}
label {
  color: rgba(232,237,242,0.82) !important;
}

/* Primary button */
.stButton > button {
  background: var(--ts-heat) !important;
  color: #fff !important;
  border: 0 !important;
  border-radius: 11px !important;
  font-weight: 680 !important;
  padding: 0.55rem 1.05rem !important;
}
.stButton > button:hover {
  background: #ff6548 !important;
}


/* ---- Tables ---- */
[data-testid="stDataFrame"] {
  background: rgba(16,22,34,0.78);
  border: 1px solid var(--ts-border);
  border-radius: 16px;
  padding: 0.35rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

/* The grid itself */
[data-testid="stDataFrame"] [role="grid"] {
  background: var(--ts-panel-alt) !important;
  color: rgba(232,237,242,0.9) !important;
  border-radius: 12px;
}

/* ---- HTML tables (our controlled dark tables) ---- */
.rc-table {
  background: rgba(16,22,34,0.78);
  border: 1px solid var(--ts-border);
  border-radius: 16px;
  padding: 0.6rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  overflow: hidden;
}

.rc-table table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.92rem;
  color: rgba(232,237,242,0.9);
}

.rc-table thead th {
  background: var(--ts-panel);
  color: rgba(232,237,242,0.95);
  text-align: left;
  font-weight: 700;
  padding: 0.55rem 0.7rem;
  border-bottom: 1px solid rgba(255,255,255,0.10);
}

.rc-table tbody tr:nth-child(even) td {
  background: rgba(15,22,35,0.92);
}

.rc-table table th:first-child,
.rc-table table td:first-child {
  width: 64px;
  text-align: center;
  font-weight: 800;
  color: rgba(232,237,242,0.94);
}

.rc-table table td:nth-child(2) {
  font-weight: 700;
  letter-spacing: 0.2px;
}

.rc-table tbody tr:hover td {
  background: rgba(255,255,255,0.03);
}

.rc-table { overflow: auto; }

.rc-table table th:first-child,
.rc-table table td:first-child {
  position: sticky;
  left: 0;
  z-index: 3;
  background: var(--ts-panel-alt);
}

.rc-table table th:nth-child(2),
.rc-table table td:nth-child(2) {
  position: sticky;
  left: 64px; /* same as Pos width */
  z-index: 2;
  background: var(--ts-panel-alt);
}

/* Header row */
[data-testid="stDataFrame"] [role="columnheader"] {
  background: var(--ts-panel) !important;
  color: rgba(232,237,242,0.95) !important;
  border-bottom: 1px solid rgba(255,255,255,0.10) !important;
}

/* Body cells */
[data-testid="stDataFrame"] [role="gridcell"] {
  background: var(--ts-panel-alt) !important;
  color: rgba(232,237,242,0.9) !important;
  border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}

/* Hover */
[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] {
  background: rgba(255,255,255,0.03) !important;
}

/* Hide/neutralize Streamlit spinner/status blocks that show as white bars */
[data-testid="stSpinner"] {
  background: transparent !important;
}
[data-testid="stSpinner"] > div {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.stCacheStatus, [data-testid="stStatusWidget"] {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

/* hide Streamlit footer/menu noise */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Cache + status widgets */
.stCacheStatus, [data-testid="stStatusWidget"] { display: none !important; }

/* Footer variants */
footer, [data-testid="stFooter"] { display: none !important; }
#MainMenu { visibility: hidden !important; }

/* Custom footer on same content rail */
.brand-footer {
  margin-top: 2.6rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(232,237,242,0.12);
  color: rgba(139,148,158,0.95);
  font-size: 0.96rem;
  letter-spacing: 0.005em;
  text-align: left;
}

</style>
"""


def _brand_asset_path(filename: str) -> Path:
    for asset_dir in BRAND_ASSET_DIRS:
        candidate = asset_dir / filename
        if candidate.exists():
            return candidate
    return BRAND_ASSET_DIRS[0] / filename


def _page_icon() -> str:
    icon_path = _brand_asset_path(BRAND_FAVICON_FILE)
    return str(icon_path) if icon_path.exists() else "F1"


def _header_alignment() -> str:
    alignment = BRAND_HEADER_ALIGNMENT.strip().lower()
    return alignment if alignment in {"left", "center"} else "left"


def configure_page() -> None:
    st.set_page_config(
        page_title=BRAND_PAGE_TITLE,
        page_icon=_page_icon(),
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_global_styles() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


@lru_cache(maxsize=4)
def _build_asset_data_uri(path_str: str) -> str:
    asset_path = Path(path_str)
    suffix = asset_path.suffix.lower()
    mime = "image/svg+xml" if suffix == ".svg" else "image/png"
    encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_header() -> None:
    shell_class = f"brand-shell brand-shell--{_header_alignment()}"
    logo_path = _brand_asset_path(BRAND_WORDMARK_FILE)
    if logo_path.exists():
        logo_data_uri = _build_asset_data_uri(str(logo_path))
        st.markdown(
            (
                f'<div class="{shell_class}">'
                '<div class="brand-row">'
                f'<img class="brand-logo" src="{logo_data_uri}" alt="{BRAND_WORDMARK_ALT}" />'
                "</div>"
                f'<div class="sub-header">{BRAND_TAGLINE}</div>'
                f'<div class="micro-disclaimer">{BRAND_DISCLAIMER}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            (
                f'<div class="{shell_class}">'
                f'<div class="main-header">{BRAND_NAME}</div>'
                f'<div class="sub-header">{BRAND_TAGLINE}</div>'
                f'<div class="micro-disclaimer">{BRAND_DISCLAIMER}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_sidebar() -> tuple[str, bool]:
    page = st.radio(
        "Navigation",
        NAVIGATION_PAGES,
        horizontal=True,
    )

    with st.expander("Settings", expanded=False):
        enable_logging = st.checkbox(
            "Save Predictions for Accuracy Tracking",
            value=False,
            help=(
                "When enabled, predictions are saved after each session (FP1/FP2/FP3/SQ) "
                "for later accuracy analysis. Max 1 prediction per session."
            ),
        )
        st.markdown(f"**Model Version:** {BRAND_MODEL_VERSION}")
        st.markdown(f"**Last Updated:** {BRAND_LAST_UPDATED}")

    return page, enable_logging

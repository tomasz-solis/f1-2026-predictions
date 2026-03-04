"""Dashboard layout, global styling, and navigation sidebar."""

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from .styles import CUSTOM_CSS

# Brand asset config: update these filenames when you want to swap branding.
BRAND_NAME = "Trackside Labs"
BRAND_PAGE_TITLE = f"{BRAND_NAME} | Motorsport Forecasting"
BRAND_ASSET_DIRS = (Path("assets/logos"),)
BRAND_WORDMARK_FILE = "trackside-labs_wordmark_w800.png"
BRAND_FAVICON_FILE = "trackside-labs_mark_32.png"
BRAND_WORDMARK_ALT = "Trackside Labs wordmark"
BRAND_TAGLINE = "Motorsport data forecasting and telemetry insights"
BRAND_DISCLAIMER = "Independent analytics project • not affiliated with any racing series, teams, or governing bodies"
# Header alignment toggle. Options: "left" or "center".
BRAND_HEADER_ALIGNMENT = "left"
BRAND_MODEL_VERSION = "v1.3"
BRAND_LAST_UPDATED = "2026-03-04"
ENABLE_PREDICTION_ACCURACY_TAB = False
NAVIGATION_PAGES = ["Prediction", "Team Comparison"]
if ENABLE_PREDICTION_ACCURACY_TAB:
    NAVIGATION_PAGES.append("Prediction Accuracy")
NAVIGATION_PAGES.extend(["Model & Learning", "Contact"])

# Backwards-compatible alias used by tests and external imports.
_CUSTOM_CSS = CUSTOM_CSS


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
        header_html = (
            '<div class="ts-sticky-header" id="tsStickyHeader">'
            f'<div class="{shell_class}">'
            '<div class="brand-row">'
            f'<img class="brand-logo" src="{logo_data_uri}" alt="{BRAND_WORDMARK_ALT}" />'
            "</div>"
            f'<div class="sub-header">{BRAND_TAGLINE}</div>'
            f'<div class="micro-disclaimer">{BRAND_DISCLAIMER}</div>'
            "</div>"
            "</div>"
        )
    else:
        header_html = (
            '<div class="ts-sticky-header" id="tsStickyHeader">'
            f'<div class="{shell_class}">'
            f'<div class="main-header">{BRAND_NAME}</div>'
            f'<div class="sub-header">{BRAND_TAGLINE}</div>'
            f'<div class="micro-disclaimer">{BRAND_DISCLAIMER}</div>'
            "</div>"
            "</div>"
        )

    st.markdown(header_html, unsafe_allow_html=True)

    components.html(
        """
        <script>
        (function () {
        // Prevent double-binding
        if (window.__tsHeaderScrollBound) return;
        window.__tsHeaderScrollBound = true;

        const thresholdDesktop = 90;
        const thresholdMobile = 50;

        function isMobile() {
            return window.matchMedia && window.matchMedia("(max-width: 760px)").matches;
        }

        function bind() {
            const header = document.getElementById("tsStickyHeader");
            if (!header) return;

            function onScroll() {
            const y = window.scrollY || document.documentElement.scrollTop || 0;
            const threshold = isMobile() ? thresholdMobile : thresholdDesktop;
            if (y > threshold) header.classList.add("is-collapsed");
            else header.classList.remove("is-collapsed");
            }

            window.addEventListener("scroll", onScroll, { passive: true });
            onScroll();
        }

        // Streamlit sometimes renders after JS runs, so retry a few times.
        let tries = 0;
        const t = setInterval(() => {
            tries += 1;
            bind();
            if (document.getElementById("tsStickyHeader") || tries > 20) clearInterval(t);
        }, 150);
        })();
        </script>
        """,
        height=0,
    )


def render_sidebar() -> tuple[str, bool]:
    segmented_control = getattr(st, "segmented_control", None)
    page: str | None = None
    if callable(segmented_control):
        try:
            selection = segmented_control(
                "Navigation",
                options=NAVIGATION_PAGES,
                selection_mode="single",
                default=NAVIGATION_PAGES[0],
                key="nav_tabs",
            )
        except TypeError:
            selection = segmented_control(
                "Navigation",
                options=NAVIGATION_PAGES,
                default=NAVIGATION_PAGES[0],
                key="nav_tabs",
            )

        if isinstance(selection, str) and selection in NAVIGATION_PAGES:
            page = selection

    if page is None:
        page = st.selectbox(
            "Navigation",
            NAVIGATION_PAGES,
            index=0,
            key="nav_tabs",
            label_visibility="collapsed",
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

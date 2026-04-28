"""Dashboard layout, global styling, and navigation sidebar."""

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.utils.model_version import format_model_version_label

from .styles import CUSTOM_CSS

# Brand asset filenames.
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
BRAND_MODEL_VERSION = format_model_version_label()
BRAND_LAST_UPDATED = "Unavailable"
ENABLE_PREDICTION_ACCURACY_TAB = True
NAVIGATION_PAGES = ["Prediction", "Team Comparison"]
if ENABLE_PREDICTION_ACCURACY_TAB:
    NAVIGATION_PAGES.append("Prediction Accuracy")
    NAVIGATION_PAGES.append("Checkpoint Viewer")
NAVIGATION_PAGES.extend(["Model & Learning", "Contact"])
_NAVIGATION_STATE_KEY = "nav_tabs"
_NAVIGATION_WIDGET_KEY = "nav_tabs_selector"
_NAVIGATION_FALLBACK_KEY = "nav_tabs_selectbox"

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
    """Set Streamlit page config and favicon."""
    st.set_page_config(
        page_title=BRAND_PAGE_TITLE,
        page_icon=_page_icon(),
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_global_styles() -> None:
    """Inject custom CSS into the Streamlit page."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def render_analytics_scripts() -> None:
    """Inject the GoatCounter pageview script into the parent document.

    components.html runs inside a sandboxed iframe; without
    window.parent.document the tracker would count iframe loads, not the
    Streamlit app. The flag on window.parent prevents double-loading on
    Streamlit reruns.
    """
    components.html(
        """
        <script>
        (function () {
          if (window.parent.__goatcounterLoaded) return;
          window.parent.__goatcounterLoaded = true;
          const s = window.parent.document.createElement('script');
          s.async = true;
          s.dataset.goatcounter = 'https://tracksidelabs.goatcounter.com/count';
          s.src = 'https://gc.zgo.at/count.js';
          window.parent.document.head.appendChild(s);
        })();
        </script>
        """,
        height=0,
    )


@lru_cache(maxsize=4)
def _build_asset_data_uri(path_str: str) -> str:
    asset_path = Path(path_str)
    suffix = asset_path.suffix.lower()
    mime = "image/svg+xml" if suffix == ".svg" else "image/png"
    encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_header() -> None:
    """Render the sticky header with logo, tagline, and disclaimer."""
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


def _coerce_navigation_page(value: object) -> str | None:
    """Return a valid navigation page from a widget value."""
    if isinstance(value, str) and value in NAVIGATION_PAGES:
        return value
    return None


def _active_navigation_page(default: str | None = None) -> str:
    """Return the last valid navigation page stored in session state."""
    fallback = default if default in NAVIGATION_PAGES else NAVIGATION_PAGES[0]
    try:
        stored_page = st.session_state.get(_NAVIGATION_STATE_KEY)
    except Exception:
        stored_page = None
    return _coerce_navigation_page(stored_page) or fallback


def _store_navigation_page(page: str) -> None:
    """Remember the active navigation page when session state is available."""
    if page not in NAVIGATION_PAGES:
        return
    try:
        st.session_state[_NAVIGATION_STATE_KEY] = page
    except Exception:
        return


def _sync_navigation_widget() -> None:
    """Pin nullable segmented-control state to the last valid page."""
    try:
        raw_page = st.session_state.get(_NAVIGATION_WIDGET_KEY)
    except Exception:
        return

    page = _coerce_navigation_page(raw_page)
    if page is None:
        page = _active_navigation_page()
        try:
            st.session_state[_NAVIGATION_WIDGET_KEY] = page
        except Exception:
            pass
    _store_navigation_page(page)


def render_sidebar() -> tuple[str, bool]:
    """Render sidebar navigation and return selected page with logging enabled."""
    segmented_control = getattr(st, "segmented_control", None)
    active_page = _active_navigation_page()
    if callable(segmented_control):
        try:
            selection = segmented_control(
                "Navigation",
                options=NAVIGATION_PAGES,
                selection_mode="single",
                default=active_page,
                key=_NAVIGATION_WIDGET_KEY,
                on_change=_sync_navigation_widget,
            )
        except TypeError:
            try:
                selection = segmented_control(
                    "Navigation",
                    options=NAVIGATION_PAGES,
                    default=active_page,
                    key=_NAVIGATION_WIDGET_KEY,
                    on_change=_sync_navigation_widget,
                )
            except TypeError:
                selection = segmented_control(
                    "Navigation",
                    options=NAVIGATION_PAGES,
                    default=active_page,
                    key=_NAVIGATION_WIDGET_KEY,
                )

        page = _coerce_navigation_page(selection) or _active_navigation_page(active_page)
        _store_navigation_page(page)
        return page, True

    page = st.selectbox(
        "Navigation",
        NAVIGATION_PAGES,
        index=NAVIGATION_PAGES.index(active_page),
        key=_NAVIGATION_FALLBACK_KEY,
        label_visibility="collapsed",
    )
    page = _coerce_navigation_page(page) or active_page
    _store_navigation_page(page)

    return page, True

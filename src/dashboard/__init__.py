"""Dashboard modules for Streamlit app composition."""

from .cache import enable_fastf1_cache
from .layout import (
    BRAND_NAME,
    configure_page,
    render_analytics_scripts,
    render_global_styles,
    render_header,
    render_sidebar,
)
from .pages import render_page

__all__ = [
    "BRAND_NAME",
    "configure_page",
    "enable_fastf1_cache",
    "render_analytics_scripts",
    "render_global_styles",
    "render_header",
    "render_page",
    "render_sidebar",
]

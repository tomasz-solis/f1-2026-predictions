"""Dashboard modules for Streamlit app composition."""

from importlib import import_module
from typing import Any

_LAZY_ATTRS = {
    "BRAND_NAME": ("layout", "BRAND_NAME"),
    "configure_page": ("layout", "configure_page"),
    "enable_fastf1_cache": ("cache", "enable_fastf1_cache"),
    "render_analytics_scripts": ("layout", "render_analytics_scripts"),
    "render_global_styles": ("layout", "render_global_styles"),
    "render_header": ("layout", "render_header"),
    "render_page": ("pages", "render_page"),
    "render_sidebar": ("layout", "render_sidebar"),
}

_LAZY_SUBMODULES = {
    "accuracy",
    "accuracy_view",
    "cache",
    "checkpoint_predictor",
    "layout",
    "live_prediction_flow",
    "pages",
    "prediction_boundary",
    "prediction_cascade",
    "prediction_checkpointing",
    "prediction_flow",
    "prediction_horizon",
    "prediction_messages",
    "prediction_serving",
    "precomputed_predictions",
    "rendering",
    "rendering_html",
    "rendering_qualifying",
    "rendering_race",
    "team_comparison",
    "team_comparison_fallbacks",
    "team_radar",
    "team_snapshot_history",
    "update_flow",
    "warmup",
    "warmup_prediction_builders",
}

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


def __getattr__(name: str) -> Any:
    """Lazily expose dashboard entry points without importing every page module."""
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    if name in _LAZY_SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

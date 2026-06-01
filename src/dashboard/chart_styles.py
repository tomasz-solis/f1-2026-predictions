"""Shared chart styling helpers for dashboard figures."""

from __future__ import annotations

_CHECKPOINT_LINE_COLORS = {
    "PRE": "#9DC6FF",
    "FP1": "#3671C6",
    "FP2": "#F4A7A3",
    "FP3": "#FF4D2D",
    "SQ": "#F5B74A",
    "SPRINT": "#48BF91",
    "Q": "#C084FC",
    "R": "#E8EDF2",
}
_DEFAULT_CHECKPOINT_LINE_COLOR = "#8B949E"


def checkpoint_line_color(checkpoint_session: str | None) -> str:
    """Return the dashboard-wide line color for one checkpoint code."""
    checkpoint = str(checkpoint_session or "").strip().upper()
    return _CHECKPOINT_LINE_COLORS.get(checkpoint, _DEFAULT_CHECKPOINT_LINE_COLOR)

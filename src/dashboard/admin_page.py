"""Operator panel: race inputs and precompute control, behind the admin token.

The penalty and substitution editors used to hang off the public Prediction page, so
running an operator job meant loading a full forecast first. They live here instead. This
page renders no prediction, no team comparison, and no accuracy work — opening it with
``?admin=<TL_ADMIN_TOKEN>`` costs a schedule lookup and an artifact-version read, nothing
more.

The other tabs stay visible on purpose. After saving a substitution and triggering a run,
the Prediction tab is the only place to see whether the new grid actually landed, and an
operator checking that should be looking at exactly what a visitor sees.

The precompute status answers the question a deploy creates: the artifact hash covers the
prediction code, so shipping a model change invalidates every warmed prediction at once.
Until the next ``preheat`` cron run there is nothing to serve, and the hero deck says so
rather than leaving it to be guessed from an empty dashboard.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import streamlit as st

from src.dashboard import render_ops
from src.dashboard.driver_substitution_admin import render_driver_substitution_editor
from src.dashboard.grid_penalty_admin import admin_access_granted, render_grid_penalty_editor

logger = logging.getLogger(__name__)

ADMIN_PAGE_NAME = "Admin"
_SEASON_STATE_KEY = "admin_season"


def render_admin_page(*, st_module: Any = st, env: Mapping[str, str] | None = None) -> None:
    """Render the operator panel for a viewer holding the admin token."""
    if not admin_access_granted(env=env, st_module=st_module):
        st_module.error("This page is operator-only.")
        return

    year = _selected_admin_season(st_module)
    status = _read_precompute_status(year)
    _render_hero(st_module, year=year, status=status, env=env)

    year, race_name = _render_selectors(st_module, year)

    st_module.markdown("#### Race inputs")
    render_grid_penalty_editor(race_name=race_name, year=year, st_module=st_module, env=env)
    render_driver_substitution_editor(race_name=race_name, year=year, st_module=st_module, env=env)

    st_module.markdown("#### Precompute")
    _render_precompute_detail(status, st_module)
    _render_ops_controls(st_module, env=env)
    _render_render_activity(st_module, env=env)


def _selected_admin_season(st_module: Any) -> int:
    """Read the panel's season from session state before the selector redraws it.

    The hero deck is rendered above the controls, the same way every other page does it,
    so it has to read the season Streamlit already holds rather than the widget's return.
    """
    from .pages import DEFAULT_SEASON

    try:
        raw_value = st_module.session_state.get(_SEASON_STATE_KEY, DEFAULT_SEASON)
    except Exception:  # noqa: BLE001 - session state is absent headless and in tests
        return int(DEFAULT_SEASON)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return int(DEFAULT_SEASON)


def _render_hero(
    st_module: Any,
    *,
    year: int,
    status: dict[str, Any],
    env: Mapping[str, str] | None,
) -> None:
    """Render the page intro in the same deck every other dashboard page uses."""
    from .rendering import render_page_hero_deck

    horizon = status.get("horizon")
    ready = _race_list(horizon, "ready_races")
    expected = _race_list(horizon, "expected_targets")
    warmed_value = f"{len(ready)}/{len(expected)}" if horizon else "None"
    artifact_hash = str(status.get("artifact_hash") or "")
    ops_ready = render_ops.render_ops_configured(env)

    render_page_hero_deck(
        title="Operator panel",
        summary=(
            "Set a race's grid penalties and stand-in drivers, then start the run that rebuilds "
            "the forecasts from them."
        ),
        eyebrow="Operator",
        cards=[
            {
                "label": "Season",
                "value": str(year),
                "meta": "The race inputs below apply to this season.",
                "tone": "neutral",
            },
            {
                "label": "Warmed",
                "value": warmed_value,
                "meta": "Races precomputed at the artifact hash now deployed.",
                "tone": "success" if horizon and not _missing_races(ready, expected) else "warning",
            },
            {
                "label": "Artifact",
                "value": artifact_hash[:8] or "Unknown",
                "meta": "Moves whenever the prediction code or its inputs change.",
                "tone": "accent",
            },
            {
                "label": "Render ops",
                "value": "Ready" if ops_ready else "Not set",
                "meta": "Whether this service can trigger runs and restarts.",
                "tone": "success" if ops_ready else "warning",
            },
        ],
        st_module=st_module,
    )


def _render_selectors(st_module: Any, year: int) -> tuple[int, str]:
    """Pick the season and race the editors below apply to."""
    from .pages import (
        _available_seasons,
        _load_local_race_options,
        _load_race_options,
        _load_race_round_meta,
        _order_races_by_round,
    )

    season_options = _available_seasons()
    if year not in season_options:
        season_options = [year, *season_options]
    year = int(
        st_module.selectbox(
            "Season",
            options=season_options,
            index=season_options.index(year),
            key=_SEASON_STATE_KEY,
        )
    )

    race_options = _load_local_race_options(year) or _load_race_options(year)
    race_options, default_index = _order_races_by_round(
        list(race_options),
        _load_race_round_meta(year),
        today_iso=datetime.now(UTC).date().isoformat(),
    )
    if not race_options:
        st_module.warning(f"No races found for {year}.")
        return year, ""

    selection = st_module.selectbox(
        "Grand Prix",
        options=race_options,
        index=default_index,
        key="admin_race",
    )
    return year, str(selection).replace(" (Sprint)", "")


def _read_precompute_status(year: int) -> dict[str, Any]:
    """Look up whether warmed predictions exist for the code and artifacts now deployed."""
    from .cache import get_artifact_versions
    from .precomputed_predictions import compute_artifact_hash, load_precompute_horizon_index

    try:
        artifact_hash = compute_artifact_hash(get_artifact_versions(year))
        horizon = load_precompute_horizon_index(year=year, artifact_hash=artifact_hash)
    except Exception as exc:  # noqa: BLE001 - a status read must not break the panel
        logger.warning("Could not read the precompute horizon for %s: %s", year, exc)
        return {"artifact_hash": "", "horizon": None, "error": str(exc)}

    return {"artifact_hash": artifact_hash, "horizon": horizon, "error": None}


def _render_precompute_detail(status: dict[str, Any], st_module: Any) -> None:
    """Spell out the horizon behind the hero deck's summary cards."""
    if status.get("error"):
        st_module.warning(f"Could not read the precompute status: {status['error']}")
        return

    horizon = status.get("horizon")
    if not horizon:
        st_module.error(
            "Nothing is warmed at this artifact hash. The dashboard has no forecast to serve "
            "until a precompute run finishes. This is the expected state right after a deploy "
            "that touched the prediction code."
        )
        return

    st_module.caption(
        f"Anchor {horizon.get('anchor_race_name', '?')} "
        f"({horizon.get('anchor_session_name', '?')}) · "
        f"model {horizon.get('model_version', '?')} · "
        f"updated {horizon.get('updated_at', '?')}"
    )
    missing = _missing_races(
        _race_list(horizon, "ready_races"), _race_list(horizon, "expected_targets")
    )
    if missing:
        st_module.warning("Not warmed yet: " + ", ".join(missing))


def _race_list(horizon: Any, key: str) -> list[str]:
    """Read one race-name list off a horizon payload, tolerating a malformed entry."""
    if not isinstance(horizon, dict):
        return []
    values = horizon.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(race).strip() for race in values if str(race).strip()]


def _missing_races(ready: list[str], expected: list[str]) -> list[str]:
    """Return the target races that have no warmed prediction yet."""
    return [race for race in expected if race not in ready]


def _render_ops_controls(st_module: Any, *, env: Mapping[str, str] | None) -> None:
    """Offer the two Render actions plus a local cache clear."""
    absent = render_ops.missing_settings(env)
    if absent:
        st_module.info(
            "Set "
            + ", ".join(absent)
            + " on this service to trigger precompute runs and restarts from here."
        )

    if st_module.button(
        "Trigger precompute run",
        disabled=bool(absent),
        help="Starts the preheat cron immediately. Render cancels any run already in flight.",
    ):
        _report(st_module, render_ops.trigger_precompute_run(env))

    if st_module.button(
        "Restart web service",
        disabled=bool(absent),
        help="Restarts this dashboard. Try clearing the caches first.",
    ):
        _report(st_module, render_ops.restart_web_service(env))

    if st_module.button(
        "Clear dashboard caches",
        help="Drops the cached predictor and schedule lookups without a restart.",
    ):
        _clear_caches(st_module)


def _render_render_activity(st_module: Any, *, env: Mapping[str, str] | None) -> None:
    """Show what the cron and the web service have actually been doing.

    Render has no endpoint for reading a cron run directly, so both lists come from each
    service's event feed. A failed run reports why — ``oomKilled`` and ``nonZeroExit``
    both show up here, and neither is visible from the horizon index alone.

    Two API calls on every render of this page, uncached. The page already does an
    artifact-store read and a schedule lookup, so this is not the slow part.
    """
    if render_ops.missing_settings(env):
        return

    st_module.markdown("#### Recent activity")
    if st_module.button("Refresh", help="Re-read the Render event feeds."):
        pass  # Any button press reruns the page, which is the refresh.

    _render_event_list(
        st_module,
        "Precompute runs",
        render_ops.precompute_run_events(env),
    )
    _render_event_list(
        st_module,
        "Web service",
        render_ops.web_service_events(env),
    )


def _render_event_list(
    st_module: Any,
    heading: str,
    outcome: tuple[list[dict[str, str]], str],
) -> None:
    """Render one service's event feed as one line per event."""
    rows, error = outcome
    if error:
        st_module.warning(f"{heading}: {error}")
        return

    st_module.markdown(f"**{heading}**")
    if not rows:
        st_module.caption("Nothing recorded yet.")
        return

    for row in rows:
        parts = [row["timestamp"], row["type"], row["outcome"]]
        st_module.caption(" · ".join(part for part in parts if part))


def _report(st_module: Any, outcome: tuple[bool, str]) -> None:
    """Show the result of a Render API call."""
    succeeded, message = outcome
    if succeeded:
        st_module.success(message)
    else:
        st_module.error(message)


def _clear_caches(st_module: Any) -> None:
    """Drop Streamlit's cached resources and data for this process."""
    try:
        st_module.cache_resource.clear()
        st_module.cache_data.clear()
    except Exception as exc:  # noqa: BLE001 - clearing is best-effort
        logger.warning("Could not clear the dashboard caches: %s", exc)
        st_module.error(f"Could not clear the caches: {exc}")
        return
    st_module.success("Caches cleared.")

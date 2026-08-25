"""Operator-only editor for post-qualifying grid penalties.

The stewards publish penalties on a Saturday night, hours before the race. Editing
``config/default.yaml`` would mean a commit and a redeploy at exactly the wrong time, so
penalties are written to the runtime artifact store from here instead and picked up by
the next precompute.

The dashboard is public, so this is hidden behind a token: nothing renders unless
``TL_ADMIN_TOKEN`` is set on the service *and* matches the ``admin`` query parameter.
Without the token the page has no penalty controls at all.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Mapping
from typing import Any

import streamlit as st

from src.utils.grid_penalties import load_configured_penalties, save_penalties

logger = logging.getLogger(__name__)

_TOKEN_ENV_VAR = "TL_ADMIN_TOKEN"
_QUERY_PARAM = "admin"


def _provided_token(st_module: Any = st) -> str:
    """Read the admin token from the query string, tolerating list-valued params."""
    try:
        value = st_module.query_params.get(_QUERY_PARAM)
    except Exception as exc:  # noqa: BLE001 - a missing query API must not break the page
        logger.warning("Could not read query parameters: %s", exc)
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def admin_access_granted(
    *,
    env: Mapping[str, str] | None = None,
    st_module: Any = st,
) -> bool:
    """Return whether this viewer may edit penalties."""
    expected = str((env if env is not None else os.environ).get(_TOKEN_ENV_VAR, "")).strip()
    if not expected:
        return False
    provided = _provided_token(st_module)
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)


def render_grid_penalty_editor(
    *,
    race_name: str,
    year: int,
    st_module: Any = st,
    env: Mapping[str, str] | None = None,
) -> None:
    """Render the penalty editor when the viewer holds the admin token."""
    if not admin_access_granted(env=env, st_module=st_module):
        return

    stored = load_configured_penalties(race_name=race_name, year=year)
    with st_module.expander(f"Grid penalties — {race_name}", expanded=not stored):
        if stored:
            st_module.caption(
                "Applied to the race grid: "
                + ", ".join(f"{driver} {penalty}" for driver, penalty in sorted(stored.items()))
            )
        else:
            st_module.caption("No penalties recorded for this race.")

        with st_module.form(f"grid_penalty_form_{year}_{race_name}"):
            driver = st_module.text_input(
                "Driver code",
                max_chars=4,
                help="Three-letter code as the timing feed spells it, for example ANT.",
            )
            penalty = st_module.text_input(
                "Penalty",
                help="Places dropped, or 'back' for a back-of-grid start, or 'pit' for a "
                "pit-lane start.",
            )
            submitted = st_module.form_submit_button("Save penalty")

        if submitted:
            _save(
                race_name=race_name,
                year=year,
                stored=stored,
                driver=driver,
                penalty=penalty,
                st_module=st_module,
            )

        if stored and st_module.button("Clear all penalties for this race"):
            _clear(race_name=race_name, year=year, st_module=st_module)


def _save(
    *,
    race_name: str,
    year: int,
    stored: dict[str, Any],
    driver: str,
    penalty: str,
    st_module: Any,
) -> None:
    """Persist one penalty, leaving the others in place."""
    code = str(driver).strip().upper()
    if not code or not str(penalty).strip():
        st_module.warning("Enter both a driver code and a penalty.")
        return

    updated = dict(stored)
    updated[code] = str(penalty).strip()
    try:
        save_penalties(race_name=race_name, year=year, penalties=updated)
    except (ValueError, RuntimeError) as exc:
        st_module.error(f"Could not save the penalty: {exc}")
        return

    st_module.success(f"Saved {code} {str(penalty).strip()} for {race_name}.")
    st_module.rerun()


def _clear(*, race_name: str, year: int, st_module: Any) -> None:
    """Remove every penalty recorded for one race."""
    try:
        save_penalties(race_name=race_name, year=year, penalties={})
    except (ValueError, RuntimeError) as exc:
        st_module.error(f"Could not clear the penalties: {exc}")
        return

    st_module.success(f"Cleared the penalties for {race_name}.")
    st_module.rerun()

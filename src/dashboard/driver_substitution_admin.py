"""Operator-only editor for one race's driver substitutions.

A stand-in is confirmed days before the weekend and usually moves two drivers, not one:
the reserve takes the injured driver's seat and someone else takes the reserve's. Editing
``data/current_lineups.json`` would mean a commit and a redeploy for a change that lasts
one or two races, so substitutions are written to the runtime artifact store from here.

The whole chain is saved in one submit, deliberately. Half a chain seats one driver in two
cars, and saving swap-by-swap would reject every valid chain at its first step.

Access is gated by the same admin token as the grid-penalty editor: the dashboard is
public, and nothing here renders without it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import streamlit as st

from src.dashboard.grid_penalty_admin import admin_access_granted
from src.utils.driver_substitutions import load_configured_substitutions, save_substitutions

logger = logging.getLogger(__name__)

_PLACEHOLDER = "HAD > LAW\nLAW > TSU"


def render_driver_substitution_editor(
    *,
    race_name: str,
    year: int,
    st_module: Any = st,
    env: Mapping[str, str] | None = None,
) -> None:
    """Render the substitution editor when the viewer holds the admin token."""
    if not admin_access_granted(env=env, st_module=st_module):
        return

    stored = load_configured_substitutions(race_name=race_name, year=year)
    with st_module.expander(f"Driver substitutions — {race_name}", expanded=bool(stored)):
        if stored:
            st_module.caption(
                "Racing this weekend: "
                + ", ".join(
                    f"{driver_in} in for {driver_out}"
                    for driver_out, driver_in in sorted(stored.items())
                )
            )
        else:
            st_module.caption("No substitutions recorded for this race.")

        with st_module.form(f"driver_substitution_form_{year}_{race_name}"):
            text = st_module.text_area(
                "Substitutions",
                value=_format(stored),
                placeholder=_PLACEHOLDER,
                help=(
                    "One swap per line, driver out first: 'HAD > LAW'. Enter the whole chain "
                    "at once — moving a driver up without freeing his own seat is rejected."
                ),
            )
            submitted = st_module.form_submit_button("Save substitutions")

        if submitted:
            _save(race_name=race_name, year=year, text=text, st_module=st_module)


def _format(substitutions: Mapping[str, str]) -> str:
    """Render the stored map back into the one-swap-per-line form."""
    return "\n".join(
        f"{driver_out} > {driver_in}" for driver_out, driver_in in sorted(substitutions.items())
    )


def _parse(text: str) -> dict[str, str]:
    """Read 'OUT > IN' lines into a substitution map, tolerating a bare separator."""
    substitutions: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.replace("->", ">").split(">")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Could not read this line as 'OUT > IN': {raw_line.strip()!r}")
        driver_out, driver_in = parts
        substitutions[driver_out.upper()] = driver_in.upper()
    return substitutions


def _save(*, race_name: str, year: int, text: str, st_module: Any) -> None:
    """Replace this race's substitutions with what the operator typed."""
    try:
        substitutions = _parse(text)
        save_substitutions(race_name=race_name, year=year, substitutions=substitutions)
    except (ValueError, RuntimeError) as exc:
        st_module.error(f"Could not save the substitutions: {exc}")
        return

    if substitutions:
        st_module.success(f"Saved {len(substitutions)} substitution(s) for {race_name}.")
    else:
        st_module.success(f"Cleared the substitutions for {race_name}.")
    st_module.rerun()

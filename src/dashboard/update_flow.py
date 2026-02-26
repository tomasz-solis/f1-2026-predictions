"""Dashboard update flows for race learning and practice capture."""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

_PRACTICE_UPDATE_STATE_FILE = Path("data/systems/practice_characteristics_state.json")


def auto_update_if_needed(force_recheck: bool = False, year: int = 2026) -> None:
    """
    Check for and apply updates from completed races.
    Also refreshes predictor if characteristic files were manually updated.

    Args:
        force_recheck: If True, clears learned races cache to force re-check
        year: Season year to evaluate for newly completed races
    """
    from src.utils.auto_updater import auto_update_from_races, needs_update

    if force_recheck:
        try:
            needs_update_flag, new_races = needs_update(year=year, force_recheck=True)
        except TypeError:
            try:
                # Backward-compatible fallback for patched or older callables without year kwargs.
                needs_update_flag, new_races = needs_update(force_recheck=True)
            except TypeError:
                needs_update_flag, new_races = needs_update()
    else:
        try:
            needs_update_flag, new_races = needs_update(year=year)
        except TypeError:
            needs_update_flag, new_races = needs_update()

    if needs_update_flag:
        st.info(f"Found {len(new_races)} new race(s) to learn from. Updating characteristics...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(message)

        try:
            updated_count = auto_update_from_races(
                progress_callback=progress_callback,
                races_to_update=new_races,
                year=year,
            )
        except TypeError:
            try:
                # Backward-compatible fallback for patched callables without year kwargs.
                updated_count = auto_update_from_races(
                    progress_callback=progress_callback,
                    races_to_update=new_races,
                )
            except TypeError:
                updated_count = auto_update_from_races(progress_callback)

        progress_bar.empty()
        status_text.empty()

        if updated_count == len(new_races):
            st.success(f"Learned from {updated_count} race(s). Predictions now use updated data.")
            st.cache_resource.clear()
            st.cache_data.clear()
        else:
            raise RuntimeError(
                f"Race refresh incomplete: updated {updated_count} of {len(new_races)} new races."
            )


def _load_practice_update_state() -> dict:
    """Load persisted state for practice characteristic updates."""
    if not _PRACTICE_UPDATE_STATE_FILE.exists():
        return {"races": {}}

    try:
        with open(_PRACTICE_UPDATE_STATE_FILE) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"races": {}}

    if not isinstance(state, dict):
        return {"races": {}}

    races = state.get("races")
    if not isinstance(races, dict):
        return {"races": {}}

    return {"races": races}


def _save_practice_update_state(state: dict) -> None:
    """Persist state for practice characteristic updates."""
    _PRACTICE_UPDATE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _PRACTICE_UPDATE_STATE_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(_PRACTICE_UPDATE_STATE_FILE)


def auto_update_practice_characteristics_if_needed(
    year: int,
    race_name: str,
    is_sprint: bool,
    force_recheck: bool = False,
) -> dict:
    """
    Update car characteristics from completed free-practice sessions (FP1/FP2/FP3).

    This is conservative and only runs when new FP sessions are completed for a race.

    Args:
        year: Season year
        race_name: Name of the race
        is_sprint: Whether this is a sprint weekend
        force_recheck: If True, ignores cached state and re-checks session completion
    """
    from src.systems.testing_updater import update_from_testing_sessions
    from src.utils import config_loader
    from src.utils.session_detector import SessionDetector

    detector = SessionDetector()
    completed = detector.get_completed_sessions(year, race_name, is_sprint)
    completed_fp_sessions = [session for session in completed if session.startswith("FP")]

    if not completed_fp_sessions:
        return {"updated": False, "completed_fp_sessions": []}

    session_order = {"FP1": 1, "FP2": 2, "FP3": 3}
    completed_fp_sessions = sorted(
        set(completed_fp_sessions), key=lambda s: session_order.get(s, 99)
    )

    race_key = f"{year}::{race_name}"
    state = _load_practice_update_state()
    processed_sessions = set(state["races"].get(race_key, {}).get("sessions", []))
    latest_processed = set(completed_fp_sessions).issubset(processed_sessions)

    # Skip if already processed (unless force_recheck enabled)
    if latest_processed and not force_recheck:
        return {"updated": False, "completed_fp_sessions": completed_fp_sessions}

    practice_new_weight = config_loader.get("baseline_predictor.practice_capture.new_weight", 0.35)
    practice_directionality_scale = config_loader.get(
        "baseline_predictor.practice_capture.directionality_scale", 0.08
    )
    practice_session_aggregation = config_loader.get(
        "baseline_predictor.practice_capture.session_aggregation", "laps_weighted"
    )
    practice_run_profile = config_loader.get(
        "baseline_predictor.practice_capture.run_profile", "balanced"
    )

    summary = update_from_testing_sessions(
        year=year,
        characteristics_year=year,
        events=[race_name],
        sessions=completed_fp_sessions,
        testing_backend="auto",
        cache_dir="data/raw/.fastf1_cache_testing",
        force_renew_cache=False,
        # Lower weight than pre-season testing to avoid abrupt directionality swings.
        new_weight=practice_new_weight,
        directionality_scale=practice_directionality_scale,
        session_aggregation=practice_session_aggregation,
        run_profile=practice_run_profile,
        dry_run=False,
    )
    updated_teams = summary.get("updated_teams", []) if isinstance(summary, dict) else []

    state["races"][race_key] = {
        "sessions": completed_fp_sessions,
        "updated_at": datetime.now().isoformat(),
        "teams_updated": len(updated_teams),
    }
    _save_practice_update_state(state)

    return {
        "updated": True,
        "completed_fp_sessions": completed_fp_sessions,
        "teams_updated": len(updated_teams),
    }

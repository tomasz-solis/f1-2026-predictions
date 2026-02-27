"""Nightly live-network FastF1 integration checks."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import fastf1
import pytest

from src.dashboard import update_flow
from src.utils.session_detector import SessionDetector


def _require_live_fastf1() -> None:
    if os.getenv("FASTF1_LIVE_TESTS", "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip("Set FASTF1_LIVE_TESTS=1 to run live FastF1 integration checks")


def _event_candidates(year: int) -> tuple[str | None, str | None]:
    schedule = fastf1.get_event_schedule(year)
    race_events = schedule[
        (schedule["EventFormat"].notna())
        & (~schedule["EventName"].str.contains("Testing", case=False, na=False))
    ]

    conventional_name: str | None = None
    sprint_name: str | None = None
    for _, event in race_events.iterrows():
        event_name = str(event.get("EventName", "")).strip()
        event_format = str(event.get("EventFormat", "")).strip().lower()
        if not event_name:
            continue
        if "sprint" in event_format and sprint_name is None:
            sprint_name = event_name
        if "sprint" not in event_format and conventional_name is None:
            conventional_name = event_name
        if conventional_name and sprint_name:
            break

    return conventional_name, sprint_name


@pytest.mark.integration
@pytest.mark.live_fastf1
def test_live_fastf1_conventional_session_completion_state():
    _require_live_fastf1()
    year = datetime.now(UTC).year
    conventional_event, _ = _event_candidates(year)
    if not conventional_event:
        pytest.skip(f"No conventional race event found in FastF1 schedule for {year}")

    detector = SessionDetector()
    state = detector.get_session_completion_state(year, conventional_event, "Q")
    assert state in {"completed", "incomplete", "unknown"}


@pytest.mark.integration
@pytest.mark.live_fastf1
def test_live_fastf1_sprint_session_completion_state():
    _require_live_fastf1()
    year = datetime.now(UTC).year
    _, sprint_event = _event_candidates(year)
    if not sprint_event:
        pytest.skip(f"No sprint race event found in FastF1 schedule for {year}")

    detector = SessionDetector()
    state = detector.get_session_completion_state(year, sprint_event, "SQ")
    assert state in {"completed", "incomplete", "unknown"}


@pytest.mark.integration
@pytest.mark.live_fastf1
@pytest.mark.parametrize(
    ("is_sprint", "anchor_session"),
    [
        (False, "FP1"),
        (True, "SQ"),
    ],
)
def test_live_fastf1_boundary_transition_signal(
    patcher, tmp_path, is_sprint: bool, anchor_session: str
):
    _require_live_fastf1()
    year = datetime.now(UTC).year
    conventional_event, sprint_event = _event_candidates(year)
    race_name = sprint_event if is_sprint else conventional_event
    if not race_name:
        pytest.skip(f"No {'sprint' if is_sprint else 'conventional'} race found in {year} schedule")

    # Keep live integration test isolated from repository state files.
    patcher.setattr(
        update_flow, "_EVENT_BOUNDARY_STATE_FILE", tmp_path / "event_boundary_state.json"
    )
    patcher.setattr(update_flow, "should_read_db_first", lambda: False)
    patcher.setattr(update_flow, "should_write_to_db", lambda: False)
    patcher.setattr(update_flow, "should_write_to_file", lambda: True)

    event = fastf1.get_event(year, race_name)
    raw_anchor = event.get_session_date(anchor_session)
    anchor = update_flow._coerce_utc_datetime(raw_anchor)
    if anchor is None:
        pytest.skip(f"Session date missing for {race_name} {anchor_session}")

    before = anchor - timedelta(minutes=30)
    after = anchor + timedelta(hours=4)

    first = update_flow.detect_event_boundary_refresh_if_needed(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        now_utc=before,
    )
    second = update_flow.detect_event_boundary_refresh_if_needed(
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
        now_utc=after,
    )

    assert first["boundary_signature"] != ""
    assert second["boundary_signature"] != ""
    assert second["reason"] in {
        "no_change",
        "first_seen_after_boundary",
        "session_data_changed",
        "session_boundary_delta",
        "schedule_changed",
        "weekend_type_changed",
    }

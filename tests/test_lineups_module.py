from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.utils import lineups


def test_get_lineups_from_session_extracts_participants(patcher):
    results = pd.DataFrame(
        {
            "TeamName": ["McLaren", "McLaren", "Ferrari"],
            "Abbreviation": ["NOR", "PIA", "LEC"],
        }
    )
    session = SimpleNamespace(results=results)
    session.load = lambda laps, telemetry, weather: None

    patcher.setattr(lineups.ff1, "get_session", lambda year, race, session_type: session)

    extracted = lineups.get_lineups_from_session(2026, "Bahrain Grand Prix", "Q")

    assert extracted == {"McLaren": ["NOR", "PIA"], "Ferrari": ["LEC"]}


def test_get_lineups_from_session_returns_none_on_exception(patcher):
    patcher.setattr(
        lineups.ff1,
        "get_session",
        lambda year, race, session_type: (_ for _ in ()).throw(TypeError("boom")),
    )

    assert lineups.get_lineups_from_session(2026, "Bahrain Grand Prix", "Q") is None


def test_load_current_lineups_handles_missing_and_existing_files(tmp_path):
    missing_path = tmp_path / "missing.json"
    assert lineups.load_current_lineups(str(missing_path)) is None

    config_path = tmp_path / "current_lineups.json"
    payload = {"current_lineups": {"McLaren": ["NOR", "PIA"]}}
    config_path.write_text(json.dumps(payload))

    loaded = lineups.load_current_lineups(str(config_path))
    assert loaded == {"McLaren": ["NOR", "PIA"]}


def test_get_lineups_historical_prefers_session_data(patcher):
    patcher.setattr(
        lineups,
        "get_lineups_from_session",
        lambda year, race, session_type: {"Ferrari": ["LEC", "HAM"]},
    )

    result = lineups.get_lineups(2025, "Monaco Grand Prix")
    assert result == {"Ferrari": ["LEC", "HAM"]}


def test_get_lineups_falls_back_to_config(patcher):
    patcher.setattr(lineups, "get_lineups_from_session", lambda year, race, session_type: None)
    patcher.setattr(
        lineups, "load_current_lineups", lambda config_path: {"McLaren": ["NOR", "PIA"]}
    )

    result = lineups.get_lineups(2025, "Monaco Grand Prix")
    assert result == {"McLaren": ["NOR", "PIA"]}


def test_get_lineups_raises_without_any_data(patcher):
    patcher.setattr(lineups, "get_lineups_from_session", lambda year, race, session_type: None)
    patcher.setattr(lineups, "load_current_lineups", lambda config_path: None)

    with pytest.raises(ValueError, match="No lineup data available"):
        lineups.get_lineups(2026, "Bahrain Grand Prix")


def test_save_current_lineups_writes_file(tmp_path):
    output = tmp_path / "nested" / "current_lineups.json"
    lineups.save_current_lineups({"McLaren": ["NOR", "PIA"]}, str(output))

    saved = json.loads(output.read_text())
    assert "last_updated" in saved
    assert saved["current_lineups"]["McLaren"] == ["NOR", "PIA"]


def test_extract_lineups_for_season_skips_testing_and_writes_output(tmp_path, patcher):
    schedule = pd.DataFrame(
        {
            "EventName": [
                "Pre-Season Testing",
                "Bahrain Grand Prix",
                "Saudi Arabian Grand Prix",
            ]
        }
    )

    patcher.setattr(lineups.ff1, "get_event_schedule", lambda year: schedule)

    def _mock_get_lineups(year, race_name, session_type="Q"):
        if race_name == "Bahrain Grand Prix":
            return {"McLaren": ["NOR", "PIA"]}
        return None

    patcher.setattr(lineups, "get_lineups_from_session", _mock_get_lineups)

    output_path = tmp_path / "lineups_2026.json"
    result = lineups.extract_lineups_for_season(2026, output_path=str(output_path))

    assert list(result.keys()) == ["Bahrain Grand Prix"]
    payload = json.loads(output_path.read_text())
    assert payload["season"] == 2026
    assert "Bahrain Grand Prix" in payload["races"]

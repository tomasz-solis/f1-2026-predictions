"""Tests for shared race data extraction helpers."""

import pandas as pd

from src.extractors import race_data


def _make_session(results: pd.DataFrame):
    class _Session:
        def __init__(self, _results: pd.DataFrame):
            self.results = _results

        def load(self, laps=False):
            return None

    return _Session(results)


def test_extract_race_data_success(patcher):
    quali_results = pd.DataFrame(
        [
            {"Abbreviation": "VER", "Position": 2},
            {"Abbreviation": "NOR", "Position": 1},
        ]
    )
    race_results = pd.DataFrame(
        [
            {"Abbreviation": "VER", "Position": 1, "TeamName": "Red Bull Racing", "dnf": False},
            {"Abbreviation": "NOR", "Position": 2, "TeamName": "McLaren", "dnf": True},
            {"Abbreviation": "LEC", "Position": None, "TeamName": "Ferrari", "dnf": False},
        ]
    )

    def _get_session(year: int, race_name: str, session_type: str):
        assert (year, race_name) == (2026, "Bahrain Grand Prix")
        if session_type == "Q":
            return _make_session(quali_results)
        return _make_session(race_results)

    patcher.setattr(race_data.ff1, "get_session", _get_session)

    result = race_data.extract_race_data(2026, "Bahrain Grand Prix")

    assert set(result.keys()) == {"VER", "NOR"}
    assert result["VER"]["quali"] == 2
    assert result["VER"]["race"] == 1
    assert result["VER"]["gain"] == 1
    assert result["VER"]["dnf"] is False
    assert result["NOR"]["dnf"] is True
    assert result["NOR"]["dn"] is True


def test_extract_race_data_uses_status_fallback_for_dnf(patcher):
    quali_results = pd.DataFrame([{"Abbreviation": "HAM", "Position": 5}])
    race_results = pd.DataFrame(
        [
            {
                "Abbreviation": "HAM",
                "Position": 8,
                "TeamName": "Mercedes",
                "Status": "Retired",
            }
        ]
    )

    patcher.setattr(
        race_data.ff1,
        "get_session",
        lambda _year, _race_name, session_type: _make_session(
            quali_results if session_type == "Q" else race_results
        ),
    )

    result = race_data.extract_race_data(2026, "Bahrain Grand Prix")

    assert result["HAM"]["dnf"] is True


def test_extract_race_data_returns_none_on_exception(patcher):
    patcher.setattr(
        race_data.ff1,
        "get_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )

    assert race_data.extract_race_data(2026, "Bahrain Grand Prix") is None


def test_extract_season_filters_testing_events(patcher):
    schedule = pd.DataFrame(
        [
            {"EventName": "Pre-Season Test", "EventFormat": "testing"},
            {"EventName": "Bahrain Grand Prix", "EventFormat": "conventional"},
            {"EventName": "Saudi Arabian Grand Prix", "EventFormat": float("nan")},
        ]
    )

    patcher.setattr(race_data.ff1, "get_event_schedule", lambda _year: schedule)
    patcher.setattr(
        race_data,
        "extract_race_data",
        lambda _year, race_name: (
            {"VER": {"race": 1}, "NOR": {"race": 2}} if race_name == "Bahrain Grand Prix" else {}
        ),
    )

    extracted = race_data.extract_season(2026, verbose=False)

    assert set(extracted.keys()) == {"VER", "NOR"}
    assert len(extracted["VER"]) == 1


def test_dnf_helper_functions_support_both_keys():
    races = [
        {"race": 1, "dnf": True},
        {"race": 2, "dn": True},
        {"race": 3, "dnf": False},
    ]

    assert race_data.count_total_dnfs(races) == 2
    assert len(race_data.get_dnf_races(races)) == 2
    assert len(race_data.get_clean_races(races)) == 1
    assert len(race_data.get_valid_races([{"has_quali": True}, {"has_quali": False}])) == 1

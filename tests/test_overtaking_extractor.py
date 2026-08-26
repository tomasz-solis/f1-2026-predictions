"""Tests for overtaking-likelihood extraction."""

import json

import pandas as pd

from src.extractors import overtaking


def _make_race_session(laps: pd.DataFrame | None):
    class _Session:
        def __init__(self, _laps):
            self.laps = _laps

        def load(self, laps=True, telemetry=False, weather=False):
            return None

    return _Session(laps)


def test_extract_overtakes_from_race_counts_position_changes(patcher):
    laps = pd.DataFrame(
        [
            # Lap 1
            {"LapNumber": 1, "Driver": "A", "Position": 1, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "B", "Position": 2, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "C", "Position": 3, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "D", "Position": 4, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "E", "Position": 5, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "F", "Position": 6, "PitOutTime": pd.NaT},
            # Lap 2: B and C swap (2 position changes counted)
            {"LapNumber": 2, "Driver": "A", "Position": 1, "PitOutTime": pd.NaT},
            {"LapNumber": 2, "Driver": "B", "Position": 3, "PitOutTime": pd.NaT},
            {"LapNumber": 2, "Driver": "C", "Position": 2, "PitOutTime": pd.NaT},
            {"LapNumber": 2, "Driver": "D", "Position": 4, "PitOutTime": pd.NaT},
            {"LapNumber": 2, "Driver": "E", "Position": 5, "PitOutTime": pd.NaT},
            {"LapNumber": 2, "Driver": "F", "Position": 6, "PitOutTime": pd.NaT},
            # Lap 3: unchanged
            {"LapNumber": 3, "Driver": "A", "Position": 1, "PitOutTime": pd.NaT},
            {"LapNumber": 3, "Driver": "B", "Position": 3, "PitOutTime": pd.NaT},
            {"LapNumber": 3, "Driver": "C", "Position": 2, "PitOutTime": pd.NaT},
            {"LapNumber": 3, "Driver": "D", "Position": 4, "PitOutTime": pd.NaT},
            {"LapNumber": 3, "Driver": "E", "Position": 5, "PitOutTime": pd.NaT},
            {"LapNumber": 3, "Driver": "F", "Position": 6, "PitOutTime": pd.NaT},
        ]
    )

    patcher.setattr(
        overtaking.ff1, "get_session", lambda *_args, **_kwargs: _make_race_session(laps)
    )

    stats = overtaking.extract_overtakes_from_race(2026, "Bahrain Grand Prix")

    assert stats["total_position_changes"] == 2
    assert stats["laps_analyzed"] == 2
    assert stats["avg_changes_per_lap"] == 1.0
    assert stats["max_changes_in_lap"] == 2


def test_extract_overtakes_from_race_drops_pit_out_laps_and_sub_five_driver_laps(patcher):
    """Pit-out laps are excluded from the swap count, and laps with fewer than 5
    drivers present in both the previous and current lap (safety car / incident)
    are skipped entirely rather than counted."""
    laps = pd.DataFrame(
        [
            # Lap 1
            {"LapNumber": 1, "Driver": "A", "Position": 1, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "B", "Position": 2, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "C", "Position": 3, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "D", "Position": 4, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "E", "Position": 5, "PitOutTime": pd.NaT},
            {"LapNumber": 1, "Driver": "F", "Position": 6, "PitOutTime": pd.NaT},
            # Lap 2: B pits and rejoins in last place (Position 6); its PitOutTime is
            # set, so it must be dropped from the swap count entirely, not counted as
            # a 5-place position change.
            {"LapNumber": 2, "Driver": "A", "Position": 1, "PitOutTime": pd.NaT},
            {
                "LapNumber": 2,
                "Driver": "B",
                "Position": 6,
                "PitOutTime": pd.Timestamp("2026-01-01"),
            },
            {"LapNumber": 2, "Driver": "C", "Position": 2, "PitOutTime": pd.NaT},
            {"LapNumber": 2, "Driver": "D", "Position": 3, "PitOutTime": pd.NaT},
            {"LapNumber": 2, "Driver": "E", "Position": 4, "PitOutTime": pd.NaT},
            {"LapNumber": 2, "Driver": "F", "Position": 5, "PitOutTime": pd.NaT},
            # Lap 3: only 4 drivers present in both laps (safety car bunching drops
            # one car out of scope) -- skipped even though positions differ.
            {"LapNumber": 3, "Driver": "A", "Position": 2, "PitOutTime": pd.NaT},
            {"LapNumber": 3, "Driver": "C", "Position": 1, "PitOutTime": pd.NaT},
            {"LapNumber": 3, "Driver": "D", "Position": 3, "PitOutTime": pd.NaT},
            {"LapNumber": 3, "Driver": "E", "Position": 4, "PitOutTime": pd.NaT},
        ]
    )

    patcher.setattr(
        overtaking.ff1, "get_session", lambda *_args, **_kwargs: _make_race_session(laps)
    )

    stats = overtaking.extract_overtakes_from_race(2026, "Bahrain Grand Prix")

    # Lap 2 only counts C, D, E, F, A (B dropped for pit-out): C 3->2, D 4->3, E 5->4,
    # F 6->5 all change = 4 changes. Lap 3 is skipped (4 drivers < 5 threshold).
    assert stats["laps_analyzed"] == 1
    assert stats["total_position_changes"] == 4
    assert stats["avg_changes_per_lap"] == 4.0


def test_extract_overtakes_from_race_handles_missing_laps(patcher):
    session = _make_race_session(laps=None)
    patcher.setattr(overtaking.ff1, "get_session", lambda *_args, **_kwargs: session)
    assert overtaking.extract_overtakes_from_race(2026, "Bahrain Grand Prix") is None


def test_classify_overtaking_difficulty_thresholds():
    assert overtaking.classify_overtaking_difficulty(2.0) == ("very_hard", 0.2)
    assert overtaking.classify_overtaking_difficulty(3.0) == ("hard", 0.4)
    assert overtaking.classify_overtaking_difficulty(5.0) == ("moderate", 0.6)
    assert overtaking.classify_overtaking_difficulty(6.0) == ("easy", 0.8)
    assert overtaking.classify_overtaking_difficulty(7.5) == ("very_easy", 1.0)


def test_calculate_overtaking_likelihood_aggregates_years(patcher):
    schedule = pd.DataFrame(
        [
            {"EventName": "Bahrain Grand Prix"},
            {"EventName": "Pre-Season Testing"},
        ]
    )
    patcher.setattr(overtaking.ff1, "get_event_schedule", lambda _year: schedule)
    patcher.setattr(
        overtaking,
        "extract_overtakes_from_race",
        lambda year, race_name: {
            "year": year,
            "race": race_name,
            "avg_changes_per_lap": 4.0 if year == 2024 else 6.0,
            "laps_analyzed": 50,
        },
    )

    result = overtaking.calculate_overtaking_likelihood(years=[2024, 2025])

    assert "Bahrain Grand Prix" in result
    track_data = result["Bahrain Grand Prix"]
    assert track_data["years_analyzed"] == 2
    assert track_data["total_laps_analyzed"] == 100
    assert track_data["avg_changes_per_lap"] == 5.0


def test_add_overtaking_to_tracks_updates_and_applies_defaults(tmp_path):
    track_file = tmp_path / "tracks.json"
    output_file = tmp_path / "tracks_updated.json"

    track_file.write_text(
        json.dumps(
            {
                "tracks": {
                    "Bahrain Grand Prix": {"is_street_circuit": 0},
                    "Monaco Grand Prix": {"is_street_circuit": 1},
                    "Silverstone": {"is_street_circuit": 0},
                }
            }
        )
    )

    overtaking_data = {
        "Bahrain Grand Prix": {
            "avg_changes_per_lap": 6.2,
            "years_analyzed": 2,
        }
    }

    overtaking.add_overtaking_to_tracks(track_file, overtaking_data, output_file)

    updated = json.loads(output_file.read_text())
    bahrain = updated["tracks"]["Bahrain Grand Prix"]
    monaco = updated["tracks"]["Monaco Grand Prix"]
    silverstone = updated["tracks"]["Silverstone"]

    assert bahrain["overtaking_difficulty"] == "easy"
    assert bahrain["overtaking_likelihood"] == 0.8
    assert bahrain["overtaking_years_analyzed"] == 2

    assert monaco["overtaking_difficulty"] == "hard"
    assert monaco["overtaking_likelihood"] == 0.4
    assert monaco["overtaking_years_analyzed"] == 0

    assert silverstone["overtaking_difficulty"] == "moderate"
    assert silverstone["overtaking_likelihood"] == 0.6

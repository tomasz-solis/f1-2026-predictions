"""Unit tests for testing/practice directionality updater helpers."""

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from fastf1.exceptions import DataNotLoadedError

from src.systems.testing_updater import (
    _aggregate_metric_samples,
    _blend_directionality,
    _build_directionality_from_metrics,
    _canonicalize_team_name,
    _classify_run_laps,
    _coerce_utc_datetime,
    _collect_session_metrics,
    _count_team_selected_laps,
    _count_team_valid_laps,
    _estimate_tire_deg_slope,
    _extract_team_payload,
    _extract_testing_day,
    _extract_testing_number,
    _is_testing_event,
    _normalize_testing_event_sessions,
    _normalize_tire_deg_scores,
    _resolve_testing_backends,
    _resolve_testing_cache_dir,
    _select_program_aware_laps,
    _testing_session_has_started,
)
from src.predictors.baseline.qualifying_preparation import _blend_strengths
from src.systems.testing_updater_metrics import (
    _drop_implausible_laps,
    _filter_valid_laps,
    _select_short_run_laps,
    _session_rain_fraction,
)
from src.utils.car_snapshot_history import detect_snapshot_anomalies
from src.utils.fp_blending import blend_team_strength


def test_canonicalize_team_name_aliases():
    known_teams = {"Red Bull Racing", "RB", "Audi", "Cadillac F1"}

    assert _canonicalize_team_name("Oracle Red Bull Racing", known_teams) == "Red Bull Racing"
    assert _canonicalize_team_name("Visa Cash App RB", known_teams) == "RB"
    assert _canonicalize_team_name("Kick Sauber", known_teams) == "Audi"
    assert _canonicalize_team_name("Cadillac", known_teams) == "Cadillac F1"


def test_build_directionality_from_metrics_is_centered():
    metrics = {
        "top_speed": 0.8,
        "slow_corner_performance": 0.2,
        "medium_corner_performance": 0.5,
        "fast_corner_performance": 0.6,
    }
    directionality = _build_directionality_from_metrics(metrics, directionality_scale=0.10)

    assert directionality["max_speed"] == 0.03
    assert directionality["slow_corner_speed"] == -0.03
    assert directionality["medium_corner_speed"] == 0.0
    assert directionality["high_corner_speed"] == 0.01


def test_build_directionality_from_metrics_uses_overall_pace_fallback():
    metrics = {"overall_pace": 0.8}
    directionality = _build_directionality_from_metrics(metrics, directionality_scale=0.10)

    assert directionality["max_speed"] == 0.0
    assert directionality["slow_corner_speed"] == 0.03
    assert directionality["medium_corner_speed"] == 0.03
    assert directionality["high_corner_speed"] == 0.03


def test_blend_directionality_respects_weight():
    old = {
        "max_speed": 0.02,
        "slow_corner_speed": 0.01,
        "medium_corner_speed": -0.01,
        "high_corner_speed": 0.00,
    }
    new = {
        "max_speed": 0.06,
        "slow_corner_speed": -0.03,
        "medium_corner_speed": 0.03,
        "high_corner_speed": 0.04,
    }

    blended = _blend_directionality(old, new, new_weight=0.75)

    assert blended["max_speed"] == 0.05
    assert blended["slow_corner_speed"] == -0.02
    assert blended["medium_corner_speed"] == 0.02
    assert blended["high_corner_speed"] == 0.03


def test_aggregate_metric_samples_supports_modes():
    samples = [(0.2, 1.0), (0.8, 9.0)]

    assert _aggregate_metric_samples(samples, "mean") == 0.5
    assert _aggregate_metric_samples(samples, "median") == 0.5
    assert round(_aggregate_metric_samples(samples, "laps_weighted"), 2) == 0.74


def test_normalize_tire_deg_scores_inverts_slope():
    slopes = {"Team A": 0.05, "Team B": 0.15, "Team C": 0.10}
    normalized = _normalize_tire_deg_scores(slopes)

    assert (
        normalized["Team A"]["tire_deg_performance"] > normalized["Team C"]["tire_deg_performance"]
    )
    assert (
        normalized["Team C"]["tire_deg_performance"] > normalized["Team B"]["tire_deg_performance"]
    )
    assert normalized["Team A"]["tire_deg_performance"] == 1.0
    assert normalized["Team B"]["tire_deg_performance"] == 0.0


def test_estimate_tire_deg_slope_ignores_obvious_cooldown_outliers():
    laps = pd.DataFrame(
        {
            "Driver": ["DRV"] * 6,
            "Stint": [1] * 6,
            "Compound": ["SOFT"] * 6,
            "LapNumber": [1, 2, 3, 4, 5, 6],
            "LapTime": [
                pd.to_timedelta("0:01:20.000"),
                pd.to_timedelta("0:01:20.500"),
                pd.to_timedelta("0:01:21.000"),
                pd.to_timedelta("0:01:50.000"),
                pd.to_timedelta("0:01:51.000"),
                pd.to_timedelta("0:01:52.000"),
            ],
        }
    )

    slope = _estimate_tire_deg_slope(laps)

    assert slope == pytest.approx(0.5)


def test_count_team_valid_laps_uses_canonical_mapping():
    class DummySession:
        def __init__(self, laps):
            self.laps = laps

    laps = pd.DataFrame(
        {
            "Team": [
                "Oracle Red Bull Racing",
                "Oracle Red Bull Racing",
                "Kick Sauber",
                "Kick Sauber",
            ],
            "LapTime": [
                pd.to_timedelta("0:01:30"),
                pd.to_timedelta("0:01:31"),
                pd.to_timedelta("0:01:32"),
                pd.NaT,
            ],
        }
    )
    session = DummySession(laps=laps)

    counts = _count_team_valid_laps(
        session,
        known_teams={"Red Bull Racing", "Audi"},
    )

    assert counts["Red Bull Racing"] == 2.0
    assert counts["Audi"] == 1.0


def test_count_team_selected_laps_respects_run_profile():
    class DummySession:
        def __init__(self, laps):
            self.laps = laps

    laps = pd.DataFrame(
        {
            "Team": ["McLaren"] * 9,
            "Driver": ["NOR"] * 9,
            "Stint": [1, 1, 1, 2, 2, 2, 2, 2, 2],
            "LapTime": [pd.to_timedelta(f"0:01:{30 + i:02d}") for i in range(9)],
            "PitOutTime": [pd.NaT] * 9,
            "PitInTime": [pd.NaT] * 9,
        }
    )
    session = DummySession(laps=laps)

    short_counts = _count_team_selected_laps(session, {"McLaren"}, run_profile="short_run")
    long_counts = _count_team_selected_laps(session, {"McLaren"}, run_profile="long_run")

    assert short_counts["McLaren"] > 0
    assert long_counts["McLaren"] > 0
    assert short_counts["McLaren"] != long_counts["McLaren"]


def test_collect_session_metrics_skips_wet_sessions():
    class DummySession:
        def __init__(self, laps):
            self.laps = laps
            self.weather_data = pd.DataFrame({"Rainfall": [1, 1, 0, 1]})

    laps = pd.DataFrame(
        {
            "Team": ["McLaren"] * 6,
            "Driver": ["NOR"] * 6,
            "LapTime": [pd.to_timedelta(f"0:01:{30 + i:02d}") for i in range(6)],
        }
    )
    diagnostics: list[str] = []

    perf, tire = _collect_session_metrics(
        session=DummySession(laps),
        session_key="FP2",
        known_teams={"McLaren"},
        diagnostics=diagnostics,
    )

    assert perf == {}
    assert tire == {}
    assert diagnostics
    assert "wet session" in diagnostics[0]


def test_collect_session_metrics_requires_minimum_valid_team_laps():
    class DummySession:
        def __init__(self, laps):
            self.laps = laps
            self.weather_data = pd.DataFrame({"Rainfall": [0, 0, 0]})

    laps = pd.DataFrame(
        {
            "Team": ["McLaren"] * 4,
            "Driver": ["NOR"] * 4,
            "LapTime": [pd.to_timedelta(f"0:01:{30 + i:02d}") for i in range(4)],
        }
    )

    perf, tire = _collect_session_metrics(
        session=DummySession(laps),
        session_key="FP1",
        known_teams={"McLaren"},
    )

    assert perf == {}
    assert tire == {}


def test_session_rain_fraction_handles_unloaded_weather_data():
    """Missing lazy weather data should behave like unknown conditions, not crash replay."""

    class DummySession:
        @property
        def weather_data(self):
            raise DataNotLoadedError("weather data not loaded")

    assert _session_rain_fraction(DummySession()) is None


def test_classify_run_laps_by_stint_length():
    laps = pd.DataFrame(
        {
            "Driver": ["DRV"] * 12,
            "Stint": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            "LapTime": [pd.to_timedelta(f"0:01:{30 + i:02d}") for i in range(12)],
            "PitOutTime": [pd.NaT] * 12,
            "PitInTime": [pd.NaT] * 12,
        }
    )

    short_laps, long_laps = _classify_run_laps(laps)
    assert len(short_laps) == 4
    assert len(long_laps) == 8


def test_select_program_aware_laps_balanced_prefers_representatives():
    laps = pd.DataFrame(
        {
            "Driver": ["DRV"] * 12,
            "Stint": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            "Compound": ["C3"] * 12,
            "LapTime": [pd.to_timedelta(f"0:01:{30 + i:02d}") for i in range(12)],
            "PitOutTime": [pd.NaT] * 12,
            "PitInTime": [pd.NaT] * 12,
        }
    )

    selected = _select_program_aware_laps(laps, run_profile="balanced")
    # One representative per stint/compound slice.
    assert len(selected) == 2


def test_select_program_aware_laps_short_run_is_pace_based_not_stint_length():
    # The fast green laps live inside a long (6-lap) stint; a 2-lap cooldown stint at the
    # end of the session is far slower. Short-run pace must be selected by lap pace, so the
    # quick laps are kept and the slow cooldown stint is excluded -- this is the Monaco-FP2
    # failure mode (a cooldown stint was previously chosen as "short run").
    laps = pd.DataFrame(
        {
            "Driver": ["DRV"] * 8,
            "Stint": [1, 1, 1, 1, 1, 1, 2, 2],
            "Compound": ["C3"] * 8,
            "LapTime": [
                pd.to_timedelta("0:01:35"),
                pd.to_timedelta("0:01:30"),  # fastest
                pd.to_timedelta("0:01:31"),
                pd.to_timedelta("0:01:33"),
                pd.to_timedelta("0:01:36"),
                pd.to_timedelta("0:01:38"),
                pd.to_timedelta("0:02:13"),  # cooldown
                pd.to_timedelta("0:02:19"),  # cooldown
            ],
            "PitOutTime": [pd.NaT] * 8,
            "PitInTime": [pd.NaT] * 8,
        }
    )

    selected = _select_program_aware_laps(laps, run_profile="short_run")
    selected_seconds = sorted(
        pd.to_timedelta(selected["LapTime"], errors="coerce").dt.total_seconds().tolist()
    )

    # Picks the driver's fastest clean green laps; cooldown laps are excluded by pace.
    assert selected_seconds == [90.0, 91.0, 93.0]
    assert max(selected_seconds) < 120.0


def test_select_short_run_laps_keeps_fast_lap_from_long_stint():
    # A driver who only ran long push-stints still gets a representative short-run sample.
    laps = pd.DataFrame(
        {
            "Driver": ["AAA"] * 7 + ["BBB"] * 2,
            "Stint": [1, 1, 1, 1, 1, 1, 1, 5, 5],
            "LapTime": [pd.to_timedelta(f"0:01:{34 + i:02d}") for i in range(7)]
            + [pd.to_timedelta("0:02:10"), pd.to_timedelta("0:02:15")],
            "PitOutTime": [pd.NaT] * 9,
            "PitInTime": [pd.NaT] * 9,
        }
    )
    selected = _select_short_run_laps(laps, top_n=3)
    by_driver = {
        drv: sorted(pd.to_timedelta(g["LapTime"]).dt.total_seconds().tolist())
        for drv, g in selected.groupby("Driver")
    }
    assert by_driver["AAA"] == [94.0, 95.0, 96.0]  # fastest 3 of the long stint
    assert by_driver["BBB"] == [130.0, 135.0]  # only laps available for BBB


def test_drop_implausible_laps_removes_cooldown_and_inout():
    laps = pd.DataFrame(
        {
            "Driver": ["DRV"] * 5,
            "LapTime": [
                pd.to_timedelta("0:01:14"),  # best 74s
                pd.to_timedelta("0:01:15"),
                pd.to_timedelta("0:01:18"),  # heavy fuel, within 20% -> kept
                pd.to_timedelta("0:02:13"),  # cooldown -> dropped
                pd.to_timedelta("0:01:44"),  # in/out-ish 104s -> dropped (>74*1.2)
            ],
        }
    )
    kept = _drop_implausible_laps(laps)
    kept_seconds = sorted(pd.to_timedelta(kept["LapTime"]).dt.total_seconds().tolist())
    assert kept_seconds == [74.0, 75.0, 78.0]


def test_detect_snapshot_anomalies_flags_slow_pace_and_big_delta():
    payload = {
        "teams": {
            "Ferrari": {
                "profiles": {"short_run": {"overall_pace_seconds": 73.1}},
                "driver_deltas_seconds": {"short_run": {"HAM": -0.1, "LEC": 0.1}},
            },
            "McLaren": {
                "profiles": {"short_run": {"overall_pace_seconds": 104.3}},  # +31s off field
                "driver_deltas_seconds": {"short_run": {"NOR": -29.0, "PIA": 29.0}},
            },
        }
    }
    warnings = detect_snapshot_anomalies(payload)
    text = " ".join(warnings)
    assert any("McLaren" in w and "off the field best" in w for w in warnings)
    assert "NOR" in text and "PIA" in text
    # The clean team raises no warning.
    assert not any("Ferrari" in w for w in warnings)


def test_detect_snapshot_anomalies_clean_snapshot_has_no_warnings():
    payload = {
        "teams": {
            "Ferrari": {
                "profiles": {"short_run": {"overall_pace_seconds": 73.1}},
                "driver_deltas_seconds": {"short_run": {"HAM": -0.1, "LEC": 0.1}},
            },
            "McLaren": {
                "profiles": {"short_run": {"overall_pace_seconds": 74.95}},
                "driver_deltas_seconds": {"short_run": {"NOR": 0.62, "PIA": -0.62}},
            },
        }
    }
    assert detect_snapshot_anomalies(payload) == []


def test_blend_strengths_caps_checkpoint_move_from_prior():
    # A corrupted checkpoint score (0.14) would drag a 0.671 prior to ~0.255 at 0.784 weight;
    # the move cap bounds the swing to prior - limit.
    model_strengths = {"McLaren": 0.671, "Ferrari": 0.83}
    checkpoint_perf = {"McLaren": 0.14, "Ferrari": 0.94}
    blended = _blend_strengths(
        model_strengths=model_strengths,
        fp_performance=None,
        testing_fallback_performance=checkpoint_perf,
        uses_checkpoint_practice_profiles=True,
        checkpoint_practice_blend_weight=0.784,
        checkpoint_testing_fallback_performance=checkpoint_perf,
        fp_blend_weight=0.784,
        practice_like_profile_label="FP2",
        practice_like_blend_weight=0.784,
        blend_team_strength_fn=blend_team_strength,
        apply_testing_fallback_adjustment_fn=lambda **_: model_strengths,
        checkpoint_max_strength_move=0.40,
    )
    # Without the cap McLaren would land ~0.255; the cap holds it at >= 0.671 - 0.40.
    assert blended["McLaren"] >= 0.671 - 0.40 - 1e-9
    assert blended["McLaren"] == pytest.approx(0.271, abs=1e-3)
    # Ferrari moves up only modestly and stays within the cap of its prior.
    assert abs(blended["Ferrari"] - 0.83) <= 0.40 + 1e-9


def test_filter_valid_laps_applies_absolute_pace_floor():
    laps = pd.DataFrame(
        {
            "Driver": ["DRV"] * 4,
            "LapTime": [
                pd.to_timedelta("0:01:14"),
                pd.to_timedelta("0:01:16"),
                pd.to_timedelta("0:02:13"),  # 133s junk -> dropped by floor
                pd.NaT,  # untimed -> dropped
            ],
            "PitOutTime": [pd.NaT] * 4,
            "PitInTime": [pd.NaT] * 4,
        }
    )
    valid = _filter_valid_laps(laps)
    secs = sorted(pd.to_timedelta(valid["LapTime"]).dt.total_seconds().tolist())
    assert secs == [74.0, 76.0]


def test_extract_team_payload_uses_per_lap_top_speed_traps():
    laps = pd.DataFrame(
        {
            "LapTime": [pd.to_timedelta("0:01:20"), pd.to_timedelta("0:01:21")],
            "SpeedST": [310.0, 315.0],
            "SpeedFL": [300.0, 301.0],
            "SpeedI1": [260.0, 261.0],
            "SpeedI2": [270.0, 271.0],
        }
    )

    payload = _extract_team_payload(laps)

    assert payload["speed_profile"]["top_speed"] == 312.5


def test_extract_team_payload_attaches_braking_proxy_from_telemetry():
    class _TelemetryLap:
        def __init__(self, telemetry: pd.DataFrame):
            self._telemetry = telemetry.copy()

        def get_telemetry(self):
            return self._telemetry.copy()

    class _TelemetryLaps(pd.DataFrame):
        _metadata = ["_lap_objects"]

        def __init__(self, *args, lap_objects=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._lap_objects = list(lap_objects or [])

        @property
        def _constructor(self):
            return _TelemetryLaps

        def iterlaps(self, require=None):
            del require
            yield from enumerate(self._lap_objects)

    laps = _TelemetryLaps(
        {
            "LapTime": [pd.to_timedelta("0:01:20"), pd.to_timedelta("0:01:21")],
            "SpeedST": [310.0, 315.0],
            "SpeedFL": [300.0, 301.0],
        },
        lap_objects=[
            _TelemetryLap(pd.DataFrame({"Brake": [0, 0, 35, 60, 0]})),
            _TelemetryLap(pd.DataFrame({"Brake": [0, 15, 25, 0, 0]})),
        ],
    )

    payload = _extract_team_payload(laps)

    assert payload["braking_profile"]["braking_pct"] == pytest.approx(40.0)


def test_extract_team_payload_ignores_unloaded_braking_telemetry():
    """Missing FastF1 telemetry should not abort practice refresh."""

    class _UnloadedTelemetryLap:
        def get_telemetry(self):
            raise DataNotLoadedError("telemetry not loaded")

        def get_car_data(self):
            raise DataNotLoadedError("car data not loaded")

    class _TelemetryLaps(pd.DataFrame):
        _metadata = ["_lap_objects"]

        def __init__(self, *args, lap_objects=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._lap_objects = list(lap_objects or [])

        @property
        def _constructor(self):
            return _TelemetryLaps

        def iterlaps(self, require=None):
            del require
            yield from enumerate(self._lap_objects)

    laps = _TelemetryLaps(
        {
            "LapTime": [pd.to_timedelta("0:01:20"), pd.to_timedelta("0:01:21")],
            "SpeedST": [310.0, 315.0],
            "SpeedFL": [300.0, 301.0],
        },
        lap_objects=[_UnloadedTelemetryLap(), _UnloadedTelemetryLap()],
    )

    payload = _extract_team_payload(laps)

    assert "braking_profile" not in payload
    assert payload["speed_profile"]["top_speed"] == pytest.approx(312.5)


def test_select_program_aware_laps_preserves_fastf1_laps_behaviour():
    """Representative lap selection should keep telemetry-capable lap containers."""

    class _TelemetryLaps(pd.DataFrame):
        _metadata = ["_lap_objects"]

        def __init__(self, *args, lap_objects=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._lap_objects = list(lap_objects or [])

        @property
        def _constructor(self):
            return _TelemetryLaps

        def iterlaps(self):
            yield from enumerate(self._lap_objects)

    laps = _TelemetryLaps(
        {
            "Driver": ["DRV"] * 12,
            "Stint": ([1] * 4) + ([2] * 8),
            "Compound": (["SOFT"] * 4) + (["MEDIUM"] * 8),
            "LapTime": [pd.to_timedelta(f"0:01:{20 + index:02d}") for index in range(12)],
        },
        lap_objects=[f"lap-{index + 1}" for index in range(12)],
    )

    selected = _select_program_aware_laps(laps, run_profile="balanced")

    assert isinstance(selected, _TelemetryLaps)
    assert callable(getattr(selected, "iterlaps", None))
    assert len(selected) == 2


def test_collect_session_metrics_attaches_raw_top_speed_from_all_valid_laps(monkeypatch):
    from src.systems import testing_updater as testing_updater

    class DummySession:
        def __init__(self, laps):
            self.laps = laps

    laps = pd.DataFrame(
        {
            "Team": ["Ferrari"] * 12,
            "Driver": ["LEC"] * 12,
            "Stint": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            "Compound": ["C3"] * 12,
            "LapNumber": list(range(1, 13)),
            "LapTime": [
                pd.to_timedelta("0:01:32"),
                pd.to_timedelta("0:01:30"),
                pd.to_timedelta("0:01:48"),
                pd.to_timedelta("0:01:31"),
                pd.to_timedelta("0:01:39"),
                pd.to_timedelta("0:01:40"),
                pd.to_timedelta("0:01:41"),
                pd.to_timedelta("0:01:42"),
                pd.to_timedelta("0:01:43"),
                pd.to_timedelta("0:01:44"),
                pd.to_timedelta("0:01:45"),
                pd.to_timedelta("0:01:46"),
            ],
            "SpeedST": [
                299.0,
                301.0,
                280.0,
                300.0,
                316.0,
                317.0,
                318.0,
                321.0,
                320.0,
                319.0,
                318.0,
                317.0,
            ],
            "PitOutTime": [pd.NaT] * 12,
            "PitInTime": [pd.NaT] * 12,
        }
    )
    session = DummySession(laps=laps)
    captured_payload = {}

    def _capture_extract(payload, session_name):
        captured_payload.update(payload)
        return {"Ferrari": {"top_speed": 0.12}}

    monkeypatch.setattr(testing_updater, "extract_all_teams_performance", _capture_extract)

    perf, tire = testing_updater._collect_session_metrics(
        session=session,
        session_key="FP1",
        known_teams={"Ferrari"},
        run_profile="short_run",
    )

    assert tire == {}
    # speed_profile is now derived from the three fastest clean laps (median SpeedST of
    # 299/300/301 = 300.0) rather than a single best-stint lap.
    assert captured_payload["Ferrari"]["FP1"]["speed_profile"]["top_speed"] == 300.0
    assert perf["Ferrari"]["top_speed"] == 0.12
    assert perf["Ferrari"]["top_speed_kph"] == pytest.approx(319.9, abs=1e-4)
    # Short-run pace is the robust median of the driver's three fastest clean laps
    # (90/91/92s) -> 91.0, instead of the single best lap.
    assert perf["Ferrari"]["overall_pace_seconds"] == 91.0


def test_count_team_selected_laps_avoids_short_stint_cooldown_laps():
    class DummySession:
        def __init__(self, laps):
            self.laps = laps

    laps = pd.DataFrame(
        {
            "Team": ["McLaren"] * 8,
            "Driver": ["NOR"] * 8,
            "Stint": [1, 1, 1, 1, 2, 2, 2, 2],
            "LapTime": [
                pd.to_timedelta("0:01:35"),
                pd.to_timedelta("0:01:30"),
                pd.to_timedelta("0:01:55"),
                pd.to_timedelta("0:01:31"),
                pd.to_timedelta("0:01:42"),
                pd.to_timedelta("0:01:40"),
                pd.to_timedelta("0:01:41"),
                pd.to_timedelta("0:01:43"),
            ],
            "PitOutTime": [pd.NaT] * 8,
            "PitInTime": [pd.NaT] * 8,
        }
    )
    session = DummySession(laps=laps)

    counts = _count_team_selected_laps(session, {"McLaren"}, run_profile="short_run")

    # Pace-based short-run keeps the driver's three fastest clean laps (90/91/95s);
    # the 115s cooldown lap is still excluded.
    assert counts["McLaren"] == 3.0


def test_testing_event_and_session_parsing():
    assert _is_testing_event("Pre-Season Testing")
    assert _is_testing_event("Testing 2")
    assert not _is_testing_event("Bahrain Grand Prix")

    assert _extract_testing_day("Day 1") == 1
    assert _extract_testing_day("Practice 2") == 2
    assert _extract_testing_day("FP3") == 3
    assert _extract_testing_day("Qualifying") is None

    assert _extract_testing_number("Testing 2") == 2
    assert _extract_testing_number("Pre-Season Test 1") == 1
    assert _extract_testing_number("Pre-Season Testing") is None


def test_resolve_testing_backends():
    assert _resolve_testing_backends("auto") == ("f1timing", "fastf1", None)
    assert _resolve_testing_backends("fastf1") == ("fastf1",)
    assert _resolve_testing_backends("f1timing") == ("f1timing",)


def test_resolve_testing_cache_dir():
    assert str(_resolve_testing_cache_dir(None)) == "data/raw/.fastf1_cache_testing"
    assert (
        str(_resolve_testing_cache_dir("_tmp_fastf1_cache_testing_2026"))
        == "data/raw/_tmp_fastf1_cache_testing_2026"
    )
    assert (
        str(_resolve_testing_cache_dir("./_tmp_fastf1_cache_testing_2026"))
        == "data/raw/_tmp_fastf1_cache_testing_2026"
    )
    assert str(_resolve_testing_cache_dir("data/raw/.fastf1_cache")) == "data/raw/.fastf1_cache"


def test_coerce_utc_datetime_and_started_window():
    now = datetime(2026, 2, 11, 12, 0, tzinfo=UTC)

    assert _coerce_utc_datetime(None) is None

    event = {"Session1DateUtc": now - timedelta(minutes=30)}
    assert _testing_session_has_started(event, 1, now_utc=now)

    future_event = {"Session1DateUtc": now + timedelta(hours=2)}
    assert not _testing_session_has_started(future_event, 1, now_utc=now)

    # Missing timestamp should not block loading.
    unknown_event = {"Session1DateUtc": None}
    assert _testing_session_has_started(unknown_event, 1, now_utc=now)


def test_normalize_testing_event_sessions_day_labels():
    event = {
        "Session1": "Day 1",
        "Session2": "Day 2",
        "Session3": "Practice 3",
    }

    _normalize_testing_event_sessions(event)

    assert event["Session1"] == "Practice 1"
    assert event["Session2"] == "Practice 2"
    assert event["Session3"] == "Practice 3"


def test_update_from_testing_sessions_handles_null_testing_characteristics(tmp_path, patcher):
    from src.systems import testing_updater

    data_dir = tmp_path / "data" / "processed" / "car_characteristics"
    data_dir.mkdir(parents=True)
    characteristics_file = data_dir / "2026_car_characteristics.json"
    characteristics_file.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "directionality": {
                            "max_speed": 0.0,
                            "slow_corner_speed": 0.0,
                            "medium_corner_speed": 0.0,
                            "high_corner_speed": 0.0,
                        },
                        "testing_characteristics": None,
                    }
                }
            }
        )
    )

    patcher.setattr(
        testing_updater,
        "_load_sessions_for_event",
        lambda **kwargs: [("Day 1", object())],
    )
    patcher.setattr(
        testing_updater,
        "_collect_session_metrics",
        lambda **kwargs: ({"McLaren": {"overall_pace": 0.7}}, {}),
    )

    summary = testing_updater.update_from_testing_sessions(
        year=2026,
        events=["Pre-Season Testing"],
        data_dir=str(tmp_path / "data" / "processed"),
        dry_run=True,
    )

    assert summary["updated_teams"] == ["McLaren"]


def test_update_from_testing_sessions_supports_characteristics_year_override(tmp_path, patcher):
    from src.systems import testing_updater

    data_dir = tmp_path / "data" / "processed" / "car_characteristics"
    data_dir.mkdir(parents=True)
    (data_dir / "2026_car_characteristics.json").write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "directionality": {
                            "max_speed": 0.0,
                            "slow_corner_speed": 0.0,
                            "medium_corner_speed": 0.0,
                            "high_corner_speed": 0.0,
                        },
                        "testing_characteristics": {},
                    }
                }
            }
        )
    )

    patcher.setattr(
        testing_updater,
        "_load_sessions_for_event",
        lambda **kwargs: [("Day 1", object())],
    )
    patcher.setattr(
        testing_updater,
        "_collect_session_metrics",
        lambda **kwargs: ({"McLaren": {"overall_pace": 0.7}}, {}),
    )

    summary = testing_updater.update_from_testing_sessions(
        year=2025,
        characteristics_year=2026,
        events=["Pre-Season Testing"],
        data_dir=str(tmp_path / "data" / "processed"),
        dry_run=True,
    )

    assert summary["characteristics_year"] == 2026


def test_update_from_testing_sessions_tracks_team_sessions_used(tmp_path, patcher):
    from src.systems import testing_updater

    data_dir = tmp_path / "data" / "processed" / "car_characteristics"
    data_dir.mkdir(parents=True)
    characteristics_file = data_dir / "2026_car_characteristics.json"
    characteristics_file.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "directionality": {
                            "max_speed": 0.0,
                            "slow_corner_speed": 0.0,
                            "medium_corner_speed": 0.0,
                            "high_corner_speed": 0.0,
                        },
                        "testing_characteristics": {},
                    }
                }
            }
        )
    )

    patcher.setattr(
        testing_updater,
        "_load_sessions_for_event",
        lambda **kwargs: [("Day 1", object()), ("Day 2", object())],
    )

    def _mock_collect_session_metrics(**kwargs):
        session_key = kwargs["session_key"]
        if session_key == "Day 1":
            return {"McLaren": {"overall_pace": 0.6}}, {}
        return {"McLaren": {"overall_pace": 0.8}}, {}

    patcher.setattr(
        testing_updater,
        "_collect_session_metrics",
        _mock_collect_session_metrics,
    )
    patcher.setattr(
        testing_updater,
        "_count_team_selected_laps",
        lambda session, known_teams, run_profile: {"McLaren": 10.0},
    )

    summary = testing_updater.update_from_testing_sessions(
        year=2026,
        events=["Pre-Season Testing"],
        data_dir=str(tmp_path / "data" / "processed"),
        session_aggregation="median",
        dry_run=False,
    )

    with open(characteristics_file) as f:
        updated = json.load(f)

    assert summary["session_aggregation"] == "median"
    assert updated["teams"]["McLaren"]["testing_characteristics"]["sessions_used"] == 2
    assert updated["teams"]["McLaren"]["testing_characteristics"]["session_aggregation"] == "median"
    assert updated["directionality_meta"]["session_aggregation"] == "median"


def test_update_from_testing_sessions_includes_run_profile_in_summary(tmp_path, patcher):
    from src.systems import testing_updater

    data_dir = tmp_path / "data" / "processed" / "car_characteristics"
    data_dir.mkdir(parents=True)
    characteristics_file = data_dir / "2026_car_characteristics.json"
    characteristics_file.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "directionality": {
                            "max_speed": 0.0,
                            "slow_corner_speed": 0.0,
                            "medium_corner_speed": 0.0,
                            "high_corner_speed": 0.0,
                        },
                        "testing_characteristics": {},
                    }
                }
            }
        )
    )

    patcher.setattr(
        testing_updater,
        "_load_sessions_for_event",
        lambda **kwargs: [("Day 1", object())],
    )
    patcher.setattr(
        testing_updater,
        "_collect_session_metrics",
        lambda **kwargs: ({"McLaren": {"overall_pace": 0.7}}, {}),
    )
    patcher.setattr(
        testing_updater,
        "_count_team_selected_laps",
        lambda session, known_teams, run_profile: {"McLaren": 8.0},
    )

    summary = testing_updater.update_from_testing_sessions(
        year=2026,
        events=["Testing 1"],
        data_dir=str(tmp_path / "data" / "processed"),
        run_profile="long_run",
        dry_run=False,
    )

    with open(characteristics_file) as f:
        updated = json.load(f)

    assert summary["run_profile"] == "long_run"
    assert updated["directionality_meta"]["run_profile"] == "long_run"
    assert "testing_characteristics_profiles" in updated["teams"]["McLaren"]
    assert "short_run" in updated["teams"]["McLaren"]["testing_characteristics_profiles"]
    assert "long_run" in updated["teams"]["McLaren"]["testing_characteristics_profiles"]


def test_update_from_testing_sessions_suggests_fresh_cache_on_data_not_loaded(tmp_path, patcher):
    from src.systems import testing_updater

    data_dir = tmp_path / "data" / "processed" / "car_characteristics"
    data_dir.mkdir(parents=True)
    (data_dir / "2026_car_characteristics.json").write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "directionality": {
                            "max_speed": 0.0,
                            "slow_corner_speed": 0.0,
                            "medium_corner_speed": 0.0,
                            "high_corner_speed": 0.0,
                        },
                        "testing_characteristics": {},
                    }
                }
            }
        )
    )

    patcher.setattr(
        testing_updater,
        "_load_sessions_for_event",
        lambda **kwargs: [],
    )

    error_messages = [
        "testing#1/day1 backend=f1timing -> DataNotLoadedError: sample",
        "testing#1/day1 backend=fastf1 -> DataNotLoadedError: sample",
    ]

    def _inject_errors(**kwargs):
        kwargs["error_messages"].extend(error_messages)
        return []

    patcher.setattr(testing_updater, "_load_sessions_for_event", _inject_errors)

    try:
        testing_updater.update_from_testing_sessions(
            year=2026,
            events=["Testing 1"],
            data_dir=str(tmp_path / "data" / "processed"),
            dry_run=True,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ValueError for no loadable sessions")

    assert "Likely cache issue" in message
    assert "--force-renew-cache" in message


def test_update_from_testing_sessions_uses_session_event_name_for_compounds(tmp_path, patcher):
    from src.systems import testing_updater

    data_dir = tmp_path / "data" / "processed" / "car_characteristics"
    data_dir.mkdir(parents=True)
    (data_dir / "2026_car_characteristics.json").write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "directionality": {
                            "max_speed": 0.0,
                            "slow_corner_speed": 0.0,
                            "medium_corner_speed": 0.0,
                            "high_corner_speed": 0.0,
                        },
                        "testing_characteristics": {},
                        "compound_characteristics": {},
                    }
                }
            }
        )
    )

    class DummySession:
        def __init__(self):
            self.laps = pd.DataFrame(
                {
                    "Team": ["McLaren", "McLaren"],
                    "LapTime": [pd.to_timedelta("0:01:30"), pd.to_timedelta("0:01:31")],
                }
            )

    def _mock_load_sessions_for_event(**kwargs):
        return [("Day 1", DummySession())]

    patcher.setattr(testing_updater, "_load_sessions_for_event", _mock_load_sessions_for_event)
    patcher.setattr(
        testing_updater,
        "_collect_session_metrics",
        lambda **kwargs: ({"McLaren": {"overall_pace": 0.7}}, {}),
    )
    patcher.setattr(
        testing_updater,
        "_count_team_selected_laps",
        lambda session, known_teams, run_profile: {"McLaren": 10.0},
    )
    patcher.setattr(
        testing_updater,
        "extract_compound_metrics",
        lambda team_laps, canonical_team, race_name: {"SOFT": {"laps_sampled": 10}},
    )
    patcher.setattr(
        testing_updater,
        "normalize_compound_metrics_across_teams",
        lambda metrics, race_name: {"McLaren": {"SOFT": {"laps_sampled": 10}}},
    )

    race_names = []

    def _capture_aggregate(existing, new, blend_weight, race_name):
        race_names.append(race_name)
        return new

    patcher.setattr(testing_updater, "aggregate_compound_samples", _capture_aggregate)

    summary = testing_updater.update_from_testing_sessions(
        year=2026,
        events=["Event One", "Event Two"],
        data_dir=str(tmp_path / "data" / "processed"),
        dry_run=True,
    )

    assert summary["updated_teams"] == ["McLaren"]
    assert race_names == ["Event One", "Event Two"]


def test_update_from_testing_sessions_blends_profiles_cumulatively_across_circuits(
    tmp_path, patcher
):
    from src.systems import testing_updater

    data_dir = tmp_path / "data" / "processed" / "car_characteristics"
    data_dir.mkdir(parents=True)
    characteristics_file = data_dir / "2026_car_characteristics.json"
    characteristics_file.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "directionality": {
                            "max_speed": 0.0,
                            "slow_corner_speed": 0.0,
                            "medium_corner_speed": 0.0,
                            "high_corner_speed": 0.0,
                        },
                        "testing_characteristics": {},
                        "testing_characteristics_profiles": {},
                    }
                }
            }
        )
    )

    class DummySession:
        def __init__(self, event_name: str):
            self.event_name = event_name
            self.laps = pd.DataFrame(
                {
                    "Team": ["McLaren"] * 6,
                    "Driver": ["NOR"] * 6,
                    "LapTime": [pd.to_timedelta(f"0:01:{30 + i:02d}") for i in range(6)],
                }
            )

    patcher.setattr(
        testing_updater,
        "_load_sessions_for_event",
        lambda **kwargs: [("FP1", DummySession(kwargs["event_name"]))],
    )

    def _mock_collect_session_metrics(**kwargs):
        session = kwargs["session"]
        pace = 0.60 if session.event_name == "Bahrain Grand Prix" else 0.80
        return {"McLaren": {"overall_pace": pace}}, {}

    patcher.setattr(testing_updater, "_collect_session_metrics", _mock_collect_session_metrics)
    patcher.setattr(
        testing_updater,
        "_count_team_selected_laps",
        lambda session, known_teams, run_profile: {"McLaren": 8.0},
    )

    testing_updater.update_from_testing_sessions(
        year=2026,
        events=["Bahrain Grand Prix"],
        data_dir=str(tmp_path / "data" / "processed"),
        new_weight=0.25,
        dry_run=False,
    )
    testing_updater.update_from_testing_sessions(
        year=2026,
        events=["Monaco Grand Prix"],
        data_dir=str(tmp_path / "data" / "processed"),
        new_weight=0.25,
        dry_run=False,
    )

    updated = json.loads(characteristics_file.read_text())
    team_payload = updated["teams"]["McLaren"]["testing_characteristics"]

    assert team_payload["sessions_blended"] == 2
    assert team_payload["sessions_used"] == 2
    assert team_payload["circuits_observed"] == ["Bahrain Grand Prix", "Monaco Grand Prix"]
    assert 0.60 < team_payload["overall_pace"] < 0.65

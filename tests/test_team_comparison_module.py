"""Focused tests for team-comparison helpers and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard import team_comparison


def _stub_streamlit_team(patcher):
    calls = {
        "info": [],
        "plotly": 0,
        "plotly_figures": [],
        "dataframe": 0,
        "captions": [],
        "success": [],
    }
    patcher.setattr(team_comparison.st, "subheader", lambda *_args, **_kwargs: None)
    patcher.setattr(team_comparison.st, "selectbox", lambda _label, options, **_kwargs: options[0])
    patcher.setattr(team_comparison.st, "button", lambda *_args, **_kwargs: False)
    patcher.setattr(
        team_comparison.st,
        "multiselect",
        lambda _label, options, default=None, **_kwargs: default if default is not None else [],
    )
    patcher.setattr(team_comparison.st, "info", lambda message: calls["info"].append(str(message)))
    patcher.setattr(
        team_comparison.st, "success", lambda message: calls["success"].append(str(message))
    )
    patcher.setattr(team_comparison.st, "rerun", lambda: None)
    patcher.setattr(
        team_comparison.st,
        "plotly_chart",
        lambda figure, **_kwargs: (
            calls["plotly_figures"].append(figure),
            calls.__setitem__("plotly", calls["plotly"] + 1),
        ),
    )
    patcher.setattr(
        team_comparison.st,
        "dataframe",
        lambda *_args, **_kwargs: calls.__setitem__("dataframe", calls["dataframe"] + 1),
    )
    patcher.setattr(
        team_comparison.st,
        "caption",
        lambda message: calls["captions"].append(str(message)),
    )
    return calls


def _write_snapshot(
    root: Path,
    *,
    year: int,
    event_name: str,
    session_name: str,
    team_profiles: dict[str, dict[str, dict[str, float]]],
    round_number: int,
    session_order: int,
    session_started_at: str | None = None,
    captured_at: str | None = None,
) -> None:
    path = root / "car_characteristics_snapshot" / str(year) / event_name / f"{session_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "year": year,
                "event_name": event_name,
                "session_name": session_name,
                "round_number": round_number,
                "session_order": session_order,
                "captured_at": captured_at or f"2027-03-0{session_order}T12:00:00+00:00",
                "session_started_at": session_started_at,
                "teams": {
                    team_name: {"profiles": profiles}
                    for team_name, profiles in team_profiles.items()
                },
            }
        )
    )


def test_coerce_unit_metric_and_color_helpers():
    assert team_comparison._coerce_unit_metric("bad") is None
    assert team_comparison._coerce_unit_metric(float("nan")) is None
    assert team_comparison._coerce_unit_metric(-1.0) == 0.0
    assert team_comparison._coerce_unit_metric(1.3) == 1.0
    assert team_comparison._hex_to_rgba("#FF0000", 0.2) == "rgba(255, 0, 0, 0.2)"
    assert team_comparison._hex_to_rgba("bad", 0.2) == "rgba(124, 135, 152, 0.2)"


def test_collect_profile_names_and_resolve_profile_metrics():
    payload = {
        "McLaren": {
            "testing_characteristics_profiles": {
                "short_run": {"overall_pace": 0.8},
                "balanced": {"overall_pace": 0.7},
            }
        },
        "Ferrari": {"testing_characteristics": {"run_profile": "long_run", "overall_pace": 0.6}},
    }

    names = team_comparison._collect_profile_names(payload)
    assert names[:3] == ["balanced", "short_run", "long_run"]
    assert team_comparison._resolve_profile_metrics(payload["McLaren"], "short_run") == {
        "overall_pace": 0.8
    }
    assert team_comparison._resolve_profile_metrics(payload["Ferrari"], "long_run") == {
        "run_profile": "long_run",
        "overall_pace": 0.6,
    }


def test_collect_profile_names_empty_when_no_testing_profiles():
    payload = {
        "McLaren": {"overall_performance": 0.8},
        "Ferrari": {"overall_performance": 0.75},
    }

    assert team_comparison._collect_profile_names(payload) == []


def test_canonicalize_teams_payload_for_comparison_merges_sauber_into_audi():
    payload = {
        "Sauber": {"overall_performance": 0.38},
        "Audi": {
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.61,
                    "slow_corner_performance": 0.49,
                    "medium_corner_performance": 0.44,
                    "fast_corner_performance": 0.52,
                    "braking_performance": 0.47,
                    "top_speed": 0.50,
                    "tire_deg_performance": 0.55,
                }
            }
        },
    }

    canonical_payload = team_comparison._canonicalize_teams_payload_for_comparison(payload)

    assert list(canonical_payload.keys()) == ["Audi"]
    assert canonical_payload["Audi"]["overall_performance"] == 0.38
    assert (
        canonical_payload["Audi"]["testing_characteristics_profiles"]["balanced"]["overall_pace"]
        == 0.61
    )


def test_build_team_comparison_dataframe_maps_sauber_payload_to_audi():
    payload = {
        "Sauber": {
            "overall_performance": 0.38,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.61,
                    "slow_corner_performance": 0.49,
                    "medium_corner_performance": 0.44,
                    "fast_corner_performance": 0.52,
                    "braking_performance": 0.47,
                    "top_speed": 0.50,
                    "tire_deg_performance": 0.55,
                }
            },
        }
    }

    canonical_payload = team_comparison._canonicalize_teams_payload_for_comparison(payload)
    frame, neutral_fallbacks = team_comparison._build_team_comparison_dataframe(
        teams_payload=canonical_payload,
        selected_teams=["Audi"],
        profile="balanced",
    )

    assert frame.iloc[0]["Team"] == "Audi"
    assert frame.iloc[0]["Overall Performance"] == 0.38
    assert neutral_fallbacks == 0


def test_build_team_comparison_dataframe_uses_neutral_fallback_for_missing_profiles():
    payload = {"Sauber": {"overall_performance": 0.38}}

    canonical_payload = team_comparison._canonicalize_teams_payload_for_comparison(payload)
    frame, neutral_fallbacks = team_comparison._build_team_comparison_dataframe(
        teams_payload=canonical_payload,
        selected_teams=["Audi"],
        profile="balanced",
    )

    assert frame.iloc[0]["Team"] == "Audi"
    assert frame.iloc[0]["Overall Performance"] == 0.38
    assert frame.iloc[0]["Overall Pace"] == 0.5
    assert frame.iloc[0]["Slow Corners"] == 0.5
    assert frame.iloc[0]["Medium Corners"] == 0.5
    assert frame.iloc[0]["Fast Corners"] == 0.5
    assert frame.iloc[0]["Braking"] == 0.5
    assert frame.iloc[0]["Top Speed"] == 0.5
    assert frame.iloc[0]["Tire Deg"] == 0.5
    assert frame.iloc[0]["Radar Minus Prior"] == 0.12
    assert neutral_fallbacks == 7


def test_build_team_comparison_dataframe_uses_slope_based_tire_deg_display_value():
    payload = {
        "Red Bull Racing": {
            "overall_performance": 0.74,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.566,
                    "slow_corner_performance": 0.462,
                    "medium_corner_performance": 0.536,
                    "fast_corner_performance": 0.577,
                    "braking_performance": 0.462,
                    "top_speed": 0.618,
                    "tire_deg_performance": 0.0,
                    "tire_deg_slope": 0.3651,
                }
            },
        }
    }

    frame, neutral_fallbacks = team_comparison._build_team_comparison_dataframe(
        teams_payload=payload,
        selected_teams=["Red Bull Racing"],
        profile="balanced",
    )

    assert neutral_fallbacks == 0
    assert frame.iloc[0]["Tire Deg"] == pytest.approx(0.2043, abs=1e-4)


def test_build_team_comparison_dataframe_prefers_raw_top_speed_display_value():
    payload = {
        "McLaren": {
            "overall_performance": 0.85,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.887,
                    "slow_corner_performance": 0.937,
                    "medium_corner_performance": 0.938,
                    "fast_corner_performance": 0.773,
                    "braking_performance": 0.937,
                    "top_speed": 0.294,
                    "top_speed_kph": 310.0,
                    "tire_deg_performance": 0.495,
                }
            },
        },
        "Ferrari": {
            "overall_performance": 0.70,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.99,
                    "slow_corner_performance": 0.988,
                    "medium_corner_performance": 1.0,
                    "fast_corner_performance": 0.903,
                    "braking_performance": 0.988,
                    "top_speed": 0.176,
                    "top_speed_kph": 312.0,
                    "tire_deg_performance": 0.471,
                }
            },
        },
        "Mercedes": {
            "overall_performance": 0.75,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 1.0,
                    "slow_corner_performance": 1.0,
                    "medium_corner_performance": 0.961,
                    "fast_corner_performance": 1.0,
                    "braking_performance": 1.0,
                    "top_speed": 0.588,
                    "top_speed_kph": 317.0,
                    "tire_deg_performance": 0.809,
                }
            },
        },
    }

    frame, neutral_fallbacks = team_comparison._build_team_comparison_dataframe(
        teams_payload=payload,
        selected_teams=["McLaren", "Ferrari", "Mercedes"],
        profile="balanced",
    )

    rows = frame.set_index("Team")
    assert neutral_fallbacks == 0
    assert rows.loc["McLaren", "Top Speed"] == pytest.approx(0.4265, abs=1e-4)
    assert rows.loc["Ferrari", "Top Speed"] == pytest.approx(0.4971, abs=1e-4)
    assert rows.loc["Mercedes", "Top Speed"] == pytest.approx(0.6735, abs=1e-4)


def test_build_team_comparison_dataframe_prefers_raw_pace_and_corner_values():
    payload = {
        "Aston Martin": {
            "overall_performance": 0.47,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.1687,
                    "overall_pace_seconds": 90.40,
                    "slow_corner_performance": 0.2907,
                    "slow_corner_seconds": 30.12,
                    "medium_corner_performance": 0.0716,
                    "medium_corner_seconds": 28.03,
                    "fast_corner_performance": 0.0,
                    "fast_corner_seconds": 31.08,
                    "braking_performance": 0.2907,
                    "top_speed": 0.0,
                    "top_speed_kph": 305.0,
                    "tire_deg_performance": 0.795,
                }
            },
        },
        "Cadillac F1": {
            "overall_performance": 0.35,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.0,
                    "overall_pace_seconds": 90.72,
                    "slow_corner_performance": 0.0,
                    "slow_corner_seconds": 30.22,
                    "medium_corner_performance": 0.0,
                    "medium_corner_seconds": 28.11,
                    "fast_corner_performance": 0.1111,
                    "fast_corner_seconds": 31.02,
                    "braking_performance": 0.0,
                    "top_speed": 0.5294,
                    "top_speed_kph": 312.6,
                    "tire_deg_performance": 0.85,
                }
            },
        },
        "Mercedes": {
            "overall_performance": 0.75,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 1.0,
                    "overall_pace_seconds": 89.85,
                    "slow_corner_performance": 1.0,
                    "slow_corner_seconds": 29.84,
                    "medium_corner_performance": 0.961,
                    "medium_corner_seconds": 27.85,
                    "fast_corner_performance": 1.0,
                    "fast_corner_seconds": 30.60,
                    "braking_performance": 1.0,
                    "top_speed": 0.588,
                    "top_speed_kph": 314.6,
                    "tire_deg_performance": 0.809,
                }
            },
        },
    }

    frame, neutral_fallbacks = team_comparison._build_team_comparison_dataframe(
        teams_payload=payload,
        selected_teams=["Aston Martin", "Cadillac F1", "Mercedes"],
        profile="balanced",
    )

    rows = frame.set_index("Team")
    assert neutral_fallbacks == 0
    assert rows.loc["Aston Martin", "Overall Pace"] == pytest.approx(0.4222, abs=1e-3)
    assert rows.loc["Aston Martin", "Slow Corners"] == pytest.approx(0.3607, abs=1e-3)
    assert rows.loc["Aston Martin", "Medium Corners"] == pytest.approx(0.3878, abs=1e-3)
    assert rows.loc["Aston Martin", "Fast Corners"] == pytest.approx(0.2059, abs=1e-3)
    assert rows.loc["Cadillac F1", "Overall Pace"] == pytest.approx(0.2059, abs=1e-3)
    assert rows.loc["Cadillac F1", "Slow Corners"] == pytest.approx(0.2062, abs=1e-3)
    assert rows.loc["Cadillac F1", "Medium Corners"] == pytest.approx(0.2063, abs=1e-3)
    assert rows.loc["Cadillac F1", "Fast Corners"] == pytest.approx(0.2794, abs=1e-3)


def test_build_latest_snapshot_comparison_payload_prefers_snapshot_profiles():
    base_teams_payload = {
        "McLaren": {
            "overall_performance": 0.82,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.41,
                    "slow_corner_performance": 0.35,
                    "medium_corner_performance": 0.36,
                    "fast_corner_performance": 0.37,
                    "braking_performance": 0.38,
                    "top_speed": 0.39,
                    "tire_deg_performance": 0.40,
                }
            },
        }
    }
    latest_snapshot = {
        "event_name": "Chinese Grand Prix",
        "session_name": "SQ",
        "teams": {
            "McLaren": {
                "profiles": {
                    "balanced": {
                        "overall_pace": 0.74,
                        "slow_corner_performance": 0.71,
                        "medium_corner_performance": 0.72,
                        "fast_corner_performance": 0.73,
                        "braking_performance": 0.70,
                        "top_speed": 0.69,
                        "tire_deg_performance": 0.68,
                    }
                }
            }
        },
    }

    payload = team_comparison._build_latest_snapshot_comparison_payload(
        base_teams_payload=base_teams_payload,
        latest_snapshot=latest_snapshot,
    )

    assert payload["McLaren"]["overall_performance"] == 0.82
    assert (
        payload["McLaren"]["testing_characteristics_profiles"]["balanced"]["overall_pace"] == 0.74
    )
    assert payload["McLaren"]["testing_characteristics"]["slow_corner_performance"] == 0.71


def test_build_latest_snapshot_comparison_payload_carries_forward_latest_long_run_tire_deg():
    base_teams_payload = {
        "Ferrari": {
            "overall_performance": 0.84,
            "testing_characteristics_profiles": {
                "balanced": {
                    "overall_pace": 0.81,
                    "slow_corner_performance": 0.78,
                    "medium_corner_performance": 0.79,
                    "fast_corner_performance": 0.77,
                    "braking_performance": 0.8,
                    "top_speed": 0.76,
                    "tire_deg_performance": 0.7,
                }
            },
        }
    }
    prior_snapshot = {
        "event_name": "Australian Grand Prix",
        "session_name": "R",
        "teams": {
            "Ferrari": {
                "profiles": {
                    "balanced": {
                        "overall_pace": 0.85,
                        "tire_deg_performance": 0.66,
                        "tire_deg_slope": 0.21,
                    },
                    "long_run": {
                        "overall_pace": 0.79,
                        "tire_deg_performance": 0.67,
                        "tire_deg_slope": 0.19,
                    },
                }
            }
        },
    }
    latest_snapshot = {
        "event_name": "Chinese Grand Prix",
        "session_name": "FP1",
        "teams": {
            "Ferrari": {
                "profiles": {
                    "balanced": {
                        "overall_pace": 0.74,
                        "slow_corner_performance": 0.71,
                        "medium_corner_performance": 0.72,
                        "fast_corner_performance": 0.73,
                        "braking_performance": 0.70,
                        "top_speed": 0.69,
                        "tire_deg_performance": None,
                        "tire_deg_slope": None,
                    },
                    "long_run": {
                        "overall_pace": 0.72,
                        "top_speed": 0.65,
                        "tire_deg_performance": None,
                        "tire_deg_slope": None,
                    },
                }
            }
        },
    }

    payload = team_comparison._build_latest_snapshot_comparison_payload(
        base_teams_payload=base_teams_payload,
        latest_snapshot=latest_snapshot,
        snapshot_history=[prior_snapshot, latest_snapshot],
    )

    balanced = payload["Ferrari"]["testing_characteristics_profiles"]["balanced"]
    long_run = payload["Ferrari"]["testing_characteristics_profiles"]["long_run"]
    assert balanced["tire_deg_performance"] == 0.67
    assert balanced["tire_deg_slope"] == 0.19
    assert long_run["tire_deg_performance"] == 0.67
    assert long_run["tire_deg_slope"] == 0.19
    assert payload["Ferrari"]["testing_characteristics"]["tire_deg_performance"] == 0.67


def test_run_characteristics_season_sync_backfills_snapshots_only(patcher):
    import src.systems.testing_updater as testing_updater

    captured_kwargs = {}

    def _mock_backfill(**kwargs):
        captured_kwargs.update(kwargs)
        return {"loaded_sessions": ["Chinese Grand Prix::SQ"]}

    patcher.setattr(testing_updater, "backfill_season_snapshot_history", _mock_backfill)

    summary = team_comparison._run_characteristics_season_sync(
        2027,
        {
            "directionality_meta": {
                "testing_backend": "f1timing",
                "force_renew_cache": True,
                "run_profile": "short_run",
            }
        },
    )

    assert summary["loaded_sessions"] == ["Chinese Grand Prix::SQ"]
    assert captured_kwargs == {
        "year": 2027,
        "characteristics_year": 2027,
        "testing_backend": "f1timing",
        "force_renew_cache": True,
        "run_profile": "short_run",
        "dry_run": False,
    }


def test_load_team_characteristics_payload_handles_missing_and_invalid(tmp_path, patcher):
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    missing_payload, missing_path = team_comparison._load_team_characteristics_payload(2027)
    assert missing_payload is None
    assert missing_path.name == "2027_car_characteristics.json"

    path = tmp_path / "car_characteristics" / "2027_car_characteristics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{bad json")

    invalid_payload, invalid_path = team_comparison._load_team_characteristics_payload(2027)
    assert invalid_payload is None
    assert invalid_path == path


def test_render_team_comparison_section_handles_empty_selection(patcher, tmp_path):
    calls = _stub_streamlit_team(patcher)
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))
    patcher.setattr(team_comparison.st, "multiselect", lambda *_args, **_kwargs: [])

    data_path = tmp_path / "car_characteristics" / "2027_car_characteristics.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "overall_performance": 0.8,
                        "testing_characteristics": {"run_profile": "balanced", "overall_pace": 0.7},
                    }
                }
            }
        )
    )

    team_comparison._load_team_characteristics_payload.clear()
    team_comparison._load_team_snapshot_history.clear()
    team_comparison._render_team_comparison_section(year=2027)

    assert any("Select at least one team" in message for message in calls["info"])


def test_render_team_comparison_section_renders_chart_and_table(patcher, tmp_path):
    calls = _stub_streamlit_team(patcher)
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    data_path = tmp_path / "car_characteristics" / "2027_car_characteristics.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "overall_performance": 0.82,
                        "testing_characteristics_profiles": {
                            "balanced": {
                                "overall_pace": 0.78,
                                "slow_corner_performance": 0.74,
                                "medium_corner_performance": 0.77,
                                "fast_corner_performance": 0.79,
                                "braking_performance": 0.76,
                                "top_speed": 0.80,
                                "tire_deg_performance": 0.73,
                            }
                        },
                    },
                    "Ferrari": {
                        "overall_performance": 0.80,
                        "testing_characteristics_profiles": {
                            "balanced": {
                                "overall_pace": 0.75,
                                "slow_corner_performance": 0.72,
                                "medium_corner_performance": 0.74,
                                "fast_corner_performance": 0.76,
                                "braking_performance": 0.73,
                                "top_speed": 0.78,
                                "tire_deg_performance": 0.71,
                            }
                        },
                    },
                }
            }
        )
    )

    team_comparison._load_team_characteristics_payload.clear()
    team_comparison._load_team_snapshot_history.clear()
    team_comparison._render_team_comparison_section(year=2027)

    assert calls["plotly"] == 1
    assert calls["dataframe"] == 1
    assert any("profile=`balanced`" in caption for caption in calls["captions"])


def test_render_team_comparison_section_prefers_latest_snapshot_metrics(patcher, tmp_path):
    calls = _stub_streamlit_team(patcher)
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    data_path = tmp_path / "car_characteristics" / "2027_car_characteristics.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "overall_performance": 0.82,
                        "testing_characteristics_profiles": {
                            "balanced": {
                                "overall_pace": 0.32,
                                "slow_corner_performance": 0.31,
                                "medium_corner_performance": 0.30,
                                "fast_corner_performance": 0.29,
                                "braking_performance": 0.28,
                                "top_speed": 0.27,
                                "tire_deg_performance": 0.26,
                            }
                        },
                    }
                }
            }
        )
    )
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Bahrain Grand Prix",
        session_name="FP1",
        round_number=1,
        session_order=1,
        team_profiles={
            "McLaren": {
                "balanced": {
                    "overall_pace": 0.68,
                    "slow_corner_performance": 0.64,
                    "medium_corner_performance": 0.67,
                    "fast_corner_performance": 0.69,
                    "braking_performance": 0.66,
                    "top_speed": 0.70,
                    "tire_deg_performance": 0.63,
                }
            }
        },
    )

    team_comparison._load_team_characteristics_payload.clear()
    team_comparison._load_team_snapshot_history.clear()
    team_comparison._render_team_comparison_section(year=2027)

    radar_figure = calls["plotly_figures"][0]
    assert list(radar_figure.data[0].r[:3]) == [0.64, 0.67, 0.69]
    assert any(
        "latest snapshot `Bahrain Grand Prix FP1`" in caption for caption in calls["captions"]
    )


def test_render_team_comparison_section_handles_missing_profile_metrics(patcher, tmp_path):
    calls = _stub_streamlit_team(patcher)
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    data_path = tmp_path / "car_characteristics" / "2027_car_characteristics.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {"overall_performance": 0.82},
                    "Ferrari": {"overall_performance": 0.8},
                }
            }
        )
    )

    team_comparison._load_team_characteristics_payload.clear()
    team_comparison._load_team_snapshot_history.clear()
    team_comparison._render_team_comparison_section(year=2027)

    assert calls["plotly"] == 0
    assert calls["dataframe"] == 0
    assert any("No session profile metrics" in message for message in calls["info"])


def test_load_team_snapshot_history_sorts_sessions(patcher, tmp_path):
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    team_profiles = {
        "McLaren": {
            "balanced": {
                "slow_corner_performance": 0.71,
                "medium_corner_performance": 0.72,
                "fast_corner_performance": 0.73,
                "braking_performance": 0.70,
                "top_speed": 0.69,
                "tire_deg_performance": 0.74,
                "overall_pace": 0.75,
            }
        }
    }
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Bahrain Grand Prix",
        session_name="FP2",
        team_profiles=team_profiles,
        round_number=1,
        session_order=2,
    )
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Bahrain Grand Prix",
        session_name="FP1",
        team_profiles=team_profiles,
        round_number=1,
        session_order=1,
    )

    team_comparison._load_team_snapshot_history.clear()
    snapshots = team_comparison._load_team_snapshot_history(2027)

    assert [snapshot["session_name"] for snapshot in snapshots] == ["FP1", "FP2"]


def test_load_team_snapshot_history_orders_same_day_snapshots_by_full_timestamp(patcher, tmp_path):
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    team_profiles = {
        "McLaren": {
            "balanced": {
                "slow_corner_performance": 0.71,
                "medium_corner_performance": 0.72,
                "fast_corner_performance": 0.73,
                "braking_performance": 0.70,
                "top_speed": 0.69,
                "tire_deg_performance": 0.74,
                "overall_pace": 0.75,
            }
        }
    }
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Saudi Arabian Grand Prix",
        session_name="FP1",
        team_profiles=team_profiles,
        round_number=2,
        session_order=1,
        session_started_at="2027-03-07T01:00:00+00:00",
    )
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Bahrain Grand Prix",
        session_name="R",
        team_profiles=team_profiles,
        round_number=1,
        session_order=7,
        session_started_at="2027-03-07T05:00:00+00:00",
    )

    team_comparison._load_team_snapshot_history.clear()
    snapshots = team_comparison._load_team_snapshot_history(2027)

    assert [(snapshot["event_name"], snapshot["session_name"]) for snapshot in snapshots] == [
        ("Saudi Arabian Grand Prix", "FP1"),
        ("Bahrain Grand Prix", "R"),
    ]


def test_load_team_snapshot_history_cache_token_notices_new_snapshot_file(patcher, tmp_path):
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    team_profiles = {
        "McLaren": {
            "balanced": {
                "slow_corner_performance": 0.71,
                "medium_corner_performance": 0.72,
                "fast_corner_performance": 0.73,
                "braking_performance": 0.70,
                "top_speed": 0.69,
                "tire_deg_performance": 0.74,
                "overall_pace": 0.75,
            }
        }
    }
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Chinese Grand Prix",
        session_name="Q",
        team_profiles=team_profiles,
        round_number=2,
        session_order=6,
    )

    team_comparison._load_team_snapshot_history.clear()
    first_token = team_comparison._snapshot_history_cache_token(2027)
    first_snapshots = team_comparison._load_team_snapshot_history(2027, first_token)

    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Chinese Grand Prix",
        session_name="R",
        team_profiles=team_profiles,
        round_number=2,
        session_order=7,
    )

    second_token = team_comparison._snapshot_history_cache_token(2027)
    second_snapshots = team_comparison._load_team_snapshot_history(2027, second_token)

    assert first_token != second_token
    assert [snapshot["session_name"] for snapshot in first_snapshots] == ["Q"]
    assert [snapshot["session_name"] for snapshot in second_snapshots] == ["Q", "R"]


def test_snapshot_label_avoids_duplicate_testing_prefix():
    payload = {"event_name": "Testing 1", "session_name": "Testing 1 Day 2"}

    assert team_comparison._snapshot_label(payload) == "Testing 1 Day 2"


def test_build_snapshot_history_dataframe_and_summary():
    snapshots = [
        {
            "event_name": "Bahrain Grand Prix",
            "session_name": "FP1",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.60,
                            "medium_corner_performance": 0.61,
                            "fast_corner_performance": 0.62,
                            "braking_performance": 0.59,
                            "top_speed": 0.58,
                            "top_speed_kph": 310.0,
                            "tire_deg_performance": 0.63,
                            "overall_pace": 0.64,
                        }
                    }
                },
                "Ferrari": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.61,
                            "medium_corner_performance": 0.62,
                            "fast_corner_performance": 0.63,
                            "braking_performance": 0.60,
                            "top_speed": 0.22,
                            "top_speed_kph": 317.0,
                            "tire_deg_performance": 0.62,
                            "overall_pace": 0.63,
                        }
                    }
                },
            },
        },
        {
            "event_name": "Bahrain Grand Prix",
            "session_name": "FP2",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.70,
                            "medium_corner_performance": 0.71,
                            "fast_corner_performance": 0.72,
                            "braking_performance": 0.69,
                            "top_speed": 0.68,
                            "top_speed_kph": 312.0,
                            "tire_deg_performance": 0.73,
                            "overall_pace": 0.74,
                        }
                    }
                },
                "Ferrari": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.69,
                            "medium_corner_performance": 0.70,
                            "fast_corner_performance": 0.71,
                            "braking_performance": 0.68,
                            "top_speed": 0.27,
                            "top_speed_kph": 318.0,
                            "tire_deg_performance": 0.72,
                            "overall_pace": 0.73,
                        }
                    }
                },
            },
        },
    ]

    frame = team_comparison._build_snapshot_history_dataframe(
        snapshots=snapshots,
        selected_teams=["McLaren"],
        profile="balanced",
    )

    assert list(frame["Snapshot"]) == ["Bahrain Grand Prix FP1", "Bahrain Grand Prix FP2"]
    assert frame.iloc[0]["Top Speed"] == pytest.approx(0.4265, abs=1e-4)
    assert round(float(frame.iloc[0]["Overall"]), 3) == 0.579


def test_build_snapshot_history_dataframe_keeps_partial_overall_points():
    snapshots = [
        {
            "event_name": "Bahrain Grand Prix",
            "session_name": "FP1",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.60,
                            "medium_corner_performance": 0.61,
                            "fast_corner_performance": 0.62,
                            "braking_performance": 0.59,
                            "top_speed": 0.58,
                            "tire_deg_performance": 0.63,
                        }
                    }
                }
            },
        },
        {
            "event_name": "Bahrain Grand Prix",
            "session_name": "FP2",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.70,
                            "medium_corner_performance": 0.71,
                            "fast_corner_performance": 0.72,
                            "braking_performance": 0.69,
                            "top_speed": 0.68,
                        }
                    }
                }
            },
        },
    ]

    frame = team_comparison._build_snapshot_history_dataframe(
        snapshots=snapshots,
        selected_teams=["McLaren"],
        profile="balanced",
    )

    assert frame["Snapshot"].tolist() == ["Bahrain Grand Prix FP1", "Bahrain Grand Prix FP2"]
    assert frame["Metric Count"].tolist() == [6, 5]
    assert frame["Metric Coverage"].tolist() == [1.0, 5 / 6]
    assert frame["Overall"].round(3).tolist() == [0.605, 0.7]
    assert round(float(frame.iloc[1]["Slow Corners"]), 2) == 0.70


def test_latest_snapshot_payload_skips_sprint_only_sessions():
    snapshots = [
        {
            "event_name": "Chinese Grand Prix",
            "session_name": "FP1",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "overall_pace": 0.64,
                        }
                    }
                }
            },
        },
        {
            "event_name": "Chinese Grand Prix",
            "session_name": "SQ",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "overall_pace": 0.07,
                        }
                    }
                }
            },
        },
    ]

    latest = team_comparison._latest_snapshot_payload(snapshots)

    assert latest is not None
    assert latest["session_name"] == "FP1"


def test_build_snapshot_history_dataframe_includes_sprint_weekend_sessions():
    snapshots = [
        {
            "event_name": "Testing 1",
            "session_name": "Testing 1 Day 1",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.50,
                            "medium_corner_performance": 0.51,
                            "fast_corner_performance": 0.52,
                            "braking_performance": 0.49,
                            "top_speed": 0.48,
                            "tire_deg_performance": 0.53,
                        }
                    }
                }
            },
        },
        {
            "event_name": "Chinese Grand Prix",
            "session_name": "FP1",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.60,
                            "medium_corner_performance": 0.61,
                            "fast_corner_performance": 0.62,
                            "braking_performance": 0.59,
                            "top_speed": 0.58,
                            "tire_deg_performance": 0.63,
                        }
                    }
                }
            },
        },
        {
            "event_name": "Australian Grand Prix",
            "session_name": "Q",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.70,
                            "medium_corner_performance": 0.71,
                            "fast_corner_performance": 0.72,
                            "braking_performance": 0.69,
                            "top_speed": 0.68,
                            "tire_deg_performance": 0.73,
                        }
                    }
                }
            },
        },
        {
            "event_name": "Australian Grand Prix",
            "session_name": "R",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.80,
                            "medium_corner_performance": 0.81,
                            "fast_corner_performance": 0.82,
                            "braking_performance": 0.79,
                            "top_speed": 0.78,
                            "tire_deg_performance": 0.83,
                        }
                    }
                }
            },
        },
        {
            "event_name": "Chinese Grand Prix",
            "session_name": "SQ",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.10,
                            "medium_corner_performance": 0.11,
                            "fast_corner_performance": 0.12,
                            "braking_performance": 0.09,
                            "top_speed": 0.08,
                            "tire_deg_performance": 0.13,
                        }
                    }
                }
            },
        },
        {
            "event_name": "Chinese Grand Prix",
            "session_name": "Sprint",
            "teams": {
                "McLaren": {
                    "profiles": {
                        "balanced": {
                            "slow_corner_performance": 0.20,
                            "medium_corner_performance": 0.21,
                            "fast_corner_performance": 0.22,
                            "braking_performance": 0.19,
                            "top_speed": 0.18,
                            "tire_deg_performance": 0.23,
                        }
                    }
                }
            },
        },
    ]

    frame = team_comparison._build_snapshot_history_dataframe(
        snapshots=snapshots,
        selected_teams=["McLaren"],
        profile="balanced",
    )

    assert frame["Snapshot"].tolist() == [
        "Testing 1 Day 1",
        "Chinese Grand Prix FP1",
        "Australian Grand Prix Q",
        "Australian Grand Prix R",
        "Chinese Grand Prix SQ",
        "Chinese Grand Prix Sprint",
    ]


def test_render_team_comparison_section_renders_development_history(patcher, tmp_path):
    calls = _stub_streamlit_team(patcher)
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    data_path = tmp_path / "car_characteristics" / "2027_car_characteristics.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "overall_performance": 0.82,
                        "testing_characteristics_profiles": {
                            "balanced": {
                                "overall_pace": 0.78,
                                "slow_corner_performance": 0.74,
                                "medium_corner_performance": 0.77,
                                "fast_corner_performance": 0.79,
                                "braking_performance": 0.76,
                                "top_speed": 0.80,
                                "tire_deg_performance": 0.73,
                            }
                        },
                    }
                }
            }
        )
    )
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Bahrain Grand Prix",
        session_name="FP1",
        round_number=1,
        session_order=1,
        team_profiles={
            "McLaren": {
                "balanced": {
                    "overall_pace": 0.68,
                    "slow_corner_performance": 0.64,
                    "medium_corner_performance": 0.67,
                    "fast_corner_performance": 0.69,
                    "braking_performance": 0.66,
                    "top_speed": 0.70,
                    "tire_deg_performance": 0.63,
                }
            }
        },
    )

    team_comparison._load_team_characteristics_payload.clear()
    team_comparison._load_team_snapshot_history.clear()
    team_comparison._render_team_comparison_section(year=2027)

    assert calls["plotly"] == 2
    assert calls["dataframe"] == 1


def test_render_team_comparison_section_overall_hover_uses_real_metric_coverage(patcher, tmp_path):
    calls = _stub_streamlit_team(patcher)
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    data_path = tmp_path / "car_characteristics" / "2027_car_characteristics.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "overall_performance": 0.82,
                        "testing_characteristics_profiles": {
                            "balanced": {
                                "overall_pace": 0.78,
                                "slow_corner_performance": 0.74,
                                "medium_corner_performance": 0.77,
                                "fast_corner_performance": 0.79,
                                "braking_performance": 0.76,
                                "top_speed": 0.80,
                                "tire_deg_performance": 0.73,
                            }
                        },
                    }
                }
            }
        )
    )
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Bahrain Grand Prix",
        session_name="FP1",
        round_number=1,
        session_order=1,
        team_profiles={
            "McLaren": {
                "balanced": {
                    "slow_corner_performance": 0.60,
                    "medium_corner_performance": 0.61,
                    "fast_corner_performance": 0.62,
                    "braking_performance": 0.59,
                    "top_speed": 0.58,
                    "tire_deg_performance": 0.63,
                }
            }
        },
    )
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Bahrain Grand Prix",
        session_name="FP2",
        round_number=1,
        session_order=2,
        team_profiles={
            "McLaren": {
                "balanced": {
                    "slow_corner_performance": 0.70,
                    "medium_corner_performance": 0.71,
                    "fast_corner_performance": 0.72,
                    "braking_performance": 0.69,
                    "top_speed": 0.68,
                }
            }
        },
    )

    team_comparison._load_team_characteristics_payload.clear()
    team_comparison._load_team_snapshot_history.clear()
    team_comparison._render_team_comparison_section(year=2027)

    development_figure = calls["plotly_figures"][1]
    customdata = development_figure.data[0].customdata.tolist()
    assert customdata == [[6.0, 1.0], [5.0, 5 / 6]]


def test_render_team_comparison_section_orders_development_axis_chronologically(patcher, tmp_path):
    calls = _stub_streamlit_team(patcher)
    patcher.setattr(team_comparison.config_loader, "get", lambda key, default=None: str(tmp_path))

    data_path = tmp_path / "car_characteristics" / "2027_car_characteristics.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(
            {
                "teams": {
                    "McLaren": {
                        "overall_performance": 0.82,
                        "testing_characteristics_profiles": {
                            "balanced": {
                                "overall_pace": 0.78,
                                "slow_corner_performance": 0.74,
                                "medium_corner_performance": 0.77,
                                "fast_corner_performance": 0.79,
                                "braking_performance": 0.76,
                                "top_speed": 0.80,
                                "tire_deg_performance": 0.73,
                            }
                        },
                    },
                    "Ferrari": {
                        "overall_performance": 0.80,
                        "testing_characteristics_profiles": {
                            "balanced": {
                                "overall_pace": 0.75,
                                "slow_corner_performance": 0.72,
                                "medium_corner_performance": 0.74,
                                "fast_corner_performance": 0.76,
                                "braking_performance": 0.73,
                                "top_speed": 0.78,
                                "tire_deg_performance": 0.71,
                            }
                        },
                    },
                }
            }
        )
    )

    complete_profile = {
        "balanced": {
            "overall_pace": 0.68,
            "slow_corner_performance": 0.64,
            "medium_corner_performance": 0.67,
            "fast_corner_performance": 0.69,
            "braking_performance": 0.66,
            "top_speed": 0.70,
            "tire_deg_performance": 0.63,
        }
    }
    missing_metric_profile = {
        "balanced": {
            "overall_pace": 0.66,
            "slow_corner_performance": 0.62,
            "medium_corner_performance": 0.65,
            "fast_corner_performance": 0.67,
            "braking_performance": 0.64,
            "top_speed": 0.68,
        }
    }
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Australian Grand Prix",
        session_name="FP1",
        round_number=1,
        session_order=1,
        session_started_at="2027-03-06T01:30:00+00:00",
        team_profiles={
            "McLaren": complete_profile,
            "Ferrari": complete_profile,
        },
    )
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Australian Grand Prix",
        session_name="FP2",
        round_number=1,
        session_order=2,
        session_started_at="2027-03-06T05:00:00+00:00",
        team_profiles={
            "McLaren": missing_metric_profile,
            "Ferrari": complete_profile,
        },
    )
    _write_snapshot(
        tmp_path,
        year=2027,
        event_name="Australian Grand Prix",
        session_name="Q",
        round_number=1,
        session_order=6,
        session_started_at="2027-03-07T05:00:00+00:00",
        team_profiles={
            "McLaren": complete_profile,
            "Ferrari": complete_profile,
        },
    )

    team_comparison._load_team_characteristics_payload.clear()
    team_comparison._load_team_snapshot_history.clear()
    team_comparison._render_team_comparison_section(year=2027)

    development_figure = calls["plotly_figures"][1]
    assert list(development_figure.layout.xaxis.categoryarray) == [
        "Australian Grand Prix FP1",
        "Australian Grand Prix FP2",
        "Australian Grand Prix Q",
    ]
    assert list(development_figure.layout.yaxis.range) == [-0.02, 1.02]

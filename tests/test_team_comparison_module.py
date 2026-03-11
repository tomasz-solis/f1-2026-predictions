"""Focused tests for team-comparison helpers and rendering."""

from __future__ import annotations

import json
from pathlib import Path

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
                            "tire_deg_performance": 0.63,
                            "overall_pace": 0.64,
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
                            "tire_deg_performance": 0.73,
                            "overall_pace": 0.74,
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

    assert list(frame["Snapshot"]) == ["Bahrain Grand Prix FP1", "Bahrain Grand Prix FP2"]
    assert round(float(frame.iloc[0]["Overall"]), 3) == 0.605


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

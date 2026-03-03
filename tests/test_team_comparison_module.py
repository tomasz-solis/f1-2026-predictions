"""Focused tests for team-comparison helpers and rendering."""

from __future__ import annotations

import json

from src.dashboard import team_comparison


def _stub_streamlit_team(patcher):
    calls = {"info": [], "plotly": 0, "dataframe": 0, "captions": []}
    patcher.setattr(team_comparison.st, "subheader", lambda *_args, **_kwargs: None)
    patcher.setattr(team_comparison.st, "selectbox", lambda _label, options, **_kwargs: options[0])
    patcher.setattr(
        team_comparison.st,
        "multiselect",
        lambda _label, options, default=None, **_kwargs: default if default is not None else [],
    )
    patcher.setattr(team_comparison.st, "info", lambda message: calls["info"].append(str(message)))
    patcher.setattr(
        team_comparison.st, "plotly_chart", lambda *_args, **_kwargs: calls.__setitem__("plotly", 1)
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
    team_comparison._render_team_comparison_section(year=2027)

    assert calls["plotly"] == 0
    assert calls["dataframe"] == 0
    assert any("No testing/practice profile metrics" in message for message in calls["info"])

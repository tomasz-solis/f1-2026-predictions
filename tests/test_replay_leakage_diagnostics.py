"""Tests for replay and leakage diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.replay_leakage_diagnostics import (
    build_replay_leakage_diagnostics,
    evaluate_fully_wet_dry_update_invariant,
    format_replay_leakage_diagnostics_markdown,
    replay_leakage_artifact_key,
)


def _write_json(path: Path, payload: dict) -> Path:
    """Write JSON test data and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mapping_payload() -> dict:
    """Return a small frozen mapping artifact."""
    folds = []
    for session_kind in ("race", "qualifying"):
        for year, slope in ((2024, 0.9), (2025, 1.1)):
            folds.append(
                {
                    "session_kind": session_kind,
                    "holdout_year": year,
                    "n_rows": 4,
                    "prediction_slope": slope,
                    "r_squared": 0.8,
                    "rmse_s": 0.2,
                }
            )
    return {
        "policy": "same_session_construct",
        "validation": {"folds": folds},
        "mappings": {
            "race": {
                "session_kind": "race",
                "policy": "same_session_construct",
                "intercept_s": 0.0,
                "slope_s_per_unit": 2.0,
                "training_years": [2022, 2023, 2024, 2025],
            },
            "qualifying": {
                "session_kind": "qualifying",
                "policy": "same_session_construct",
                "intercept_s": 0.0,
                "slope_s_per_unit": 2.0,
                "training_years": [2022, 2023, 2024, 2025],
            },
        },
    }


def test_replay_leakage_reports_legacy_dry_leakage_proxy(tmp_path):
    """Legacy driver state should be measured as a proxy, not mislabeled as seconds."""
    mapping_path = _write_json(tmp_path / "mapping.json", _mapping_payload())
    candidate_path = _write_json(
        tmp_path / "candidate.json",
        {
            "policy_evaluations": {
                "same_session_construct": {
                    "per_driver_residual_means": [
                        {
                            "session_kind": "race",
                            "driver_code": "D1",
                            "residual_mean_s": 0.5,
                            "n_rows": 4,
                        }
                    ]
                }
            }
        },
    )
    prior_path = _write_json(tmp_path / "prior.json", {"race_network": {}, "quali_network": {}})
    baseline_driver = _write_json(
        tmp_path / "baseline_driver.json",
        {
            "drivers": {
                code: {"bayesian": {"rating_mu": 10.0}, "wet_skill": 0.7}
                for code in ("D1", "D2", "D3", "D4")
            }
        },
    )
    current_driver = _write_json(
        tmp_path / "current_driver.json",
        {
            "drivers": {
                "D1": {"bayesian": {"rating_mu": 12.0}, "wet_skill": 0.7},
                "D2": {"bayesian": {"rating_mu": 12.0}, "wet_skill": 0.7},
                "D3": {"bayesian": {"rating_mu": 8.0}, "wet_skill": 0.7},
                "D4": {"bayesian": {"rating_mu": 8.0}, "wet_skill": 0.7},
            }
        },
    )
    current_car = _write_json(
        tmp_path / "current_car.json",
        {
            "races_completed": 4,
            "teams": {
                "Team A": {
                    "preseason_overall_performance": 0.5,
                    "overall_performance": 1.0,
                },
                "Team B": {
                    "preseason_overall_performance": 0.5,
                    "overall_performance": 0.0,
                },
            },
        },
    )
    lineup = _write_json(
        tmp_path / "lineups.json",
        {"current_lineups": {"Team A": ["D1", "D2"], "Team B": ["D3", "D4"]}},
    )

    artifact = build_replay_leakage_diagnostics(
        year=2026,
        replay_root=tmp_path / "missing_replay",
        mapping_artifact_path=mapping_path,
        candidate_diagnostics_path=candidate_path,
        prior_artifact_path=prior_path,
        baseline_driver_path=baseline_driver,
        current_driver_path=current_driver,
        current_car_path=current_car,
        lineup_path=lineup,
    )

    assert artifact["dry_leakage"]["state"] == "measured_legacy_proxy"
    assert artifact["dry_leakage"]["exact_metric_state"] == "blocked_missing_race_seconds_state"
    assert artifact["dry_leakage"]["correlation"] == 1.0
    assert artifact["historical_scale_reference"]["residual_outliers"][0]["driver_code"] == "D1"
    assert artifact["regulation_reset_monitoring"]["state"] == "not_available"
    assert artifact["status"] == "provisional_with_warnings"
    assert not any("Dry leakage" in warning for warning in artifact["warnings"])
    assert any("legacy proxy" in limitation for limitation in artifact["limitations"])
    assert any("no wet weather-routed rows" in limitation for limitation in artifact["limitations"])


def test_replay_leakage_uses_exact_dry_seconds_when_baseline_is_migrated(tmp_path):
    """Dry leakage should stop using rating-mu once both driver files have seconds."""
    mapping_path = _write_json(tmp_path / "mapping.json", _mapping_payload())
    candidate_path = _write_json(tmp_path / "candidate.json", {"policy_evaluations": {}})
    prior_path = _write_json(tmp_path / "prior.json", {"race_network": {}, "quali_network": {}})
    baseline_driver = _write_json(
        tmp_path / "baseline_driver.json",
        {
            "drivers": {
                code: {
                    "bayesian": {"rating_mu": 10.0, "race_rating_mu_s": 0.0},
                    "wet_skill": 0.7,
                }
                for code in ("D1", "D2", "D3", "D4")
            }
        },
    )
    current_driver = _write_json(
        tmp_path / "current_driver.json",
        {
            "drivers": {
                "D1": {"bayesian": {"rating_mu": 99.0, "race_rating_mu_s": 0.5}},
                "D2": {"bayesian": {"rating_mu": 99.0, "race_rating_mu_s": 0.5}},
                "D3": {"bayesian": {"rating_mu": -99.0, "race_rating_mu_s": -0.5}},
                "D4": {"bayesian": {"rating_mu": -99.0, "race_rating_mu_s": -0.5}},
            }
        },
    )
    current_car = _write_json(
        tmp_path / "current_car.json",
        {
            "teams": {
                "Team A": {
                    "preseason_overall_performance": 0.5,
                    "overall_performance": 0.75,
                },
                "Team B": {
                    "preseason_overall_performance": 0.5,
                    "overall_performance": 0.25,
                },
            }
        },
    )
    lineup = _write_json(
        tmp_path / "lineups.json",
        {"current_lineups": {"Team A": ["D1", "D2"], "Team B": ["D3", "D4"]}},
    )

    artifact = build_replay_leakage_diagnostics(
        year=2026,
        replay_root=tmp_path / "missing_replay",
        mapping_artifact_path=mapping_path,
        candidate_diagnostics_path=candidate_path,
        prior_artifact_path=prior_path,
        baseline_driver_path=baseline_driver,
        current_driver_path=current_driver,
        current_car_path=current_car,
        lineup_path=lineup,
    )

    dry = artifact["dry_leakage"]
    assert dry["state"] == "measured_seconds"
    assert dry["exact_metric_state"] == "measured"
    assert dry["driver_field"] == "bayesian.race_rating_mu_s"
    assert dry["correlation"] == 1.0
    assert dry["rows"][0]["delta_race_rating_mu_s"] == 0.5
    assert not any("rating-mu" in limitation for limitation in artifact["limitations"])


def test_replay_leakage_measures_regulation_reset_observations(tmp_path):
    """A supplied 2026 observation file should produce transfer metrics."""
    mapping_path = _write_json(tmp_path / "mapping.json", _mapping_payload())
    candidate_path = _write_json(tmp_path / "candidate.json", {"policy_evaluations": {}})
    prior_path = _write_json(tmp_path / "prior.json", {"race_network": {}, "quali_network": {}})
    baseline_driver = _write_json(
        tmp_path / "baseline_driver.json",
        {"drivers": {"D1": {"bayesian": {"rating_mu": 10.0}, "wet_skill": 0.7}}},
    )
    current_driver = _write_json(
        tmp_path / "current_driver.json",
        {"drivers": {"D1": {"bayesian": {"rating_mu": 10.0}, "wet_skill": 0.7}}},
    )
    current_car = _write_json(
        tmp_path / "current_car.json",
        {
            "races_completed": 1,
            "teams": {
                "Team A": {
                    "preseason_overall_performance": 0.5,
                    "overall_performance": 0.5,
                }
            },
        },
    )
    lineup = _write_json(tmp_path / "lineups.json", {"current_lineups": {"Team A": ["D1"]}})
    observations_path = tmp_path / "observations.csv"
    pd.DataFrame(
        [
            {
                "year": 2026,
                "race_name": "Race 1",
                "session_name": "Race",
                "session_kind": "race",
                "team": "Team A",
                "driver_code": "D1",
                "observed_driver_to_field_s": -1.0,
                "driver_rating_mu_s": 0.0,
                "team_strength_same_session": 0.0,
            },
            {
                "year": 2026,
                "race_name": "Race 1",
                "session_name": "Race",
                "session_kind": "race",
                "team": "Team B",
                "driver_code": "D2",
                "observed_driver_to_field_s": 1.0,
                "driver_rating_mu_s": 0.0,
                "team_strength_same_session": 1.0,
            },
            {
                "year": 2026,
                "race_name": "Race 1",
                "session_name": "Qualifying",
                "session_kind": "qualifying",
                "team": "Team A",
                "driver_code": "D1",
                "observed_driver_to_field_s": -1.0,
                "driver_rating_mu_s": 0.0,
                "team_strength_same_session": 0.0,
            },
            {
                "year": 2026,
                "race_name": "Race 1",
                "session_name": "Qualifying",
                "session_kind": "qualifying",
                "team": "Team B",
                "driver_code": "D2",
                "observed_driver_to_field_s": 1.0,
                "driver_rating_mu_s": 0.0,
                "team_strength_same_session": 1.0,
            },
        ]
    ).to_csv(observations_path, index=False)

    artifact = build_replay_leakage_diagnostics(
        year=2026,
        replay_root=tmp_path / "missing_replay",
        mapping_artifact_path=mapping_path,
        candidate_diagnostics_path=candidate_path,
        prior_artifact_path=prior_path,
        regulation_reset_observations_path=observations_path,
        baseline_driver_path=baseline_driver,
        current_driver_path=current_driver,
        current_car_path=current_car,
        lineup_path=lineup,
    )

    race_metrics = artifact["regulation_reset_monitoring"]["metrics_by_session_kind"]["race"]
    quali_metrics = artifact["regulation_reset_monitoring"]["metrics_by_session_kind"]["qualifying"]
    assert race_metrics["state"] == "measured"
    assert race_metrics["prediction_slope"] == 1.0
    assert race_metrics["rmse_s"] == 0.0
    assert quali_metrics["r_squared"] == 1.0


def test_regulation_reset_band_drift_is_a_monitoring_note(tmp_path):
    """A reset-year scale shift should remain visible without raising a warning."""
    mapping_payload = _mapping_payload()
    for fold in mapping_payload["validation"]["folds"]:
        fold["prediction_slope"] = 0.1 if fold["holdout_year"] == 2024 else 0.2
    mapping_path = _write_json(tmp_path / "mapping.json", mapping_payload)
    candidate_path = _write_json(tmp_path / "candidate.json", {"policy_evaluations": {}})
    prior_path = _write_json(tmp_path / "prior.json", {"race_network": {}, "quali_network": {}})
    baseline_driver = _write_json(
        tmp_path / "baseline_driver.json",
        {"drivers": {"D1": {"bayesian": {"rating_mu": 10.0}, "wet_skill": 0.7}}},
    )
    current_driver = _write_json(
        tmp_path / "current_driver.json",
        {"drivers": {"D1": {"bayesian": {"rating_mu": 10.0}, "wet_skill": 0.7}}},
    )
    current_car = _write_json(
        tmp_path / "current_car.json",
        {
            "races_completed": 1,
            "teams": {
                "Team A": {
                    "preseason_overall_performance": 0.5,
                    "overall_performance": 0.5,
                }
            },
        },
    )
    lineup = _write_json(tmp_path / "lineups.json", {"current_lineups": {"Team A": ["D1"]}})
    observations_path = tmp_path / "observations.csv"
    pd.DataFrame(
        [
            {
                "year": 2026,
                "race_name": "Race 1",
                "session_name": "Race",
                "session_kind": session_kind,
                "team": team,
                "driver_code": driver_code,
                "observed_driver_to_field_s": observed,
                "driver_rating_mu_s": 0.0,
                "team_strength_same_session": strength,
            }
            for session_kind in ("race", "qualifying")
            for team, driver_code, strength, observed in (
                ("Team A", "D1", 0.0, -1.0),
                ("Team B", "D2", 1.0, 1.0),
            )
        ]
    ).to_csv(observations_path, index=False)

    artifact = build_replay_leakage_diagnostics(
        year=2026,
        replay_root=tmp_path / "missing_replay",
        mapping_artifact_path=mapping_path,
        candidate_diagnostics_path=candidate_path,
        prior_artifact_path=prior_path,
        regulation_reset_observations_path=observations_path,
        baseline_driver_path=baseline_driver,
        current_driver_path=current_driver,
        current_car_path=current_car,
        lineup_path=lineup,
    )

    race_metrics = artifact["regulation_reset_monitoring"]["metrics_by_session_kind"]["race"]
    assert race_metrics["outside_historical_one_se_band"] is True
    assert not any("one-SE band" in warning for warning in artifact["warnings"])
    assert any("Reset-year scale differs" in note for note in artifact["monitoring_notes"])


def test_replay_leakage_artifact_key_and_markdown_are_stable():
    """Artifact routing and Markdown formatting should stay deterministic."""
    assert replay_leakage_artifact_key(2026) == "2026::replay_leakage_diagnostics"
    markdown = format_replay_leakage_diagnostics_markdown(
        {
            "built_at": "2026-05-20T00:00:00+00:00",
            "model_version": "2.1",
            "status": "measured",
            "source_state": {"replay_race_count": 4, "live_artifact_races_completed": 4},
            "historical_scale_reference": {"reference_band_2024_2025": {}},
            "regulation_reset_monitoring": {"metrics_by_session_kind": {}},
            "dry_leakage": {"state": "measured_legacy_proxy", "correlation": 0.1},
            "wet_leakage": {"state": "not_evaluable", "fully_wet_dry_update_invariant": {}},
            "warnings": [],
        }
    )
    assert "# Replay And Leakage Diagnostics" in markdown
    assert "Dry Leakage" in markdown


def test_fully_wet_invariant_accepts_zero_dry_trace_delta() -> None:
    """Fully wet trace rows pass when dry update flags and deltas stay zero."""
    invariant = evaluate_fully_wet_dry_update_invariant(
        [
            {
                "event_name": "Wet Grand Prix",
                "session_name": "Race",
                "session_kind": "race",
                "weather_route": "rain",
                "driver_code": "ANT",
                "dry_race_update_applied": False,
                "legacy_rating_mu_delta": 0.0,
                "race_rating_mu_s_delta": 0.0,
            },
            {
                "event_name": "Wet Grand Prix",
                "session_name": "Qualifying",
                "session_kind": "qualifying",
                "weather_route": "rain",
                "driver_code": "ANT",
                "dry_quali_update_applied": False,
                "legacy_rating_mu_delta": 0.0,
                "quali_rating_mu_s_delta": 0.0,
            },
        ]
    )

    assert invariant["state"] == "passed_from_update_trace"
    assert invariant["fully_wet_trace_rows"] == 2
    assert invariant["violations"] == []


def test_fully_wet_invariant_reports_dry_trace_movement() -> None:
    """Fully wet trace failures should name the driver and moved dry field."""
    invariant = evaluate_fully_wet_dry_update_invariant(
        [
            {
                "event_name": "Wet Grand Prix",
                "session_name": "Race",
                "session_kind": "race",
                "weather_route": "rain",
                "driver_code": "ANT",
                "dry_race_update_applied": True,
                "legacy_rating_mu_delta": 0.2,
                "race_rating_mu_s_delta": 0.1,
            }
        ]
    )

    assert invariant["state"] == "failed"
    assert invariant["violations"] == [
        {
            "event_name": "Wet Grand Prix",
            "session_name": "Race",
            "session_kind": "race",
            "driver_code": "ANT",
            "dry_update_flag": "dry_race_update_applied",
            "moved_fields": ["legacy_rating_mu_delta", "race_rating_mu_s_delta"],
        }
    ]

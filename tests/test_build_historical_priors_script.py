"""Tests for the historical preseason-prior builder script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_builder_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_historical_priors.py"
    spec = importlib.util.spec_from_file_location("build_historical_priors_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_history_years_uses_previous_completed_seasons():
    module = _load_builder_module()

    assert module.default_history_years(2025, 3) == [2022, 2023, 2024]


def test_build_team_prior_payload_resets_to_preseason_schema():
    module = _load_builder_module()
    payload = module.build_team_prior_payload(
        target_year=2025,
        source_year=2024,
        generated_at="2026-04-21T00:00:00+00:00",
        raw_team_payload={
            "McLaren": {
                "overall_performance": 0.82,
                "uncertainty": 0.14,
                "races_analyzed": 24,
                "championship_position": 1,
                "normalized_overall_pace": 1.0,
                "normalized_top_speed": 0.9,
                "normalized_slow_corner_performance": 0.8,
                "normalized_medium_corner_performance": 0.85,
                "normalized_fast_corner_performance": 0.88,
                "normalized_consistency": 0.82,
                "normalized_tire_deg_performance": 0.76,
            },
            "Sauber": {
                "overall_performance": 0.75,
                "uncertainty": 0.22,
                "races_analyzed": 24,
                "championship_position": 10,
                "normalized_overall_pace": 0.0,
                "normalized_top_speed": 0.2,
                "normalized_slow_corner_performance": 0.1,
                "normalized_medium_corner_performance": 0.1,
                "normalized_fast_corner_performance": 0.1,
                "normalized_consistency": 0.3,
                "normalized_tire_deg_performance": 0.25,
            },
        },
    )

    assert payload["year"] == 2025
    assert payload["data_freshness"] == "BASELINE_PRESEASON"
    assert payload["races_completed"] == 0
    assert payload["teams"]["McLaren"]["current_season_performance"] == []
    assert payload["teams"]["McLaren"]["races_completed"] == 0
    assert "races_analyzed" not in payload["teams"]["McLaren"]
    assert "championship_position" not in payload["teams"]["McLaren"]
    assert (
        payload["teams"]["McLaren"]["overall_performance"]
        > payload["teams"]["Audi"]["overall_performance"]
    )
    assert payload["teams"]["McLaren"]["testing_characteristics"]["run_profile"] == "balanced"
    assert "short_run" in payload["teams"]["McLaren"]["testing_characteristics_profiles"]
    assert "directionality" in payload["teams"]["McLaren"]


def test_build_driver_extraction_command_includes_explicit_lineup_seed(tmp_path):
    module = _load_builder_module()

    command = module.build_driver_extraction_command(
        source_years=[2022, 2023, 2024],
        output_path=tmp_path / "2025_driver_characteristics.json",
        lineup_file=tmp_path / "2025_lineups.json",
        request_delay=0.25,
        max_attempts=4,
        timeout_budget_seconds=90.0,
    )

    assert "--lineup-file" in command
    assert str(tmp_path / "2025_lineups.json") in command
    assert command[0]


def test_rewrite_driver_prior_payload_strips_legacy_bayesian_fields():
    module = _load_builder_module()
    payload = module.rewrite_driver_prior_payload(
        {
            "drivers": {
                "NOR": {
                    "racecraft": {"skill_score": 0.82, "overtaking_skill": 0.80},
                    "pace": {"quali_pace": 0.83, "race_pace": 0.81},
                    "dnf_risk": {"dnf_rate": 0.04},
                    "bayesian": {
                        "rating_mu": 17.4,
                        "rating_sigma": 2.5,
                        "normalized_skill_score": 0.82,
                        "sessions_observed": 0,
                        "seeded_from": "extraction_prior",
                    },
                }
            }
        },
        target_year=2025,
        source_years=[2022, 2023, 2024],
        lineup_payload={"current_lineups": {"McLaren": ["NOR", "PIA"]}},
    )

    assert payload["year"] == 2025
    assert "normalized_skill_score" not in payload["drivers"]["NOR"]["bayesian"]
    assert payload["drivers"]["NOR"]["bayesian"]["season_year"] == 2025


def test_rewrite_driver_prior_payload_repairs_team_based_rookie_metadata():
    module = _load_builder_module()
    payload = module.rewrite_driver_prior_payload(
        {
            "drivers": {
                "BOR": {
                    "racecraft": {"skill_score": 0.714, "overtaking_skill": 0.714},
                    "pace": {"quali_pace": 0.678, "race_pace": 0.714},
                    "experience": {
                        "years_of_experience": 0,
                        "debut_year": 2026,
                        "total_races": 0,
                        "tier": "rookie",
                    },
                    "dnf_risk": {"dnf_rate": 0.03},
                    "prior_source": "team_based_prior",
                },
                "HUL": {
                    "racecraft": {"skill_score": 0.794, "overtaking_skill": 0.672},
                    "pace": {"quali_pace": 0.747, "race_pace": 0.844},
                    "experience": {
                        "years_of_experience": 14,
                        "debut_year": 2010,
                        "total_races": 48,
                        "tier": "veteran",
                    },
                    "dnf_risk": {"dnf_rate": 0.104},
                },
            }
        },
        target_year=2025,
        source_years=[2022, 2023, 2024],
        lineup_payload={"current_lineups": {"Audi": ["HUL", "BOR"]}},
    )

    bor = payload["drivers"]["BOR"]
    hul = payload["drivers"]["HUL"]

    assert bor["experience"]["debut_year"] == 2025
    assert bor["racecraft"]["skill_score"] <= 0.62
    assert bor["bayesian"]["season_year"] == 2025
    assert hul["racecraft"]["skill_score"] < 0.794


def test_main_builds_team_track_and_driver_outputs_with_patched_dependencies(tmp_path, monkeypatch):
    module = _load_builder_module()
    output_dir = tmp_path / "processed"

    monkeypatch.setattr(
        module,
        "calculate_team_performance_from_races",
        lambda year: {
            "McLaren": {
                "overall_performance": 0.81,
                "uncertainty": 0.12,
                "races_analyzed": 24,
            }
        },
    )
    monkeypatch.setattr(module, "rank_teams_by_performance", lambda payload: payload)

    def _fake_track_builder(source_years, temp_dir):
        track_path = temp_dir / "track_characteristics" / "2026_track_characteristics.json"
        track_path.parent.mkdir(parents=True, exist_ok=True)
        track_path.write_text(
            json.dumps(
                {
                    "year": 2026,
                    "data_freshness": "BASELINE_PRESEASON",
                    "tracks": {
                        "Bahrain Grand Prix": {
                            "pit_stop_loss": 22.0,
                            "safety_car_prob": 0.35,
                            "overtaking_difficulty": 0.45,
                        }
                    },
                }
            )
        )

    monkeypatch.setattr(module, "calculate_track_characteristics", _fake_track_builder)
    monkeypatch.setattr(
        module,
        "resolve_lineup_seed_payload",
        lambda target_year, lineup_file=None: {
            "season": target_year,
            "current_lineups": {"McLaren": ["NOR", "PIA"]},
        },
    )

    def _fake_subprocess_run(command, check, cwd):
        assert check is True
        assert cwd == module.PROJECT_ROOT
        output_index = command.index("--output") + 1
        output_path = Path(command[output_index])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "years": [2022, 2023, 2024],
                    "drivers": {
                        "NOR": {
                            "racecraft": {"skill_score": 0.82},
                            "pace": {"quali_pace": 0.83, "race_pace": 0.81},
                            "dnf_risk": {"dnf_rate": 0.04},
                        }
                    },
                }
            )
        )

    monkeypatch.setattr(module.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(module, "validate_team_characteristics", lambda payload, **kwargs: None)
    monkeypatch.setattr(module, "validate_track_characteristics", lambda payload, **kwargs: None)
    monkeypatch.setattr(module, "validate_driver_characteristics", lambda payload, **kwargs: None)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "build_historical_priors.py",
            "--year",
            "2025",
            "--data-dir",
            str(output_dir),
            "--team-source-year",
            "2024",
            "--driver-years",
            "2022,2023,2024",
            "--track-years",
            "2022,2023,2024",
        ],
    )

    exit_code = module.main()

    assert exit_code == 0

    team_payload = json.loads(
        (output_dir / "car_characteristics" / "2025_car_characteristics.json").read_text()
    )
    track_payload = json.loads(
        (output_dir / "track_characteristics" / "2025_track_characteristics.json").read_text()
    )
    driver_payload = json.loads(
        (output_dir / "driver_characteristics" / "2025_driver_characteristics.json").read_text()
    )

    assert team_payload["year"] == 2025
    assert track_payload["year"] == 2025
    assert driver_payload["year"] == 2025
    assert team_payload["teams"]["McLaren"]["current_season_performance"] == []
    assert track_payload["note"].startswith("Preseason 2025 prior built from")
    assert driver_payload["note"].startswith("Preseason 2025 prior built from")

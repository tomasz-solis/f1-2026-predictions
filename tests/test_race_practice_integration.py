from __future__ import annotations

from typing import Any

from src.predictors.baseline.race.race_simulation import (
    _apply_race_practice_challenger,
)


class _Config:
    def __init__(self, variant: str) -> None:
        self.variant = variant

    def get(self, key: str, default: Any = None) -> Any:
        if key == "baseline_predictor.model_variant":
            return self.variant
        return default


def _driver_info() -> dict[str, dict[str, Any]]:
    return {
        "AAA": {
            "team": "Team A",
            "team_strength_by_compound": {"SOFT": 0.50},
            "tire_deg_by_compound": {"SOFT": 0.20},
        }
    }


def _evidence() -> dict[str, Any]:
    return {
        "artifact_type": "race_practice_evidence",
        "schema_version": 1,
        "weather": "dry",
        "year": 2026,
        "event_name": "Example Grand Prix",
        "teams": {
            "Team A": {
                "compounds": {
                    "SOFT": {
                        "n_laps": 16,
                        "n_stints": 2,
                        "pace_performance": 0.90,
                        "pace_comparison_status": "matched",
                        "matched_pace_buckets": 1,
                        "matched_pace_laps": 16,
                        "matched_pace_stints": 2,
                        "tire_deg_slope_s_per_lap": 0.10,
                    }
                }
            }
        },
    }


def test_champion_ignores_supplied_race_practice_evidence() -> None:
    info = _driver_info()

    diagnostics = _apply_race_practice_challenger(
        info,
        _evidence(),
        weather="dry",
        year=2026,
        race_name="Example Grand Prix",
        cfg=_Config("champion"),
    )

    assert diagnostics == {
        "requested": False,
        "applied": False,
        "fallback_reason": None,
        "drivers_applied": 0,
    }
    assert info["AAA"]["team_strength_by_compound"]["SOFT"] == 0.50


def test_r0_applies_matching_compound_evidence() -> None:
    info = _driver_info()

    diagnostics = _apply_race_practice_challenger(
        info,
        _evidence(),
        weather="dry",
        year=2026,
        race_name="Example Grand Prix",
        cfg=_Config("r0_long_run"),
    )

    assert diagnostics["applied"] is True
    assert diagnostics["drivers_applied"] == 1
    assert info["AAA"]["team_strength_by_compound"]["SOFT"] > 0.50
    assert 0.10 < info["AAA"]["tire_deg_by_compound"]["SOFT"] < 0.20


def test_r0_fails_closed_for_mismatched_weekend() -> None:
    info = _driver_info()

    diagnostics = _apply_race_practice_challenger(
        info,
        _evidence(),
        weather="dry",
        year=2026,
        race_name="Different Grand Prix",
        cfg=_Config("r0_long_run"),
    )

    assert diagnostics["applied"] is False
    assert diagnostics["fallback_reason"] == "evidence_event_mismatch"
    assert info["AAA"]["team_strength_by_compound"]["SOFT"] == 0.50


def test_r0_fails_closed_when_matching_evidence_is_not_robust() -> None:
    info = _driver_info()
    evidence = _evidence()
    evidence["teams"]["Team A"]["compounds"]["SOFT"]["n_stints"] = 1

    diagnostics = _apply_race_practice_challenger(
        info,
        evidence,
        weather="dry",
        year=2026,
        race_name="Example Grand Prix",
        cfg=_Config("r0_long_run"),
    )

    assert diagnostics["applied"] is False
    assert diagnostics["fallback_reason"] == "insufficient_field_evidence_coverage"
    assert info["AAA"]["team_strength_by_compound"]["SOFT"] == 0.50

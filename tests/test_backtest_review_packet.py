"""Tests for reviewer-facing backtest packet rendering."""

from __future__ import annotations

from pathlib import Path

from scripts import backtest_2025_season
from scripts.backtest_2025_season import (
    _emit_review_packet_markdown,
    _inspect_season_prior_status,
    _prepare_backtest_data_dir,
    _resolve_effective_season_prior_mode,
)


def test_build_backtest_predictor_uses_requested_season_year(monkeypatch) -> None:
    """Historical backtests should build the predictor with the target season year."""
    captured: dict[str, object] = {}

    class _PredictorStub:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(backtest_2025_season, "Baseline2026Predictor", _PredictorStub)

    predictor = backtest_2025_season._build_backtest_predictor(
        season_year=2025,
        seed=17,
        merged_config={"learning": {"interval_max_adjustment": 6.0}},
        artifact_store=object(),
    )

    assert predictor is not None
    assert captured["season_year"] == 2025
    assert captured["seed"] == 17


def test_resolve_effective_season_prior_mode_defaults_to_proxy_only_for_historical() -> None:
    """Historical replay should use proxy-only priors unless we opt in."""
    assert (
        _resolve_effective_season_prior_mode(
            requested_mode="auto",
            evaluation_mode="historical",
        )
        == "proxy-only"
    )
    assert (
        _resolve_effective_season_prior_mode(
            requested_mode="auto",
            evaluation_mode="live",
        )
        == "allow"
    )
    assert (
        _resolve_effective_season_prior_mode(
            requested_mode="allow",
            evaluation_mode="historical",
        )
        == "allow"
    )


def test_prepare_backtest_data_dir_proxy_only_removes_target_year_priors(tmp_path: Path) -> None:
    """Proxy-only backtests should strip season-scoped priors from the copied data root."""
    source_data_dir = tmp_path / "source" / "processed"
    (source_data_dir / "car_characteristics").mkdir(parents=True, exist_ok=True)
    (source_data_dir / "driver_characteristics").mkdir(parents=True, exist_ok=True)
    (source_data_dir / "track_characteristics").mkdir(parents=True, exist_ok=True)
    (source_data_dir / "car_characteristics" / "2025_car_characteristics.json").write_text("{}")
    (source_data_dir / "driver_characteristics" / "2025_driver_characteristics.json").write_text(
        "{}"
    )
    (source_data_dir / "track_characteristics" / "2025_track_characteristics.json").write_text("{}")
    (source_data_dir / "car_characteristics" / "2026_car_characteristics.json").write_text("{}")

    prepared = _prepare_backtest_data_dir(
        source_data_dir=source_data_dir,
        output_dir=tmp_path / "outputs",
        season_year=2025,
        season_prior_mode="proxy-only",
    )

    assert prepared != source_data_dir
    assert not (prepared / "car_characteristics" / "2025_car_characteristics.json").exists()
    assert not (prepared / "driver_characteristics" / "2025_driver_characteristics.json").exists()
    assert not (prepared / "track_characteristics" / "2025_track_characteristics.json").exists()
    assert (prepared / "car_characteristics" / "2026_car_characteristics.json").exists()


def test_emit_review_packet_markdown_includes_new_review_sections(tmp_path: Path) -> None:
    """Review packet markdown should surface segments, error analysis, and takeaways."""
    output_path = tmp_path / "REVIEW_PACKET.md"
    packet = {
        "season": 2025,
        "evaluation_mode": "historical",
        "weather": "dry",
        "canonical_model": {
            "name": "baseline_adaptive",
            "summary": {
                "race_mae_mean": 3.25,
                "qualifying_mae_mean": 2.10,
                "top3_accuracy_mean": 61.5,
                "winner_accuracy_percent": 33.3,
                "qualifying_interval_races": 4,
                "qualifying_interval_count": 80,
                "qualifying_interval_empirical_coverage": 0.875,
                "qualifying_interval_nominal_coverage": 0.90,
                "qualifying_interval_calibration_error": -0.025,
                "qualifying_interval_width_mean": 6.2,
                "race_interval_races": 4,
                "race_interval_count": 80,
                "race_interval_empirical_coverage": 0.925,
                "race_interval_nominal_coverage": 0.90,
                "race_interval_calibration_error": 0.025,
                "race_interval_width_mean": 7.4,
            },
            "generalization": {"generalization_gap_race_mae": 0.22},
        },
        "static_baseline": {
            "name": "baseline_static",
            "summary": {"race_mae_mean": 3.60},
        },
        "overlap_comparison": {"race_mae_improvement": 0.45},
        "adaptive_vs_static_comparison": {
            "race_mae_improvement": 0.35,
            "top3_accuracy_delta": 4.0,
        },
        "canonical_segment_breakdown": {
            "track_type": {
                "permanent": {
                    "events": 4,
                    "race_mae_mean": 2.8,
                    "top3_accuracy_mean": 66.0,
                    "winner_accuracy_percent": 40.0,
                },
                "street": {
                    "events": 2,
                    "race_mae_mean": 4.6,
                    "top3_accuracy_mean": 45.0,
                    "winner_accuracy_percent": 0.0,
                },
            }
        },
        "canonical_error_analysis": {
            "worst_race_events": [
                {
                    "race_name": "Monaco Grand Prix",
                    "track_type": "street",
                    "weekend_format": "normal",
                    "weather": "dry",
                    "race_mae": 5.8,
                }
            ],
            "winner_miss_events": [
                {
                    "race_name": "Monaco Grand Prix",
                    "race_mae": 5.8,
                    "top3_accuracy": 33.0,
                }
            ],
        },
        "reviewer_takeaways": [
            "Adaptive learning improved race MAE versus the static baseline.",
            "Street circuits remain the weakest slice.",
        ],
        "season_prior_status": {
            "team": {
                "path": "data/processed/car_characteristics/2025_car_characteristics.json",
                "season_scoped": True,
                "fallback_path": None,
            },
            "driver": {
                "path": "data/processed/driver_characteristics/2025_driver_characteristics.json",
                "season_scoped": False,
                "fallback_path": "data/processed/driver_characteristics.json",
            },
            "track": {
                "path": "data/processed/track_characteristics/2025_track_characteristics.json",
                "season_scoped": True,
                "fallback_path": None,
            },
        },
        "season_prior_mode": "proxy-only",
        "season_prior_source_data_dir": "data/processed",
        "season_prior_data_dir": "reports/backtest_2025/_backtest_inputs/2025_proxy_only/processed",
        "recommended_experiments": [
            {
                "name": "higher_grid_anchor",
                "test_race_mae": 3.10,
                "test_race_mae_improvement_vs_baseline": 0.15,
                "generalization_gap_race_mae": 0.12,
            }
        ],
    }

    _emit_review_packet_markdown(output_path, packet)
    markdown = output_path.read_text()

    assert "## Plain-Language Takeaways" in markdown
    assert "## Prior Provenance" in markdown
    assert "- Mode: `proxy-only`" in markdown
    assert "## Interval Calibration" in markdown
    assert "## Segment Breakdown" in markdown
    assert "## Error Analysis" in markdown
    assert "## Recommended Ablations" in markdown
    assert "Monaco Grand Prix" in markdown
    assert "Qualifying: coverage `87.5%` vs target `90.0%`" in markdown


def test_inspect_season_prior_status_reports_missing_files(tmp_path: Path) -> None:
    """Backtest prior status should flag missing season files and legacy fallbacks."""
    data_dir = tmp_path / "processed"
    (data_dir / "car_characteristics").mkdir(parents=True, exist_ok=True)
    (data_dir / "track_characteristics").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "driver_characteristics.json").write_text("{}")

    status = _inspect_season_prior_status(data_dir=data_dir, season_year=2025)

    assert status["team"]["season_scoped"] is False
    assert status["driver"]["season_scoped"] is False
    assert status["driver"]["fallback_path"] == str(data_dir / "driver_characteristics.json")
    assert status["track"]["season_scoped"] is False

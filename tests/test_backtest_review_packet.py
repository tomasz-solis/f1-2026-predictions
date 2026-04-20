"""Tests for reviewer-facing backtest packet rendering."""

from __future__ import annotations

from pathlib import Path

from scripts.backtest_2025_season import _emit_review_packet_markdown


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
    assert "## Interval Calibration" in markdown
    assert "## Segment Breakdown" in markdown
    assert "## Error Analysis" in markdown
    assert "## Recommended Ablations" in markdown
    assert "Monaco Grand Prix" in markdown
    assert "Qualifying: coverage `87.5%` vs target `90.0%`" in markdown

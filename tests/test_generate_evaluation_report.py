"""Tests for evaluation report generation and output schema."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_evaluation_report import (
    build_report,
    render_error_analysis_markdown,
    render_markdown,
)


def _write_prediction(
    root: Path,
    *,
    race_slug: str,
    weekend_format: str,
    weather: str = "dry",
    session_name: str = "R",
    predicted_at: str | None = None,
    top_level_qualifying_target: str = "main_qualifying",
    top_level_race_target: str = "grand_prix_race",
    qualifying_rows: list[dict],
    race_rows: list[dict],
    actual_quali: list[dict],
    actual_race: list[dict],
    target_qualifying_rows: list[dict] | None = None,
    target_race_rows: list[dict] | None = None,
    target_actual_quali: list[dict] | None = None,
    target_actual_race: list[dict] | None = None,
) -> None:
    """Write one synthetic prediction artifact for report testing."""
    race_dir = root / "2026" / race_slug
    race_dir.mkdir(parents=True, exist_ok=True)
    stored_target_qualifying_rows = target_qualifying_rows or qualifying_rows
    stored_target_race_rows = target_race_rows or race_rows
    stored_target_actual_quali = target_actual_quali or actual_quali
    stored_target_actual_race = target_actual_race or actual_race
    payload = {
        "metadata": {
            "year": 2026,
            "race_name": race_slug.replace("_", " ").title(),
            "session_name": session_name,
            "weekend_format": weekend_format,
            "weather": weather,
            "predicted_at": predicted_at or f"2026-01-01T00:00:0{len(race_rows)}+00:00",
            "information_cutoff_at": predicted_at or f"2026-01-01T00:00:0{len(race_rows)}+00:00",
            "top_level_qualifying_target": top_level_qualifying_target,
            "top_level_race_target": top_level_race_target,
        },
        "qualifying": {"predicted_grid": qualifying_rows},
        "race": {"predicted_results": race_rows},
        "targets": {
            top_level_qualifying_target: {
                "target_session": "Q",
                "predicted_order": stored_target_qualifying_rows,
                "result_mode": "PREDICTED",
                "grid_source": "PREDICTED",
                "mean_confidence": 0.6,
                "eligible_at_save": True,
            },
            top_level_race_target: {
                "target_session": "R",
                "predicted_order": stored_target_race_rows,
                "result_mode": "PREDICTED",
                "grid_source": "PREDICTED",
                "mean_confidence": 0.6,
                "eligible_at_save": True,
            },
        },
        "actuals": {
            "qualifying": actual_quali,
            "race": actual_race,
            "targets": {
                top_level_qualifying_target: stored_target_actual_quali,
                top_level_race_target: stored_target_actual_race,
            },
        },
    }
    (race_dir / f"{race_slug}_{session_name.lower()}.json").write_text(json.dumps(payload))


def test_build_report_includes_accuracy_and_format_breakdown(tmp_path):
    """Report payload should include standardized accuracy and coverage sections."""
    predictions_dir = tmp_path / "predictions"
    rows_a = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 3},
        {"position": 2, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 4},
        {"position": 3, "driver": "LEC", "team": "Ferrari", "p5": 2, "p95": 5},
    ]
    rows_b = [
        {"position": 1, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 3},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 4},
        {"position": 3, "driver": "LEC", "team": "Ferrari", "p5": 2, "p95": 5},
    ]
    actual_a = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "NOR", "team": "McLaren"},
        {"position": 3, "driver": "LEC", "team": "Ferrari"},
    ]
    actual_b = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "LEC", "team": "Ferrari"},
        {"position": 3, "driver": "NOR", "team": "McLaren"},
    ]

    _write_prediction(
        predictions_dir,
        race_slug="bahrain_grand_prix",
        weekend_format="normal",
        weather="dry",
        qualifying_rows=rows_a,
        race_rows=rows_a,
        actual_quali=actual_a,
        actual_race=actual_a,
    )
    _write_prediction(
        predictions_dir,
        race_slug="monaco_grand_prix",
        weekend_format="sprint",
        weather="mixed",
        qualifying_rows=rows_b,
        race_rows=rows_b,
        actual_quali=actual_b,
        actual_race=actual_b,
    )

    report = build_report(2026, predictions_dir)

    assert report["predictions_analyzed"] == 2
    assert report["qualifying_accuracy"]["events_evaluated"] == 2
    assert report["race_accuracy"]["events_evaluated"] == 2
    assert report["evaluation_scope"]["selection_policy"] == "latest_checkpoint_per_race_and_target"
    assert report["format_breakdown"]["normal"]["prediction_files"] == 1
    assert report["format_breakdown"]["sprint"]["prediction_files"] == 1
    assert report["segment_breakdown"]["race"]["weather"]["mixed"]["events"] == 1
    assert report["segment_breakdown"]["race"]["track_type"]["street"]["events"] == 1
    assert report["error_analysis"]["race"]["worst_events"][0]["race_name"] == "Monaco Grand Prix"
    assert "mae" in report["qualifying_accuracy"]
    assert "mae" in report["race_accuracy"]


def test_render_markdown_mentions_accuracy_overview_and_format_breakdown(tmp_path):
    """Rendered markdown should surface the new review-oriented sections."""
    predictions_dir = tmp_path / "predictions"
    rows = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 3},
        {"position": 2, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 4},
    ]
    actual = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "NOR", "team": "McLaren"},
    ]
    _write_prediction(
        predictions_dir,
        race_slug="bahrain_grand_prix",
        weekend_format="normal",
        weather="dry",
        qualifying_rows=rows,
        race_rows=rows,
        actual_quali=actual,
        actual_race=actual,
    )

    markdown = render_markdown(build_report(2026, predictions_dir))

    assert "## 0. Accuracy Overview" in markdown
    assert "### Weekend Format Coverage" in markdown
    assert "## Selection Policy" in markdown
    assert "## 1. Segmented Performance" in markdown
    assert "## 3. Error Analysis" in markdown


def test_build_report_uses_latest_checkpoint_per_race_and_target(tmp_path):
    """Canonical evaluation should dedupe repeated race checkpoints down to the latest artifact."""
    predictions_dir = tmp_path / "predictions"
    earlier_rows = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 3},
        {"position": 2, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 4},
    ]
    later_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 3},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 4},
    ]
    actual_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren"},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
    ]
    _write_prediction(
        predictions_dir,
        race_slug="australian_grand_prix",
        weekend_format="normal",
        session_name="PRE",
        predicted_at="2026-03-10T10:00:00+00:00",
        qualifying_rows=earlier_rows,
        race_rows=earlier_rows,
        actual_quali=actual_rows,
        actual_race=actual_rows,
    )
    _write_prediction(
        predictions_dir,
        race_slug="australian_grand_prix",
        weekend_format="normal",
        session_name="FP3",
        predicted_at="2026-03-12T10:00:00+00:00",
        qualifying_rows=later_rows,
        race_rows=later_rows,
        actual_quali=actual_rows,
        actual_race=actual_rows,
    )

    report = build_report(2026, predictions_dir)

    assert report["predictions_analyzed"] == 2
    assert report["qualifying_pairs"] == 1
    assert report["race_pairs"] == 1
    assert report["qualifying_accuracy"]["mae"] == 0.0
    assert report["race_accuracy"]["mae"] == 0.0
    assert report["evaluation_scope"]["selected_prediction_counts"]["qualifying"] == 1
    assert report["evaluation_scope"]["selected_checkpoint_breakdown"]["qualifying"] == {"FP3": 1}
    assert report["evaluation_scope"]["ignored_intermediate_checkpoints"]["qualifying"] == 1


def test_render_error_analysis_markdown_surfaces_worst_events(tmp_path):
    """Standalone error-analysis markdown should focus on worst weekends and repeat misses."""
    predictions_dir = tmp_path / "predictions"
    predicted_rows = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 3},
        {"position": 2, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 4},
        {"position": 3, "driver": "LEC", "team": "Ferrari", "p5": 2, "p95": 5},
    ]
    actual_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren"},
        {"position": 2, "driver": "LEC", "team": "Ferrari"},
        {"position": 3, "driver": "VER", "team": "Red Bull Racing"},
    ]
    _write_prediction(
        predictions_dir,
        race_slug="monaco_grand_prix",
        weekend_format="normal",
        weather="mixed",
        qualifying_rows=predicted_rows,
        race_rows=predicted_rows,
        actual_quali=actual_rows,
        actual_race=actual_rows,
    )

    markdown = render_error_analysis_markdown(build_report(2026, predictions_dir))

    assert "# Model Error Analysis" in markdown
    assert "Worst weekends:" in markdown
    assert "Drivers that show up repeatedly among the largest misses:" in markdown
    assert "Monaco Grand Prix" in markdown


def test_build_report_reads_target_aware_intervals_when_legacy_rows_lack_bands(tmp_path):
    """Calibration and accuracy should use canonical target payloads before legacy rows."""
    predictions_dir = tmp_path / "predictions"
    legacy_rows = [
        {"position": 1, "driver": "VER", "team": "Red Bull Racing"},
        {"position": 2, "driver": "NOR", "team": "McLaren"},
    ]
    target_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren", "p5": 1, "p95": 2},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing", "p5": 1, "p95": 2},
    ]
    actual_rows = [
        {"position": 1, "driver": "NOR", "team": "McLaren"},
        {"position": 2, "driver": "VER", "team": "Red Bull Racing"},
    ]
    _write_prediction(
        predictions_dir,
        race_slug="australian_grand_prix",
        weekend_format="normal",
        qualifying_rows=legacy_rows,
        race_rows=legacy_rows,
        actual_quali=actual_rows,
        actual_race=actual_rows,
        target_qualifying_rows=target_rows,
        target_race_rows=target_rows,
    )

    report = build_report(2026, predictions_dir)

    assert report["qualifying_accuracy"]["mae"] == 0.0
    assert report["race_accuracy"]["mae"] == 0.0
    assert report["calibration"]["races_with_band_data"] == 1.0
    assert report["calibration"]["total_races_evaluated"] == 1.0

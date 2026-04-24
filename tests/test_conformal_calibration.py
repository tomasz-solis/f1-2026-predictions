"""Tests for conformal calibration helpers."""

from __future__ import annotations

from src.models.conformal_calibration import (
    apply_interval_radius_floor,
    build_conformal_calibration_artifact,
)


def test_build_conformal_calibration_artifact_groups_by_session_and_regime():
    """Conformal summaries should learn one radius per session/regime bucket."""
    artifact = build_conformal_calibration_artifact(
        rows=[
            {
                "session": "qualifying",
                "regime": "practice_backed",
                "residual": 1.0,
                "covered": True,
            },
            {
                "session": "qualifying",
                "regime": "practice_backed",
                "residual": 3.0,
                "covered": False,
            },
            {
                "session": "race",
                "regime": "checkpoint_backed",
                "residual": 2.0,
                "covered": True,
            },
        ],
        target_coverage=0.90,
        min_samples=2,
        max_radius=5.0,
        generated_at="2026-04-21T00:00:00+00:00",
    )

    qualifying_bucket = artifact.buckets["qualifying"]["practice_backed"]
    race_bucket = artifact.buckets["race"]["checkpoint_backed"]

    assert artifact.generated_at == "2026-04-21T00:00:00+00:00"
    assert qualifying_bucket["sample_count"] == 2
    assert qualifying_bucket["radius"] >= 2.0
    assert race_bucket["sample_count"] == 1
    assert race_bucket["radius"] == 0.0


def test_apply_interval_radius_floor_widens_rows_in_place():
    """Interval floor should widen narrow rows without leaving field bounds."""
    rows = [
        {"median_position": 2, "p5": 2, "p95": 2},
        {"median_position": 10, "p5": 9, "p95": 10},
    ]

    apply_interval_radius_floor(rows=rows, radius=2.0, field_size=12)

    assert rows == [
        {"median_position": 2, "p5": 1, "p95": 4},
        {"median_position": 10, "p5": 8, "p95": 12},
    ]

"""Tests for the Team Comparison snapshot migration CLI."""

from scripts.backfill_team_comparison_snapshots import (
    _audit_latest_snapshots,
    _build_snapshot_jobs,
)


def test_build_snapshot_jobs_normalizes_testing_session_labels() -> None:
    jobs = _build_snapshot_jobs(
        (
            "2026::British Grand Prix::FP1",
            "2026::British Grand Prix::Q",
            "2026::Testing 1::Testing 1 Day 1",
            "2026::Testing 1::Testing 1 Day 2",
        ),
        year=2026,
    )

    by_event = {job.event_name: job for job in jobs}
    assert by_event["British Grand Prix"].sessions == ("FP1", "Q")
    assert by_event["British Grand Prix"].cache_dir.endswith(".fastf1_cache")
    assert by_event["Testing 1"].sessions == ("Day 1", "Day 2")
    assert by_event["Testing 1"].cache_dir.endswith(".fastf1_cache_testing")


def test_audit_latest_snapshots_accepts_complete_backfill_payload() -> None:
    artifact_key = "2026::British Grand Prix::FP1"
    audit = _audit_latest_snapshots(
        {
            artifact_key: {
                "source": "snapshot_history_backfill",
                "teams": {
                    "Mercedes": {
                        "profiles": {
                            "balanced": {
                                "slow_corner_performance": 0.7,
                                "slow_corner_seconds": 29.131,
                                "medium_corner_performance": 1.0,
                                "medium_corner_seconds": 36.906,
                                "fast_corner_performance": 0.4,
                                "fast_corner_seconds": 25.631,
                                "braking_performance": 0.0,
                                "braking_pct": 14.5533,
                            }
                        }
                    }
                },
            }
        },
        migrated_keys=(artifact_key,),
    )

    assert audit.passed is True
    assert audit.source_counts == {"snapshot_history_backfill": 1}


def test_audit_latest_snapshots_rejects_stale_or_normalized_only_payload() -> None:
    artifact_key = "2026::British Grand Prix::FP1"
    audit = _audit_latest_snapshots(
        {
            artifact_key: {
                "source": "testing_practice_extraction",
                "teams": {
                    "Mercedes": {
                        "profiles": {
                            "balanced": {
                                "slow_corner_performance": 0.7,
                                "braking_performance": 0.0,
                            }
                        }
                    }
                },
            }
        },
        migrated_keys=(artifact_key,),
    )

    assert audit.passed is False
    assert audit.stale_sources == ((artifact_key, "testing_practice_extraction"),)
    assert len(audit.missing_raw_metrics) == 2

"""Tests for background session automation workflow."""

from __future__ import annotations

from src.systems import session_automation as automation


def test_schedule_config_roundtrip_file_mode(patcher, tmp_path):
    """Schedule config should round-trip via local file fallback."""
    state_file = tmp_path / "session_automation_schedule.json"
    patcher.setattr(automation, "_SCHEDULE_STATE_FILE", state_file)
    patcher.setattr(automation, "should_read_db_first", lambda: False)
    patcher.setattr(automation, "should_write_to_db", lambda: False)
    patcher.setattr(automation, "should_write_to_file", lambda: True)

    created = automation.ensure_session_automation_config(
        2026,
        enabled=True,
        auto_predict=False,
        weather="mixed",
        lookback_days=9,
        lookahead_days=1,
    )
    loaded = automation.load_session_automation_config(2026)

    assert created.year == 2026
    assert loaded.enabled is True
    assert loaded.auto_predict is False
    assert loaded.weather == "mixed"
    assert loaded.lookback_days == 9
    assert loaded.lookahead_days == 1


def test_run_cycle_generates_prediction_for_latest_session(patcher):
    """Cycle should auto-generate prediction after a newly completed session."""
    saved_predictions: list[dict] = []

    class _Logger:
        def has_prediction_for_session(self, year: int, race_name: str, session_name: str) -> bool:
            del year, race_name, session_name
            return False

        def save_prediction(self, **kwargs):
            saved_predictions.append(kwargs)

        def update_actuals(self, **kwargs) -> bool:
            del kwargs
            return False

        def load_prediction(self, year: int, race_name: str, session_name: str):
            del year, race_name, session_name
            return None

    class _Detector:
        def get_session_completion_state(self, year: int, race_name: str, session_name: str) -> str:
            del year, race_name, session_name
            return "incomplete"

    patcher.setattr(
        automation,
        "load_session_automation_config",
        lambda year: automation.SessionAutomationConfig(year=year, enabled=True, auto_predict=True),
    )
    patcher.setattr(automation, "PredictionLogger", _Logger)
    patcher.setattr(automation, "SessionDetector", _Detector)
    patcher.setattr(automation, "needs_update", lambda year, force_recheck=False: (False, []))
    patcher.setattr(
        automation,
        "_iter_candidate_events",
        lambda year, lookback_days, lookahead_days: [("Bahrain Grand Prix", False)],
    )
    patcher.setattr(
        automation,
        "detect_event_boundary_refresh_if_needed",
        lambda **kwargs: {
            "refresh_needed": True,
            "latest_elapsed_session": "FP1",
        },
    )
    patcher.setattr(
        automation,
        "auto_update_practice_characteristics_if_needed",
        lambda **kwargs: {
            "updated": True,
            "completed_fp_sessions": ["FP1"],
        },
    )
    patcher.setattr(automation, "is_sprint_weekend", lambda year, race_name: False)
    patcher.setattr(automation, "get_artifact_versions", lambda year=2026: {"k": (1, "ts")})
    patcher.setattr(
        automation,
        "run_prediction",
        lambda race_name, weather, versions, is_sprint, year: {
            "qualifying": {"grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
            "race": {"finish_order": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        },
    )

    summary = automation.run_session_automation_cycle(
        year=2026,
        auto_predict=True,
        reconcile_actuals=False,
    )

    assert summary.checked_events == 1
    assert summary.updated_practice_events == ["Bahrain Grand Prix"]
    assert summary.generated_predictions == ["Bahrain Grand Prix::FP1"]
    assert len(saved_predictions) == 1
    assert saved_predictions[0]["session_name"] == "FP1"
    assert saved_predictions[0]["weather"] == "dry"


def test_run_cycle_reconciles_sprint_actuals_and_writes_accuracy_snapshots(patcher):
    """Completed sprint weekends should reconcile saved checkpoints and write snapshots."""
    fetch_calls: list[str] = []
    accuracy_saves: list[dict] = []

    class _Logger:
        def __init__(self):
            self._records = {
                session: {
                    "metadata": {
                        "year": 2026,
                        "race_name": "Chinese Grand Prix",
                        "session_name": session,
                        "weekend_format": "sprint",
                    },
                    "qualifying": {
                        "predicted_grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]
                    },
                    "race": {
                        "predicted_results": [{"position": 1, "driver": "VER", "team": "Red Bull"}]
                    },
                    "targets": {
                        "FP1": {
                            "sprint_qualifying": {
                                "target_session": "SQ",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                            "sprint_race": {
                                "target_session": "SPRINT",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                            "main_qualifying": {
                                "target_session": "Q",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                            "grand_prix_race": {
                                "target_session": "R",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                        },
                        "SQ": {
                            "sprint_race": {
                                "target_session": "SPRINT",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                            "main_qualifying": {
                                "target_session": "Q",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                            "grand_prix_race": {
                                "target_session": "R",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                        },
                        "Sprint": {
                            "main_qualifying": {
                                "target_session": "Q",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                            "grand_prix_race": {
                                "target_session": "R",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                        },
                        "Q": {
                            "grand_prix_race": {
                                "target_session": "R",
                                "predicted_order": [
                                    {"position": 1, "driver": "VER", "team": "Red Bull"}
                                ],
                                "eligible_at_save": True,
                            },
                        },
                        "R": {},
                    }.get(session, {}),
                    "actuals": {"qualifying": None, "race": None, "targets": {}},
                }
                for session in ["FP1", "SQ", "Sprint", "Q", "R"]
            }

        def has_prediction_for_session(self, year: int, race_name: str, session_name: str) -> bool:
            del year, race_name
            return session_name in self._records

        def save_prediction(self, **kwargs):
            del kwargs
            raise AssertionError("save_prediction should not be called in this test")

        def update_actuals(
            self,
            year: int,
            race_name: str,
            session_name: str,
            qualifying_results=None,
            race_results=None,
            run_id=None,
            target_actual_results=None,
        ) -> bool:
            del year, race_name, run_id
            record_key = "Sprint" if session_name == "SPRINT" else session_name
            if record_key not in self._records:
                return False
            self._records[record_key]["actuals"]["qualifying"] = qualifying_results
            self._records[record_key]["actuals"]["race"] = race_results
            self._records[record_key]["actuals"]["targets"] = target_actual_results or {}
            return True

        def load_prediction(self, year: int, race_name: str, session_name: str):
            del year, race_name
            record_key = "Sprint" if session_name == "SPRINT" else session_name
            return self._records.get(record_key)

        def get_all_predictions(self, year: int):
            del year
            return list(self._records.values())

    class _Detector:
        def get_session_completion_state(self, year: int, race_name: str, session_name: str) -> str:
            del year, race_name
            if session_name == "R":
                return "completed"
            return "incomplete"

    class _Metrics:
        def calculate_prediction_target_metrics(self, prediction, *, is_sprint):
            del is_sprint
            return {
                target_key: {
                    "field_size": 1.0,
                    "overall_mae": 0.0,
                    "top_3_hits": 1.0,
                    "top_3_pct": 100.0,
                    "top_10_hits": 1.0,
                    "top_10_pct": 100.0,
                    "exact_accuracy": 100.0,
                    "within_1": 100.0,
                    "within_3": 100.0,
                    "correlation": 1.0,
                }
                for target_key in prediction.get("targets", {})
            }

        def calculate_all_metrics(self, prediction):
            return {"metadata": prediction["metadata"], "qualifying": {}, "race": {}}

    class _Store:
        def __init__(self, data_root: str = "data"):
            self.data_root = data_root

        def save_artifact(self, **kwargs):
            accuracy_saves.append(kwargs)

    patcher.setattr(
        automation,
        "load_session_automation_config",
        lambda year: automation.SessionAutomationConfig(
            year=year, enabled=True, auto_predict=False
        ),
    )
    patcher.setattr(automation, "PredictionLogger", _Logger)
    patcher.setattr(automation, "SessionDetector", _Detector)
    patcher.setattr(automation, "PredictionMetrics", _Metrics)
    patcher.setattr(automation, "ArtifactStore", _Store)
    patcher.setattr(automation, "needs_update", lambda year, force_recheck=False: (False, []))
    patcher.setattr(
        automation,
        "_iter_candidate_events",
        lambda year, lookback_days, lookahead_days: [("Chinese Grand Prix", True)],
    )
    patcher.setattr(
        automation,
        "detect_event_boundary_refresh_if_needed",
        lambda **kwargs: {"refresh_needed": False, "latest_elapsed_session": "R"},
    )
    patcher.setattr(
        automation,
        "auto_update_practice_characteristics_if_needed",
        lambda **kwargs: {"updated": False, "completed_fp_sessions": ["FP1"]},
    )
    patcher.setattr(automation, "is_sprint_weekend", lambda year, race_name: True)
    patcher.setattr(
        automation,
        "fetch_actual_session_results",
        lambda year, race_name, session_name: (
            fetch_calls.append(session_name)
            or [{"position": 1, "driver": "VER", "team": "Red Bull"}]
        ),
    )

    summary = automation.run_session_automation_cycle(
        year=2026,
        auto_predict=False,
        reconcile_actuals=True,
    )

    assert summary.checked_events == 1
    assert summary.generated_predictions == []
    assert summary.reconciled_actuals == ["Chinese Grand Prix::5"]
    assert summary.accuracy_snapshots == 7
    assert sorted(set(fetch_calls)) == ["Q", "R", "SQ", "Sprint"]
    assert len(accuracy_saves) == 7

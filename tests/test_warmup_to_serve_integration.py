"""Integration coverage for the warmup-to-serve dashboard flow."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.dashboard import live_prediction_flow, warmup


def _base_store_key(
    *,
    year: int,
    race_name: str,
    checkpoint: str,
    artifact_hash: str,
    boundary_signature: str,
    **_: Any,
) -> tuple[str, str, str, str, str]:
    """Build the in-memory key used for persisted warmup base features."""
    return (
        str(year),
        str(race_name),
        str(checkpoint),
        str(artifact_hash),
        str(boundary_signature),
    )


def _prediction_store_key(
    *,
    year: int,
    race_name: str,
    weather: str,
    artifact_hash: str,
    boundary_signature: str,
    **_: Any,
) -> tuple[str, str, str, str, str]:
    """Build the in-memory key used for persisted warmed predictions."""
    return (
        str(year),
        str(race_name),
        str(weather).strip().lower(),
        str(artifact_hash),
        str(boundary_signature),
    )


class _IntegrationPredictor:
    """Minimal predictor that turns a predicted FP3 grid into a stable race payload."""

    def predict_qualifying(
        self,
        *,
        year: int,
        race_name: str,
        qualifying_stage: str,
        n_simulations: int,
        practice_signal_mode: str,
        checkpoint_session_name: str,
    ) -> dict[str, Any]:
        """Return one deterministic qualifying prediction for the integration path."""
        del year, race_name, qualifying_stage, n_simulations, practice_signal_mode
        assert checkpoint_session_name == "FP3"
        return {
            "grid": [
                {"position": 1, "driver": "RUS", "team": "Mercedes"},
                {"position": 2, "driver": "LEC", "team": "Ferrari"},
            ],
            "data_confidence_score": 0.95,
            "data_source": "FP3 short-stint",
        }

    def predict_race(
        self,
        *,
        qualifying_grid: list[dict[str, Any]],
        weather: str,
        race_name: str,
        n_simulations: int,
        year: int,
        input_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Return a stable race payload so warmup persistence stays deterministic."""
        del weather, race_name, n_simulations, year
        pole_sitter = dict(qualifying_grid[0])
        second_place = dict(qualifying_grid[1])
        return {
            "finish_order": [
                {
                    "position": 1,
                    "driver": pole_sitter["driver"],
                    "team": pole_sitter["team"],
                    "confidence": 78.4,
                },
                {
                    "position": 2,
                    "driver": second_place["driver"],
                    "team": second_place["team"],
                    "confidence": 71.2,
                },
            ],
            "data_source": "Warmup integration predictor",
            "input_confidence_echo": input_confidence,
        }


def test_warmup_cycle_persists_payload_that_dashboard_serves(monkeypatch):
    """Warmup should persist one race payload that the request path serves unchanged."""
    fixed_now = datetime(2026, 3, 24, 9, 0, tzinfo=UTC)
    race_name = "Japanese Grand Prix"
    weather = "dry"
    artifact_hash = "artifact_hash"
    boundary_signature = "sig_fp3"
    artifact_versions = {"car_characteristics::2026::car_characteristics": (7, "ts")}
    predictor = _IntegrationPredictor()

    base_feature_store: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    prediction_store: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    horizon_store: dict[tuple[int, str], dict[str, Any]] = {}

    def _fetch_actual_results(
        year: int,
        race_name: str,
        session_name: str,
    ) -> tuple[list[dict[str, Any]] | None, str]:
        """Leave Q and R unresolved so the warmup path stays on the FP3 checkpoint."""
        del year, race_name, session_name
        return None, "INCOMPLETE"

    def _load_base_features(**kwargs: Any) -> dict[str, Any] | None:
        """Read persisted warmup base features from the shared in-memory store."""
        return base_feature_store.get(_base_store_key(**kwargs))

    def _save_base_features(**kwargs: Any) -> None:
        """Persist warmup base features into the shared in-memory store."""
        base_feature_store[_base_store_key(**kwargs)] = kwargs["base_features"]

    def _load_prediction(**kwargs: Any) -> dict[str, Any] | None:
        """Read warmed predictions from the shared in-memory store."""
        return prediction_store.get(_prediction_store_key(**kwargs))

    def _save_prediction(**kwargs: Any) -> None:
        """Persist warmed predictions into the shared in-memory store."""
        prediction_store[_prediction_store_key(**kwargs)] = kwargs["prediction_results"]

    def _save_horizon_index(**kwargs: Any) -> None:
        """Persist horizon metadata into the shared in-memory store."""
        horizon_store[(int(kwargs["year"]), str(kwargs["artifact_hash"]))] = dict(kwargs)

    def _load_horizon_index(*, year: int, artifact_hash: str, **_: Any) -> dict[str, Any] | None:
        """Read the horizon metadata for the current warmed artifact state."""
        return horizon_store.get((int(year), str(artifact_hash)))

    monkeypatch.setattr(warmup, "should_write_to_db", lambda: False)
    monkeypatch.setattr(warmup, "_refresh_anchor_practice_characteristics", lambda **kwargs: {})
    monkeypatch.setattr(
        warmup,
        "get_prediction_precompute_config",
        lambda: {
            "enabled": True,
            "horizon_races": 1,
            "weather_scenarios": [weather],
            "max_file_entries": 32,
            "qualifying_n_simulations": 8,
            "race_n_simulations": 8,
        },
    )
    monkeypatch.setattr(
        warmup,
        "_resolve_warmup_targets",
        lambda year, now_utc, horizon_races: warmup.WarmupTargets(
            anchor_race_name=race_name,
            anchor_is_sprint=False,
            target_races=(race_name,),
        ),
    )
    monkeypatch.setattr(
        warmup,
        "_resolve_checkpoint_context",
        lambda year, race_name, is_sprint, now_utc, session_detector: warmup.CheckpointContext(
            checkpoint="Q",
            expected_checkpoint="Q",
            latest_ready_checkpoint="Q",
            checkpoint_ready=True,
            reason="ready",
            boundary_signature=boundary_signature,
        ),
    )
    monkeypatch.setattr(warmup, "get_artifact_versions", lambda year=2026: dict(artifact_versions))
    monkeypatch.setattr(warmup, "compute_artifact_hash", lambda versions: artifact_hash)
    monkeypatch.setattr(warmup, "_load_predictor", lambda artifact_versions, year: predictor)
    monkeypatch.setattr(
        warmup,
        "build_checkpoint_overlay_predictor",
        lambda **kwargs: kwargs["base_predictor"],
    )
    monkeypatch.setattr(warmup, "is_sprint_weekend", lambda year, race_name: False)
    monkeypatch.setattr(
        warmup,
        "fetch_actual_competitive_results_if_completed",
        _fetch_actual_results,
    )
    monkeypatch.setattr(warmup, "load_precomputed_base_features", _load_base_features)
    monkeypatch.setattr(warmup, "save_precomputed_base_features", _save_base_features)
    monkeypatch.setattr(warmup, "load_precomputed_prediction", _load_prediction)
    monkeypatch.setattr(warmup, "save_precomputed_prediction", _save_prediction)
    monkeypatch.setattr(warmup, "save_precompute_horizon_index", _save_horizon_index)

    summary = warmup.run_warmup_precompute_cycle(
        2026,
        now_utc=fixed_now,
        verify_db_writes=False,
    )

    assert summary.status == "success"
    assert summary.base_generated == 1
    assert summary.predictions_generated == 1
    assert summary.ready_races == [race_name]
    assert len(base_feature_store) == 1
    assert len(prediction_store) == 1
    assert horizon_store[(2026, artifact_hash)]["ready_races"] == [race_name]

    warmed_prediction = prediction_store[
        _prediction_store_key(
            year=2026,
            race_name=race_name,
            weather=weather,
            artifact_hash=artifact_hash,
            boundary_signature=boundary_signature,
        )
    ]
    assert warmed_prediction["qualifying"]["grid_source"] == "PREDICTED"
    assert warmed_prediction["qualifying"]["grid"][0]["driver"] == "RUS"
    assert warmed_prediction["race"]["grid_source"] == "PREDICTED"
    assert warmed_prediction["race"]["input_confidence"] == 0.95

    live_prediction_flow.clear_prediction_result_cache()
    progress_messages: list[str] = []
    monkeypatch.setattr(
        live_prediction_flow,
        "_get_prediction_precompute_settings",
        lambda: {
            "enabled": True,
            "horizon_races": 1,
            "weather_scenarios": [weather],
            "max_file_entries": 32,
        },
    )
    monkeypatch.setattr(
        live_prediction_flow,
        "_resolve_precompute_targets",
        lambda year, race_name, horizon_races: [race_name],
    )
    monkeypatch.setattr(
        live_prediction_flow, "compute_artifact_hash", lambda versions: artifact_hash
    )
    monkeypatch.setattr(live_prediction_flow, "load_precomputed_prediction", _load_prediction)
    monkeypatch.setattr(live_prediction_flow, "load_precompute_horizon_index", _load_horizon_index)

    output = live_prediction_flow.execute_live_prediction_pipeline_core(
        race_name=race_name,
        weather=weather,
        year=2026,
        force_refresh=False,
        progress_callback=progress_messages.append,
        clear_fastf1_race_cache_fn=lambda year, race_name: None,
        auto_update_if_needed_fn=lambda force_recheck=False, year=2026: None,
        is_sprint_weekend_fn=lambda year, race_name: False,
        detect_event_boundary_refresh_if_needed_fn=(
            lambda year, race_name, is_sprint, session_detector=None: {
                "refresh_needed": False,
                "reason": "no_change",
                "new_sessions": [],
                "boundary_signature": boundary_signature,
                "latest_elapsed_session": "Q",
            }
        ),
        auto_update_practice_characteristics_if_needed_fn=(
            lambda year, race_name, is_sprint, force_recheck=False, session_detector=None: {
                "updated": False,
                "completed_fp_sessions": [],
            }
        ),
        clear_resource_cache_fn=lambda: None,
        clear_data_cache_fn=lambda: None,
        get_artifact_versions_fn=lambda year=2026: dict(artifact_versions),
        run_prediction_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("request path should serve warmed predictions instead of simulating")
        ),
    )

    assert output["prediction_results"] == warmed_prediction
    assert output["boundary_session_name"] == "FP3"
    assert output["precompute_summary"]["ready_races"] == [race_name]
    assert output["prediction_cache_hit"] is False
    assert progress_messages[-1] == "Loaded persisted prediction..."

    served_prediction = output["prediction_results"]
    assert served_prediction["qualifying"]["grid"][0] == {
        "position": 1,
        "driver": "RUS",
        "team": "Mercedes",
    }
    assert served_prediction["race"]["finish_order"][0]["position"] == 1
    assert served_prediction["race"]["finish_order"][0]["driver"] == "RUS"
    assert served_prediction["race"]["finish_order"][0]["confidence"] == 78.4
    assert served_prediction["race"]["input_confidence"] == 0.95

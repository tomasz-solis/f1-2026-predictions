"""Tests for persisted model-diagnostics dashboard rendering."""

from __future__ import annotations

from typing import Any

from src.dashboard.model_diagnostics import (
    load_team_strength_construct_audit,
    load_team_strength_prediction_replay_test,
    load_team_strength_refit_candidate_test,
    render_model_diagnostics,
)


class _FakeArtifactStore:
    """Small ArtifactStore stand-in keyed by artifact type and artifact key."""

    def __init__(self, payloads: dict[tuple[str, str], dict[str, Any] | None]) -> None:
        """Store payloads for deterministic dashboard tests."""
        self.payloads = payloads

    def load_artifact(self, artifact_type: str, artifact_key: str) -> dict[str, Any] | None:
        """Return the configured payload for one artifact lookup."""
        return self.payloads.get((artifact_type, artifact_key))


class _FakeColumn:
    """Context-manager stand-in for Streamlit columns."""

    def __enter__(self) -> _FakeColumn:
        """Enter the fake column context."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """Exit the fake column context without suppressing errors."""
        return False


class _FakeStreamlit:
    """Capture dashboard rendering calls without importing Streamlit runtime."""

    def __init__(self) -> None:
        """Initialize call storage for assertions."""
        self.dataframes: list[Any] = []
        self.captions: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.subheaders: list[str] = []
        self.metrics: list[tuple[str, Any]] = []

    def info(self, message: str) -> None:
        """Record an info message."""
        self.infos.append(str(message))

    def warning(self, message: str) -> None:
        """Record a warning message."""
        self.warnings.append(str(message))

    def caption(self, message: str) -> None:
        """Record a caption message."""
        self.captions.append(str(message))

    def subheader(self, message: str) -> None:
        """Record a subheader message."""
        self.subheaders.append(str(message))

    def metric(self, label: str, value: Any) -> None:
        """Record a metric call."""
        self.metrics.append((label, value))

    def columns(self, count: int) -> list[_FakeColumn]:
        """Return fake column context managers."""
        return [_FakeColumn() for _ in range(count)]

    def dataframe(self, frame: Any, **_kwargs: Any) -> None:
        """Record a dataframe payload."""
        self.dataframes.append(frame)


def test_load_team_strength_construct_audit_uses_stable_artifact_key() -> None:
    """The dashboard loader should use the persisted construct-audit key."""
    payload = {"status": "measured"}
    store = _FakeArtifactStore(
        {("model_diagnostics", "2026::team_strength_construct_row_audit"): payload}
    )

    assert load_team_strength_construct_audit(year=2026, artifact_store=store) == payload


def test_load_team_strength_refit_candidate_test_uses_stable_artifact_key() -> None:
    """The dashboard loader should use the persisted refit-candidate key."""
    payload = {"status": "measured"}
    store = _FakeArtifactStore(
        {("model_diagnostics", "2026::team_strength_refit_candidate_test"): payload}
    )

    assert load_team_strength_refit_candidate_test(year=2026, artifact_store=store) == payload


def test_load_team_strength_prediction_replay_test_uses_stable_artifact_key() -> None:
    """The dashboard loader should use the persisted prediction-replay key."""
    payload = {"status": "measured"}
    store = _FakeArtifactStore(
        {("model_diagnostics", "2026::team_strength_prediction_replay_test"): payload}
    )

    assert load_team_strength_prediction_replay_test(year=2026, artifact_store=store) == payload


def test_render_model_diagnostics_includes_construct_audit_table() -> None:
    """The diagnostics page should render persisted audit artifacts when available."""
    replay_payload = {
        "built_at": "now",
        "status": "measured",
        "warnings": [],
        "source_state": {
            "replay_race_count": 4,
            "live_artifact_races_completed": 4,
        },
        "dry_leakage": {"correlation": 0.1, "rows": []},
        "wet_leakage": {"state": "not_evaluable"},
        "regulation_reset_monitoring": {
            "metrics_by_session_kind": {
                "race": {
                    "state": "measured",
                    "n_rows": 2,
                    "n_races": 1,
                    "prediction_slope": 1.2,
                    "r_squared": 0.5,
                    "rmse_s": 0.3,
                    "outside_historical_one_se_band": True,
                }
            }
        },
        "historical_scale_reference": {"residual_outliers": []},
    }
    audit_payload = {
        "built_at": "now",
        "status": "measured",
        "metrics_by_session_kind": {
            "race": {
                "n_rows": 2,
                "n_races": 1,
                "prediction_slope": 1.2,
                "team_target_slope": 1.4,
                "rmse_s": 0.3,
                "outside_historical_one_se_band": True,
            }
        },
        "largest_abs_residual_rows": [
            {
                "session_kind": "race",
                "race_name": "Race 1",
                "team": "Team A",
                "driver_code": "D1",
                "observed_driver_to_field_s": 1.0,
                "predicted_driver_to_field_s": 0.5,
                "residual_s": 0.5,
                "n_construct_laps": 4,
            }
        ],
    }
    refit_payload = {
        "built_at": "now",
        "status": "measured",
        "aggregate": [
            {
                "candidate": "current_frozen_mapping",
                "n_folds": 1,
                "n_rows": 2,
                "weighted_mse_s2": 1.0,
                "weighted_rmse_s": 1.0,
            }
        ],
        "decision_assessment": {
            "state": "not_enough_evidence",
            "recommendation": "Keep testing.",
        },
    }
    replay_test_payload = {
        "built_at": "now",
        "status": "measured",
        "race_target_aggregate": [
            {
                "n_rows": 1,
                "current_mse": 3.0,
                "candidate_mse": 2.0,
                "mse_pct_delta": -33.3,
            }
        ],
        "decision_assessment": {
            "state": "supports_race_only_prediction_replay_candidate",
            "recommendation": "Review for release.",
        },
    }
    store = _FakeArtifactStore(
        {
            ("model_diagnostics", "2026::replay_leakage_diagnostics"): replay_payload,
            ("model_diagnostics", "2026::team_strength_construct_row_audit"): audit_payload,
            ("model_diagnostics", "2026::team_strength_refit_candidate_test"): refit_payload,
            (
                "model_diagnostics",
                "2026::team_strength_prediction_replay_test",
            ): replay_test_payload,
        }
    )
    fake_st = _FakeStreamlit()

    render_model_diagnostics(year=2026, st_module=fake_st, artifact_store=store)

    assert "Construct-row audit" in fake_st.subheaders
    assert "Refit candidate test" in fake_st.subheaders
    assert "Prediction replay test" in fake_st.subheaders
    assert len(fake_st.dataframes) >= 5

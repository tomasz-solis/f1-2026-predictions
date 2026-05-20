"""Read-only dashboard rendering for persisted model diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from src.analysis.replay_leakage_diagnostics import (
    REPLAY_LEAKAGE_ARTIFACT_TYPE,
    replay_leakage_artifact_key,
)
from src.analysis.team_strength_construct_audit import (
    TEAM_STRENGTH_CONSTRUCT_AUDIT_ARTIFACT_TYPE,
    team_strength_construct_audit_artifact_key,
)
from src.analysis.team_strength_prediction_replay_test import (
    TEAM_STRENGTH_PREDICTION_REPLAY_ARTIFACT_TYPE,
    team_strength_prediction_replay_artifact_key,
)
from src.analysis.team_strength_refit_candidate_test import (
    TEAM_STRENGTH_REFIT_TEST_ARTIFACT_TYPE,
    team_strength_refit_test_artifact_key,
)
from src.persistence.artifact_store import ArtifactStore


def load_replay_leakage_diagnostics(
    *,
    year: int,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Load the persisted replay/leakage diagnostics artifact for one season."""
    store = artifact_store or ArtifactStore(data_root="data")
    payload = store.load_artifact(
        REPLAY_LEAKAGE_ARTIFACT_TYPE, replay_leakage_artifact_key(int(year))
    )
    return payload if isinstance(payload, dict) else None


def load_team_strength_construct_audit(
    *,
    year: int,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Load the persisted team-strength construct-row audit for one season."""
    store = artifact_store or ArtifactStore(data_root="data")
    payload = store.load_artifact(
        TEAM_STRENGTH_CONSTRUCT_AUDIT_ARTIFACT_TYPE,
        team_strength_construct_audit_artifact_key(int(year)),
    )
    return payload if isinstance(payload, dict) else None


def load_team_strength_refit_candidate_test(
    *,
    year: int,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Load the persisted team-strength refit-candidate test for one season."""
    store = artifact_store or ArtifactStore(data_root="data")
    payload = store.load_artifact(
        TEAM_STRENGTH_REFIT_TEST_ARTIFACT_TYPE,
        team_strength_refit_test_artifact_key(int(year)),
    )
    return payload if isinstance(payload, dict) else None


def load_team_strength_prediction_replay_test(
    *,
    year: int,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, Any] | None:
    """Load the persisted full prediction replay test for one season."""
    store = artifact_store or ArtifactStore(data_root="data")
    payload = store.load_artifact(
        TEAM_STRENGTH_PREDICTION_REPLAY_ARTIFACT_TYPE,
        team_strength_prediction_replay_artifact_key(int(year)),
    )
    return payload if isinstance(payload, dict) else None


def render_model_diagnostics(
    *,
    year: int,
    st_module: Any,
    artifact_store: ArtifactStore | None = None,
) -> None:
    """Render persisted model diagnostics without recomputing them."""
    artifact = load_replay_leakage_diagnostics(year=year, artifact_store=artifact_store)
    if not artifact:
        st_module.info(
            "No replay/leakage diagnostics artifact found yet. Run "
            "`scripts/build_replay_leakage_diagnostics.py` after rebuilding replay outputs."
        )
        return

    source_state = artifact.get("source_state", {})
    dry_leakage = artifact.get("dry_leakage", {})
    regulation_reset = artifact.get("regulation_reset_monitoring", {})
    wet_leakage = artifact.get("wet_leakage", {})

    st_module.caption(
        f"Built {artifact.get('built_at', 'unknown')} • "
        f"status `{artifact.get('status', 'unknown')}`"
    )
    _render_warning_block(artifact.get("warnings", []), st_module=st_module)

    cols = st_module.columns(4)
    with cols[0]:
        st_module.metric("Replay races", source_state.get("replay_race_count", "—"))
    with cols[1]:
        st_module.metric(
            "Live races",
            source_state.get("live_artifact_races_completed", "—"),
        )
    with cols[2]:
        st_module.metric("Dry leakage corr", _fmt(dry_leakage.get("correlation")))
    with cols[3]:
        st_module.metric("Wet status", str(wet_leakage.get("state", "unknown")))

    st_module.subheader("Regulation-reset monitoring")
    _render_regulation_reset_table(regulation_reset, st_module=st_module)

    st_module.subheader("Construct-row audit")
    construct_audit = load_team_strength_construct_audit(year=year, artifact_store=artifact_store)
    _render_construct_audit(construct_audit, st_module=st_module)

    st_module.subheader("Refit candidate test")
    refit_test = load_team_strength_refit_candidate_test(year=year, artifact_store=artifact_store)
    _render_refit_candidate_test(refit_test, st_module=st_module)

    st_module.subheader("Prediction replay test")
    prediction_replay = load_team_strength_prediction_replay_test(
        year=year,
        artifact_store=artifact_store,
    )
    _render_prediction_replay_test(prediction_replay, st_module=st_module)

    st_module.subheader("Dry leakage")
    st_module.caption(
        "The current artifact reports a legacy proxy until race/quali seconds fields exist."
    )
    dry_rows = dry_leakage.get("rows", [])
    if isinstance(dry_rows, list) and dry_rows:
        st_module.dataframe(
            pd.DataFrame(dry_rows)[
                [
                    "driver_code",
                    "team",
                    "delta_rating_mu",
                    "delta_team_strength",
                    "delta_team_seconds",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st_module.info("No dry-leakage rows were available in the diagnostics artifact.")

    st_module.subheader("Per-driver residual outliers")
    residual_outliers = (
        artifact.get("historical_scale_reference", {}).get("residual_outliers", [])
        if isinstance(artifact.get("historical_scale_reference"), Mapping)
        else []
    )
    if isinstance(residual_outliers, list) and residual_outliers:
        st_module.dataframe(pd.DataFrame(residual_outliers), width="stretch", hide_index=True)
    else:
        st_module.caption("No historical residual outliers above the review threshold.")


def _render_regulation_reset_table(
    regulation_reset: Mapping[str, Any],
    *,
    st_module: Any,
) -> None:
    """Render the measured 2026 transfer metrics table."""
    metrics = regulation_reset.get("metrics_by_session_kind", {})
    rows: list[dict[str, Any]] = []
    if isinstance(metrics, Mapping):
        for session_kind, payload in metrics.items():
            if not isinstance(payload, Mapping):
                continue
            rows.append(
                {
                    "session": session_kind,
                    "state": payload.get("state", "not_available"),
                    "rows": payload.get("n_rows", 0),
                    "races": payload.get("n_races", 0),
                    "slope": payload.get("prediction_slope"),
                    "r_squared": payload.get("r_squared"),
                    "rmse_s": payload.get("rmse_s"),
                    "outside_1se": payload.get("outside_historical_one_se_band"),
                }
            )
    if not rows:
        st_module.info(str(regulation_reset.get("reason", "No transfer metrics available.")))
        return
    st_module.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_construct_audit(
    construct_audit: Mapping[str, Any] | None,
    *,
    st_module: Any,
) -> None:
    """Render the construct-row audit when the persisted artifact exists."""
    if not construct_audit:
        st_module.caption("No construct-row audit artifact found yet.")
        return

    st_module.caption(
        f"Built {construct_audit.get('built_at', 'unknown')} • "
        f"status `{construct_audit.get('status', 'unknown')}`"
    )
    metrics = construct_audit.get("metrics_by_session_kind", {})
    rows: list[dict[str, Any]] = []
    if isinstance(metrics, Mapping):
        for session_kind, payload in metrics.items():
            if not isinstance(payload, Mapping):
                continue
            rows.append(
                {
                    "session": session_kind,
                    "rows": payload.get("n_rows", 0),
                    "races": payload.get("n_races", 0),
                    "combined_slope": payload.get("prediction_slope"),
                    "team_target_slope": payload.get("team_target_slope"),
                    "rmse_s": payload.get("rmse_s"),
                    "outside_1se": payload.get("outside_historical_one_se_band"),
                }
            )
    if rows:
        st_module.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st_module.info(str(construct_audit.get("reason", "No construct audit rows available.")))

    residual_rows = construct_audit.get("largest_abs_residual_rows", [])
    if isinstance(residual_rows, list) and residual_rows:
        st_module.caption("Largest absolute residual rows")
        st_module.dataframe(
            pd.DataFrame(residual_rows)[
                [
                    "session_kind",
                    "race_name",
                    "team",
                    "driver_code",
                    "observed_driver_to_field_s",
                    "predicted_driver_to_field_s",
                    "residual_s",
                    "n_construct_laps",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


def _render_refit_candidate_test(
    refit_test: Mapping[str, Any] | None,
    *,
    st_module: Any,
) -> None:
    """Render held-out refit-candidate metrics when the artifact exists."""
    if not refit_test:
        st_module.caption("No refit-candidate test artifact found yet.")
        return

    st_module.caption(
        f"Built {refit_test.get('built_at', 'unknown')} • "
        f"status `{refit_test.get('status', 'unknown')}`"
    )
    aggregate = refit_test.get("aggregate", [])
    if isinstance(aggregate, list) and aggregate:
        st_module.dataframe(pd.DataFrame(aggregate), width="stretch", hide_index=True)
    else:
        st_module.info(str(refit_test.get("reason", "No refit-candidate metrics available.")))

    decision = refit_test.get("decision_assessment", {})
    if isinstance(decision, Mapping):
        st_module.caption(
            f"Decision state: `{decision.get('state', 'unknown')}` — "
            f"{decision.get('recommendation', '')}"
        )


def _render_prediction_replay_test(
    replay_test: Mapping[str, Any] | None,
    *,
    st_module: Any,
) -> None:
    """Render full prediction replay comparison metrics when available."""
    if not replay_test:
        st_module.caption("No full prediction replay test artifact found yet.")
        return

    st_module.caption(
        f"Built {replay_test.get('built_at', 'unknown')} • "
        f"status `{replay_test.get('status', 'unknown')}`"
    )
    race_target_aggregate = replay_test.get("race_target_aggregate", [])
    if isinstance(race_target_aggregate, list) and race_target_aggregate:
        st_module.dataframe(
            pd.DataFrame(race_target_aggregate),
            width="stretch",
            hide_index=True,
        )
    else:
        st_module.info(str(replay_test.get("reason", "No prediction replay metrics available.")))

    decision = replay_test.get("decision_assessment", {})
    if isinstance(decision, Mapping):
        st_module.caption(
            f"Decision state: `{decision.get('state', 'unknown')}` — "
            f"{decision.get('recommendation', '')}"
        )


def _render_warning_block(warnings: Any, *, st_module: Any) -> None:
    """Render artifact warnings compactly."""
    if not isinstance(warnings, list) or not warnings:
        return
    for warning in warnings:
        st_module.warning(str(warning))


def _fmt(value: Any) -> str:
    """Format optional metric values for dashboard display."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.3f}"

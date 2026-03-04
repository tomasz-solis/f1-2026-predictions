"""Rendering helpers for prediction-accuracy dashboard page."""

from typing import Any

import pandas as pd
import streamlit as st

_SESSION_ORDER = {
    "FP1": 1,
    "FP2": 2,
    "FP3": 3,
    "SQ": 4,
    "SPRINT": 5,
    "Q": 6,
    "R": 7,
}


def render_overall_accuracy_metrics(agg_metrics: dict[str, Any]) -> None:
    """Render aggregate qualifying/race accuracy metrics."""
    st.markdown("---")
    st.subheader("Overall Accuracy")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Qualifying Metrics**")
        if "qualifying" in agg_metrics:
            q_metrics = agg_metrics["qualifying"]
            st.metric(
                "Exact Position Accuracy",
                f"{q_metrics['exact_accuracy']['mean']:.1f}%",
                help="% of drivers predicted in exact correct position",
            )
            st.metric(
                "Mean Position Error (MAE)",
                f"{q_metrics['mae']['mean']:.2f} positions",
                help="Average position error",
            )
            st.metric(
                "Within ±3 Positions",
                f"{q_metrics['within_3']['mean']:.1f}%",
                help="% of predictions within 3 positions",
            )
            st.metric(
                "Correlation",
                f"{q_metrics['correlation']['mean']:.3f}",
                help="Spearman correlation (-1 to 1, higher is better)",
            )

    with col2:
        st.markdown("**Race Metrics**")
        if "race" in agg_metrics:
            r_metrics = agg_metrics["race"]
            st.metric(
                "Exact Position Accuracy",
                f"{r_metrics['exact_accuracy']['mean']:.1f}%",
                help="% of drivers predicted in exact correct position",
            )
            st.metric(
                "Mean Position Error (MAE)",
                f"{r_metrics['mae']['mean']:.2f} positions",
                help="Average position error",
            )
            st.metric(
                "Within ±3 Positions",
                f"{r_metrics['within_3']['mean']:.1f}%",
                help="% of predictions within 3 positions",
            )
            st.metric(
                "Winner Prediction Accuracy",
                f"{r_metrics['winner_accuracy']['percentage']:.1f}%",
                help="% of races where winner was correctly predicted",
            )


def render_per_race_breakdown(
    predictions_with_actuals: list[dict[str, Any]],
    metrics_calc: Any,
) -> None:
    """Render per-race detailed metrics for predictions with actual outcomes."""
    st.markdown("---")
    st.subheader("Per-Race Breakdown")

    for prediction in predictions_with_actuals:
        metrics = metrics_calc.calculate_all_metrics(prediction)
        if not metrics:
            continue

        race_name = metrics["metadata"]["race_name"]
        session_name = metrics["metadata"]["session_name"]

        with st.expander(f"{race_name} (Predicted after {session_name})"):
            col1, col2 = st.columns(2)

            with col1:
                if "qualifying" in metrics:
                    qualifying_metrics = metrics["qualifying"]
                    st.markdown("**Qualifying**")
                    st.write(f"- Exact: {qualifying_metrics['exact_accuracy']:.1f}%")
                    st.write(f"- MAE: {qualifying_metrics['mae']:.2f} positions")
                    st.write(f"- Within ±1: {qualifying_metrics['within_1']:.1f}%")
                    st.write(f"- Correlation: {qualifying_metrics['correlation']:.3f}")

            with col2:
                if "race" in metrics:
                    race_metrics = metrics["race"]
                    st.markdown("**Race**")
                    st.write(f"- Exact: {race_metrics['exact_accuracy']:.1f}%")
                    st.write(f"- MAE: {race_metrics['mae']:.2f} positions")
                    st.write(f"- Within ±3: {race_metrics['within_3']:.1f}%")
                    st.write(
                        f"- Winner: {'Correct' if race_metrics['winner_correct'] else 'Incorrect'}"
                    )
                    st.write(
                        f"- Podium: {race_metrics['podium']['correct_drivers']}/3 drivers correct"
                    )


def render_saved_predictions_summary(all_predictions: list[dict[str, Any]]) -> None:
    """Render status list of all saved predictions."""
    st.markdown("---")
    st.subheader("All Saved Predictions")

    for prediction in all_predictions:
        metadata = prediction["metadata"]
        race_name = metadata["race_name"]
        session_name = metadata["session_name"]
        has_actuals = bool(
            prediction.get("actuals")
            and (prediction["actuals"].get("qualifying") or prediction["actuals"].get("race"))
        )

        status_text = "Results added" if has_actuals else "Awaiting results"
        st.write(f"**{race_name}** (after {session_name}) - {status_text}")


def _average_prediction_confidence(entries: Any) -> float | None:
    """Return mean confidence for prediction rows that expose a numeric confidence field."""
    if not isinstance(entries, list):
        return None
    confidences: list[float] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_value = entry.get("confidence")
        if raw_value is None:
            continue
        try:
            confidences.append(float(raw_value))
        except (TypeError, ValueError):
            continue
    if not confidences:
        return None
    return float(sum(confidences) / len(confidences))


def render_checkpoint_accuracy_trend(
    predictions_with_actuals: list[dict[str, Any]],
    metrics_calc: Any,
) -> None:
    """Render checkpoint-level trend table for confidence and realized accuracy."""
    rows: list[dict[str, Any]] = []

    for prediction in predictions_with_actuals:
        metrics = metrics_calc.calculate_all_metrics(prediction)
        if not metrics:
            continue

        metadata = prediction.get("metadata", {})
        race_name = str(metadata.get("race_name", "")).strip()
        checkpoint = str(metadata.get("session_name", "")).strip().upper()
        if not race_name or not checkpoint:
            continue

        quali_prediction = (prediction.get("qualifying") or {}).get("predicted_grid", [])
        race_prediction = (prediction.get("race") or {}).get("predicted_results", [])
        qualifying_metrics = metrics.get("qualifying") or {}
        race_metrics = metrics.get("race") or {}

        rows.append(
            {
                "Race": race_name,
                "Checkpoint": checkpoint,
                "Quali Confidence %": _average_prediction_confidence(quali_prediction),
                "Race Confidence %": _average_prediction_confidence(race_prediction),
                "Quali Exact %": qualifying_metrics.get("exact_accuracy"),
                "Race Exact %": race_metrics.get("exact_accuracy"),
                "_order": _SESSION_ORDER.get(checkpoint, 99),
            }
        )

    if not rows:
        return

    details_df = pd.DataFrame(rows).sort_values(["_order", "Race"]).drop(columns=["_order"])
    stage_df = (
        pd.DataFrame(rows)
        .groupby(["Checkpoint", "_order"], as_index=False)[
            [
                "Quali Confidence %",
                "Race Confidence %",
                "Quali Exact %",
                "Race Exact %",
            ]
        ]
        .mean(numeric_only=True)
        .sort_values("_order")
        .drop(columns=["_order"])
    )
    stage_df = stage_df.round(2)

    st.markdown("---")
    st.subheader("Checkpoint Trend")
    st.caption(
        "Mean values by prediction checkpoint. Use this view to track how confidence and "
        "realized accuracy move from early sessions to qualifying/race."
    )
    st.dataframe(stage_df, width="stretch", hide_index=True)

    with st.expander("Per-race checkpoint details"):
        st.dataframe(details_df.round(2), width="stretch", hide_index=True)

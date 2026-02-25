"""Rendering helpers for prediction-accuracy dashboard page."""

from typing import Any

import streamlit as st


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

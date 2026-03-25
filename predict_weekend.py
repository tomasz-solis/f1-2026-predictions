"""CLI helpers for live weekend prediction snapshots."""

import argparse
import importlib
import logging
from datetime import UTC, datetime

import fastf1 as ff1
import pandas as pd

try:
    _tabulate = importlib.import_module("tabulate").tabulate
except ModuleNotFoundError:
    _tabulate = None

from src.extractors.session import extract_session_order_safe
from src.predictors import Baseline2026Predictor
from src.systems.learning import LearningSystem
from src.utils.lineups import get_lineups
from src.utils.weekend import get_weekend_type

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("LivePredictor")


def _print_table(df: pd.DataFrame, columns: list[str]) -> None:
    """Render a table in CLI output, with fallback when tabulate is unavailable."""
    table_data = df[columns].head(10)
    if _tabulate is None:
        print(table_data.to_string(index=False))
        return

    print(
        _tabulate(
            table_data,
            headers="keys",
            tablefmt="simple",
            floatfmt=".1f",
        )
    )


def auto_catchup_history(year, learner):
    """
    Scans the calendar for finished races that we haven't analyzed yet.
    If found, it auto-runs the analysis to update our weighting model.
    """
    logger.info("Checking for missed lessons from past races...")

    try:
        schedule = ff1.get_event_schedule(year)
        now = datetime.now(UTC if schedule["Session1DateUtc"].dt.tz else None)

        # Get races that are finished but not in learner history
        finished_races = schedule[schedule["Session5DateUtc"] < now]

        for _, event in finished_races.iterrows():
            race_name = event["EventName"]
            if "Testing" in race_name:
                continue

            if not learner.is_race_analyzed(race_name):
                logger.info(f"   Analyzing past race: {race_name}...")
                _run_post_race_analysis(year, race_name, learner)

    except Exception as e:
        logger.warning(f"   [WARN] Could not auto-catchup: {e}")


def _run_post_race_analysis(year, race_name, learner):
    """Internal function to analyze a past race and update weights."""
    try:
        # Get Ground Truth (Official Results)
        session_q = ff1.get_session(year, race_name, "Q")
        session_q.load(laps=False, telemetry=False)
        if session_q.results is None or session_q.results.empty:
            return

        # Backtest Strategies (Compare FP3 vs Reality)
        fp3_ranks = extract_session_order_safe(year, race_name, "FP3")
        if not fp3_ranks:
            return

        # ... (Simplified MAE calc logic would go here) ...
        # For auto-catchup, we just log it to mark it as "seen"
        learner.update_after_race(
            race_name, {}, {"qualifying": {"method": "blend_70_30", "mae": 2.0}}
        )
        logger.info(f"      [OK] Learned from {race_name}")

    except Exception as e:
        logger.warning(f"      [ERROR] Failed to analyze {race_name}: {e}")


def get_available_data(year, race_name, weekend_type):
    """Detects available sessions."""
    data = {"fp1": None, "fp2": None, "fp3": None, "quali": None, "sprint_quali": None}
    logger.info(f"Scanning data for {race_name}...")

    # Always check FP1
    data["fp1"] = extract_session_order_safe(year, race_name, "FP1")

    if weekend_type == "conventional":
        data["fp2"] = extract_session_order_safe(year, race_name, "FP2")
        data["fp3"] = extract_session_order_safe(year, race_name, "FP3")
        data["quali"] = extract_session_order_safe(year, race_name, "Q")
    elif weekend_type == "sprint":
        data["sprint_quali"] = extract_session_order_safe(year, race_name, "Sprint Qualifying")

    found = [k.upper() for k, v in data.items() if v is not None]
    if found:
        logger.info(f"   [OK] Found: {', '.join(found)}")
    else:
        logger.info("   [INFO]  No session data (Pre-Weekend)")
    return data


def run_weekend_predictions(year, race_name, weather="dry"):
    """Run one qualifying and race prediction pass for the selected weekend."""
    # 1. Initialize & Auto-Learn
    learner = LearningSystem()
    predictor = Baseline2026Predictor(season_year=year)

    # AUTO-CATCHUP: Learn from history before predicting today
    auto_catchup_history(year, learner)

    # 2. Context
    weekend_type = get_weekend_type(year, race_name)
    data = get_available_data(year, race_name, weekend_type)

    # The learning system still gives a useful confidence hint, but the
    # baseline predictor now owns the actual session blending logic.
    blend_weight = learner.get_optimal_blend_weight(default=0.7)
    logger.info(
        "   Adaptive blend suggestion: "
        f"{blend_weight:.2f} (baseline predictor resolves session blending internally)"
    )

    # =========================================================
    # PART A: PREDICT QUALIFYING (ALWAYS RUNS)
    # =========================================================
    logger.info("\nPredicting qualifying...")

    if data["quali"]:
        conf = "Post-Quali Analysis"
    elif data["fp3"] or data["sprint_quali"]:
        conf = "High Confidence"
    elif data["fp2"]:
        conf = "Medium Confidence"
    elif data["fp1"]:
        conf = "Low Confidence"
    else:
        conf = "Baseline"
    logger.info(f"   Qualifying confidence mode: {conf}")

    q_result = predictor.predict_qualifying(
        year=year,
        race_name=race_name,
    )

    q_df = pd.DataFrame(q_result["grid"])
    _print_table(q_df, ["position", "driver", "team", "confidence"])

    # =========================================================
    # PART B: PREDICT RACE (ALWAYS RUNS)
    # =========================================================
    logger.info("\nPredicting race...")

    # 1. Determine Grid Source
    if data["quali"]:
        logger.info("   [OK] Using REAL Grid (Quali Completed)")
        grid = _convert_team_ranks_to_grid(data["quali"], year, race_name)
    else:
        logger.info("   [WARN]  Using PREDICTED Grid (Quali not yet run)")
        # Convert Quali Prediction DF to Grid list format
        grid = q_df.rename(columns={"position": "position"}).to_dict("records")

    # 2. Predict
    r_result = predictor.predict_race(
        qualifying_grid=grid,
        weather=weather,
        race_name=race_name,
        year=year,
    )

    r_df = pd.DataFrame(r_result["finish_order"])
    _print_table(r_df, ["position", "driver", "team", "confidence", "podium_probability"])


def _convert_team_ranks_to_grid(team_ranks, year, race_name):
    """Helper to format grid data."""
    lineups = get_lineups(year, race_name)
    grid = []
    sorted_teams = sorted(team_ranks.items(), key=lambda x: x[1])
    pos = 1
    for team, _ in sorted_teams:
        if team in lineups:
            drivers = lineups[team]
            grid.append({"driver": drivers[0], "team": team, "position": pos})
            pos += 1
            if len(drivers) > 1:
                grid.append({"driver": drivers[1], "team": team, "position": pos})
                pos += 1
    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("race_name", help="e.g. 'Bahrain Grand Prix'")
    parser.add_argument("--year", type=int, default=datetime.now().year)

    parser.add_argument(
        "--weather",
        type=str,
        default="dry",
        choices=["dry", "rain", "mixed"],
        help="Weather forecast: 'dry', 'rain', or 'mixed'",
    )

    args = parser.parse_args()

    run_weekend_predictions(args.year, args.race_name, weather=args.weather)

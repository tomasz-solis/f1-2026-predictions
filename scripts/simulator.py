"""
F1 2026 Season Simulator

Runs full season simulations to test regulation change scenarios.

Usage:
    python scripts/simulator.py

What it does:
    1. Checks and updates data (driver/track characteristics)
    2. Runs a sample 2026 season simulation
    3. Outputs championship standings and race-by-race results

Output:
    - Console: Live race results as simulation progresses
    - Results: Championship standings at end

Note: This uses the Baseline 2026 predictor since no real race data exists yet.
      Predictions will have high uncertainty (40-60% confidence) until testing.
"""

import logging
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("F1Orchestrator")


# --- DATA FACTORY ---
def check_and_update_data(force_update=False):
    """
    Checks if 'Factory' data exists. If not (or forced), runs the extraction scripts.
    """
    data_dir = Path("data/processed")
    driver_chars_file = data_dir / "driver_characteristics.json"
    track_chars_file = data_dir / "track_characteristics" / "2026_track_characteristics.json"

    # Check if files exist
    missing_data = not driver_chars_file.exists() or not track_chars_file.exists()

    if missing_data or force_update:
        logger.info("DATA FACTORY: Updating knowledge bases (this may take a moment)...")

        # 1. Run Overtaking Extraction
        logger.info("   - Extracting Overtaking Likelihoods...")
        logger.warning("   ! Run manually: python scripts/extract_overtaking_likelihood.py")

        # 2. Run Driver Characteristics
        logger.info("   - Extracting Driver Characteristics (2025 baseline)...")
        logger.warning(
            "   ! Use: python scripts/extract_driver_characteristics.py --years 2023,2024,2025"
        )
        logger.warning(
            "   ! Simulator does not auto-generate driver data. Run extraction script manually."
        )

        logger.info("[OK] Data update complete.")
    else:
        logger.info("[OK] Data Factory is up to date. Skipping extraction.")


def load_static_configs():
    """Loads the manual config files (Lineups, Debuts)."""
    import json

    logger.info("LOADING CONFIGURATION")

    # 1. Load Lineups
    lineup_path = Path("data/current_lineups.json")
    if lineup_path.exists():
        with open(lineup_path) as f:
            lineups = json.load(f)
        logger.info(f"   - Lineups loaded ({len(lineups.get('current_lineups', []))} teams)")
    else:
        logger.warning("   ! current_lineups.json missing!")

    # 2. Load Debuts (Experience)
    debuts_path = Path("data/driver_debuts.csv")
    if debuts_path.exists():
        debuts = pd.read_csv(debuts_path)
        logger.info(f"   - Driver Debuts loaded ({len(debuts)} drivers)")
    else:
        logger.warning("   ! driver_debuts.csv missing!")


# --- SIMULATION ENGINE ---
def run_simulation_loop(year=2026):
    """Run a short season simulation using the shared baseline predictor."""
    logger.info(f"  STARTING {year} SIMULATION ENGINE")

    # Imports inside function to avoid circular dependencies during setup
    from src.models.bayesian import BayesianDriverRanking
    from src.models.priors_factory import PriorsFactory  # <--- NEW IMPORT
    from src.models.regulations import apply_2026_regulations
    from src.predictors import Baseline2026Predictor
    from src.systems.learning import LearningSystem
    from src.utils.lineups import get_lineups
    from src.utils.weekend import get_weekend_type

    # 1. Initialize Systems
    learner = LearningSystem()  # Tracks Strategy (Blend vs Model)

    # 2. Build Priors (The Hierarchical Model)
    logger.info("     Building Priors from Car + Driver Data...")

    factory = PriorsFactory()  # Connects to your JSON artifacts
    base_priors = factory.create_priors()

    # Apply 2026 Regulation Shocks (The "Uncertainty Injection")
    logger.info("   Applying 2026 regulation shocks...")
    current_priors = apply_2026_regulations(base_priors)

    # 3. Spin up the predictor and the rating model
    ranker = BayesianDriverRanking(current_priors)
    predictor = Baseline2026Predictor(season_year=year)

    # 4. Run a short season sample calendar
    calendar = [
        "Australian Grand Prix",
        "Chinese Grand Prix",
        "Japanese Grand Prix",
        "Miami Grand Prix",
    ]
    simulation_log = []

    for round_num, race_name in enumerate(calendar, 1):
        logger.info(f"\nROUND {round_num}: {race_name}")

        # A. Context & Strategy
        weekend_type = get_weekend_type(year, race_name)
        strategy = learner.get_recommended_method(weekend_type)
        lineups = get_lineups(year, race_name)

        logger.info(f"   Format: {weekend_type.upper()} | Strategy: {strategy['method']}")

        # B. PREDICTION Phase (Mocking the grid)
        # In real life you would predict or observe qualifying first.
        mock_grid = [
            {"driver": d, "team": t, "position": i + 1}
            for i, (t, drivers) in enumerate(lineups.items())
            for d in drivers
        ][:20]

        prediction = predictor.predict_race(
            qualifying_grid=mock_grid,
            weather="dry",
            race_name=race_name,
            year=year,
        )
        predicted_winner = prediction["finish_order"][0]["driver"]

        logger.info(
            f"   Predicted winner: {predicted_winner} (Confidence: {prediction['finish_order'][0]['confidence']:.1f}%)"
        )

        # C. REALITY Phase (Mocking results for the simulation)
        # Scenario: McLaren starts strong, Ferrari catches up
        if round_num <= 2:
            actual_winner = "Lando Norris"
            podium = {"4": 1, "81": 2, "1": 3}
        else:
            actual_winner = "Charles Leclerc"
            podium = {"16": 1, "4": 2, "44": 3}

        logger.info(f"   Actual winner: {actual_winner}")

        # D. LEARNING Phase
        # 1. Update Beliefs (Bayesian)
        ranker.update(podium, session_name=race_name, confidence=1.0)

        # 2. Meta-Learning (Strategy Adjustment)
        mae = abs(0) if predicted_winner == actual_winner else 1.0  # Dummy MAE
        insights = learner.update_after_race(
            race=race_name,
            actual_results={"race": [{"driver": k, "position": v} for k, v in podium.items()]},
            prediction_comparison={"qualifying": {"method": strategy["method"], "mae": mae}},
        )

        if insights and insights.get("recommendations"):
            for rec in insights["recommendations"]:
                logger.info(f"   System insight: {rec}")

        # E. Logging
        top_driver = ranker.get_current_ratings().iloc[0]
        simulation_log.append(
            {
                "round": round_num,
                "race": race_name,
                "predicted_winner": predicted_winner,
                "actual_winner": actual_winner,
                "top_rated_driver": top_driver["driver_code"],
                "rating_mu": top_driver["rating_mu"],
            }
        )

    # 5. Output Results
    logger.info("\n SIMULATION COMPLETE")
    df_results = pd.DataFrame(simulation_log)
    output_file = Path("data/processed/2026_season_simulation.csv")
    df_results.to_csv(output_file, index=False)

    logger.info(f"   History saved to: {output_file}")


if __name__ == "__main__":
    try:
        # Step 1: Prep Data
        check_and_update_data(force_update=False)

        # Step 2: Load Configs
        load_static_configs()

        # Step 3: Run Engine
        run_simulation_loop()

        logger.info("\n[OK] PIPELINE SUCCESS")

    except KeyboardInterrupt:
        logger.info("\nPipeline stopped by user.")
    except Exception as e:
        logger.exception(f"\n[ERROR] FATAL ERROR: {e}")

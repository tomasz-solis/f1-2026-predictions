"""
Production Config Helper

Uses historical testing results to select best method.
NO hardcoded performance values!
"""

import json
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


class ProductionConfig:
    """
    Load and use production configuration.

    Based on historical testing (Notebook 21B, 24 races).
    """

    def __init__(self, config_path="config/production_config.json"):
        """Load production config."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Production config not found at {config_path}\n"
                "Run historical testing (Notebook 21B) to generate it."
            )

        with open(config_file) as f:
            self.config = json.load(f)

    def get_qualifying_strategy(self, weekend_type: Literal["sprint", "conventional"]) -> dict:
        """Get best qualifying prediction strategy for the given weekend type."""
        quali_config = self.config["qualifying_methods"]

        if weekend_type == "sprint":
            return quali_config["sprint_weekends"].copy()
        else:
            return quali_config["conventional_weekends"].copy()

    def __str__(self):
        """Display config summary."""
        lines = []
        notes = self.config.get("notes", {})
        source_notebook = next(
            (value for key, value in notes.items() if "testing_notebook" in key),
            "N/A",
        )
        lines.append("PRODUCTION CONFIGURATION")
        lines.append("")
        lines.append(f"Source: {source_notebook}")
        lines.append(f"Races analyzed: {notes.get('total_races_analyzed', 'N/A')}")
        lines.append(f"Last updated: {notes.get('last_updated', 'N/A')}")
        lines.append("")

        lines.append("QUALIFYING STRATEGY:")

        sprint = self.config["qualifying_methods"]["sprint_weekends"]
        lines.append("  Sprint weekends:")
        lines.append(f"    Method: {sprint['method']}")
        lines.append(f"    Session: {sprint.get('session', 'N/A')}")
        lines.append(f"    Expected MAE: {sprint['expected_mae']:.2f}")
        lines.append(f"    Confidence: {sprint['confidence']}")

        conv = self.config["qualifying_methods"]["conventional_weekends"]
        lines.append("  Conventional weekends:")
        lines.append(f"    Method: {conv['method']}")
        lines.append(f"    Blend weight: {conv.get('blend_weight', 'N/A')}")
        lines.append(f"    Expected MAE: {conv['expected_mae']:.2f}")
        lines.append(f"    Confidence: {conv['confidence']}")

        lines.append("")
        lines.append("PERFORMANCE RANKING (2025):")
        for rank, info in notes.get("performance_ranking_2025", {}).items():
            lines.append(f"  {rank}. {info}")

        return "\n".join(lines)


# Quick helper functions
def load_production_config(
    config_path="config/production_config.json",
) -> ProductionConfig:
    """Load production config."""
    return ProductionConfig(config_path)

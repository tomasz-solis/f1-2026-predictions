"""Legacy adaptive-learning state for historical blend recommendations.

This module predates the current checkpoint-aware prediction flow. It still
backs a few scripts and tests, but it is no longer the main production learning
path. The current dashboard relies on ``SystematicLearningSystem`` instead.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class LearningSystem:
    """Store lightweight historical MAE summaries for legacy callers."""

    def __init__(self, data_dir="data/systems"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "learning_state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load persisted state if it exists, otherwise return an empty shell."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "history": [],  # List of races already analyzed
            "method_performance": {},
            "last_updated": None,
        }

    def save_state(self):
        """Persist the current learning snapshot to disk."""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def is_race_analyzed(self, race_name: str) -> bool:
        """Check if we have already learned from this race."""
        for record in self.state["history"]:
            if record["race"] == race_name:
                return True
        return False

    def get_optimal_blend_weight(self, default=0.7) -> float:
        """Returns the best blend weight (0.0-1.0) based on history."""
        best_mae = float("inf")
        best_method = None

        for method, stats in self.state["method_performance"].items():
            if "blend" in method and stats.get("avg_mae") is not None:
                if stats["avg_mae"] < best_mae:
                    best_mae = stats["avg_mae"]
                    best_method = method

        if best_method:
            try:
                # Format: 'blend_70_30' -> 0.7
                parts = best_method.split("_")
                if len(parts) >= 2:
                    return int(parts[1]) / 100.0
            except (ValueError, IndexError) as exc:
                logger.debug("Could not parse feedback value: %s", exc)
        return default

    def get_recommended_method(
        self, weekend_type: str = "conventional", default_blend_weight: float = 0.7
    ) -> dict:
        """
        Return a compatibility strategy payload for legacy callers.

        The method label follows `blend_XX_YY`, where XX is session data weight.
        """
        blend_weight = self.get_optimal_blend_weight(default=default_blend_weight)
        blend_pct = int(round(blend_weight * 100))
        model_pct = max(0, 100 - blend_pct)

        return {
            "weekend_type": weekend_type,
            "method": f"blend_{blend_pct}_{model_pct}",
            "blend_weight": blend_weight,
        }

    def update_after_race(
        self, race: str, actual_results: dict, prediction_comparison: dict
    ) -> dict:
        """Update stats after a race weekend."""
        self.state["history"].append(
            {
                "race": race,
                "date": datetime.now().isoformat(),
                "comparisons": prediction_comparison,
            }
        )

        for _, data in prediction_comparison.items():
            method = data.get("method")
            mae = data.get("mae")

            if method and mae is not None:
                if method not in self.state["method_performance"]:
                    self.state["method_performance"][method] = {
                        "count": 0,
                        "avg_mae": 0.0,
                        "history": [],
                    }

                perf = self.state["method_performance"][method]
                perf["history"].append(mae)
                # Keep last 5 for responsiveness
                recent = perf["history"][-5:]
                perf["avg_mae"] = sum(recent) / len(recent)
                perf["count"] += 1

        self.save_state()

        recommended = self.get_recommended_method()
        return {
            "recommendations": [
                (
                    "Best recent blend weight: "
                    f"{recommended['blend_weight']:.2f} ({recommended['method']})"
                )
            ]
        }

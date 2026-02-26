"""Saves predictions to JSON for accuracy tracking."""

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.persistence.artifact_store import ArtifactStore
from src.systems.systematic_learning import SystematicLearningSystem

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Handles saving and loading race predictions for accuracy tracking."""

    def __init__(self, predictions_dir: str = "data/predictions"):
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ArtifactStore
        # Note: ArtifactStore expects data_root, and will append "/predictions" internally
        # So if predictions_dir is "data/predictions", data_root should be "data"
        # For temp dirs in tests, we need to point to parent so ArtifactStore creates
        # the predictions subdirectory in the right place
        data_root = self.predictions_dir.parent
        self.artifact_store = ArtifactStore(data_root=data_root)
        self.learning_system = SystematicLearningSystem(
            state_file=data_root / "learning_state.json"
        )

    def save_prediction(
        self,
        year: int,
        race_name: str,
        session_name: str,
        qualifying_prediction: list[dict[str, Any]],
        race_prediction: list[dict[str, Any]],
        weather: str,
        fp_blend_info: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> Path:
        """Save a prediction to disk with validation and run-based versioning."""
        # Validate inputs
        if not qualifying_prediction or not race_prediction:
            raise ValueError("Predictions cannot be empty")

        # Validate required fields in predictions
        for i, pred in enumerate(qualifying_prediction):
            if "driver" not in pred or "team" not in pred:
                raise ValueError(f"Qualifying prediction {i} missing 'driver' or 'team' field")

        for i, pred in enumerate(race_prediction):
            if "driver" not in pred or "team" not in pred:
                raise ValueError(f"Race prediction {i} missing 'driver' or 'team' field")

        # Validate weather
        valid_weather = ["dry", "rain", "mixed", "wet"]
        if weather not in valid_weather:
            logger.warning(f"Unusual weather value: {weather}. Expected one of {valid_weather}")

        # Generate run_id if not provided (for run-based versioning)
        if run_id is None:
            run_id = str(uuid.uuid4())

        # Sanitize race name for filename
        safe_race_name = race_name.lower().replace(" ", "_").replace("'", "")

        # Create year directory (for file fallback)
        year_dir = self.predictions_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        # Create race directory
        race_dir = year_dir / safe_race_name
        race_dir.mkdir(parents=True, exist_ok=True)

        # Filename includes session to allow multiple predictions per race
        filename = f"{safe_race_name}_{session_name.lower()}.json"
        filepath = race_dir / filename

        # Build prediction data structure
        prediction_data = {
            "metadata": {
                "run_id": run_id,
                "year": year,
                "race_name": race_name,
                "session_name": session_name,
                "predicted_at": datetime.now(UTC).isoformat(),
                "weather": weather,
                "fp_blend_info": fp_blend_info or {},
                **({} if metadata is None else metadata),
            },
            "qualifying": {
                "predicted_grid": [
                    {
                        "position": i + 1,
                        "driver": result["driver"],
                        "team": result["team"],
                        "expected_time": result.get("expected_time"),
                        "confidence": result.get("confidence"),
                    }
                    for i, result in enumerate(qualifying_prediction)
                ]
            },
            "race": {
                "predicted_results": [
                    {
                        "position": i + 1,
                        "driver": result["driver"],
                        "team": result["team"],
                        "confidence": result.get("confidence"),
                        "dnf_risk": result.get("dnf_risk"),
                    }
                    for i, result in enumerate(race_prediction)
                ]
            },
            # Actual results are appended later by the post-session reconciliation flow.
            "actuals": {"qualifying": None, "race": None},
        }

        # Save via ArtifactStore (dual-write mode if configured)
        artifact_key = f"{year}::{race_name}::{session_name}"
        try:
            self.artifact_store.save_artifact(
                artifact_type="prediction",
                artifact_key=artifact_key,
                data=prediction_data,
                version=1,
                run_id=run_id,
            )
            logger.info(f"Saved prediction via ArtifactStore (run_id={run_id})")
        except Exception as e:
            logger.warning(f"ArtifactStore save failed: {e}, falling back to file")
            with open(filepath, "w") as f:
                json.dump(prediction_data, f, indent=2)
            logger.info(f"Saved prediction to {filepath}")

        return filepath

    def load_prediction(
        self, year: int, race_name: str, session_name: str, run_id: str | None = None
    ) -> dict[str, Any] | None:
        """Load a saved prediction with schema validation (DB-first, file fallback)."""
        # Try loading from ArtifactStore (latest run by default)
        artifact_key = f"{year}::{race_name}::{session_name}"
        try:
            data = self.artifact_store.load_artifact(
                artifact_type="prediction",
                artifact_key=artifact_key,
                version="latest",
                run_id=run_id,
            )

            if data:
                # Validate schema
                if self._validate_prediction_schema(data):
                    return data
                else:
                    logger.error(f"Invalid prediction schema from DB for {artifact_key}")

        except Exception as e:
            logger.warning(f"ArtifactStore load failed: {e}, trying file fallback")

        # Fallback to file
        safe_race_name = race_name.lower().replace(" ", "_").replace("'", "")
        filepath = (
            self.predictions_dir
            / str(year)
            / safe_race_name
            / f"{safe_race_name}_{session_name.lower()}.json"
        )

        if not filepath.exists():
            logger.warning(f"Prediction not found in file: {filepath}")
            return None

        try:
            with open(filepath) as f:
                data = json.load(f)

            if self._validate_prediction_schema(data):
                return data
            else:
                logger.error(f"Invalid prediction schema in file {filepath}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted JSON file {filepath}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load prediction from {filepath}: {e}")
            return None

    def _validate_prediction_schema(self, data: dict[str, Any]) -> bool:
        """Validate prediction data structure."""
        # Validate required keys
        required_keys = ["metadata", "qualifying", "race", "actuals"]
        if not all(key in data for key in required_keys):
            return False

        # Validate metadata
        if not isinstance(data["metadata"], dict):
            return False

        # Validate prediction lists
        if "predicted_grid" not in data["qualifying"]:
            return False

        if "predicted_results" not in data["race"]:
            return False

        return True

    def update_actuals(
        self,
        year: int,
        race_name: str,
        session_name: str,
        qualifying_results: list[dict[str, Any]] | None = None,
        race_results: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
    ) -> bool:
        """Update a saved prediction with actual race results."""
        prediction = self.load_prediction(year, race_name, session_name, run_id=run_id)
        if prediction is None:
            return False

        # Update actuals
        if qualifying_results is not None:
            prediction["actuals"]["qualifying"] = [
                {"position": i + 1, "driver": r["driver"], "team": r["team"]}
                for i, r in enumerate(qualifying_results)
            ]

        if race_results is not None:
            prediction["actuals"]["race"] = [
                {"position": i + 1, "driver": r["driver"], "team": r["team"]}
                for i, r in enumerate(race_results)
            ]

        # Extract run_id from prediction metadata
        actual_run_id = run_id or prediction["metadata"].get("run_id")

        # Save updated prediction via ArtifactStore
        artifact_key = f"{year}::{race_name}::{session_name}"
        try:
            self.artifact_store.save_artifact(
                artifact_type="prediction",
                artifact_key=artifact_key,
                data=prediction,
                version=1,
                run_id=actual_run_id,
            )
            logger.info(f"Updated actuals via ArtifactStore (run_id={actual_run_id})")
        except Exception as e:
            logger.warning(f"ArtifactStore save failed: {e}, falling back to file")
            safe_race_name = race_name.lower().replace(" ", "_").replace("'", "")
            filepath = (
                self.predictions_dir
                / str(year)
                / safe_race_name
                / f"{safe_race_name}_{session_name.lower()}.json"
            )
            with open(filepath, "w") as f:
                json.dump(prediction, f, indent=2)
            logger.info(f"Updated actuals in {filepath}")

        try:
            learning_summary = self.learning_system.update_from_prediction_record(prediction)
            if learning_summary.get("sessions_updated", 0) > 0:
                logger.info(
                    "Updated adaptive calibration from actual results "
                    f"(sessions={learning_summary['sessions_updated']}, "
                    f"drivers={learning_summary['driver_updates']}, "
                    f"pairs={learning_summary['pair_updates']})"
                )
        except Exception as e:
            logger.warning(f"Could not update adaptive calibration from actuals: {e}")

        return True

    def get_all_predictions(self, year: int) -> list[dict[str, Any]]:
        """Load all predictions for a given year."""
        year_dir = self.predictions_dir / str(year)
        if not year_dir.exists():
            return []

        predictions = []
        for race_dir in year_dir.iterdir():
            if race_dir.is_dir():
                for pred_file in race_dir.glob("*.json"):
                    with open(pred_file) as f:
                        predictions.append(json.load(f))

        return predictions

    def has_prediction_for_session(self, year: int, race_name: str, session_name: str) -> bool:
        """Check if a prediction exists for a given session (DB-first, file fallback)."""
        # Try DB first
        artifact_key = f"{year}::{race_name}::{session_name}"
        try:
            data = self.artifact_store.load_artifact(
                artifact_type="prediction", artifact_key=artifact_key, version="latest"
            )
            if data:
                return True
        except Exception:
            pass

        # Fallback to file
        safe_race_name = race_name.lower().replace(" ", "_").replace("'", "")
        filepath = (
            self.predictions_dir
            / str(year)
            / safe_race_name
            / f"{safe_race_name}_{session_name.lower()}.json"
        )
        return filepath.exists()

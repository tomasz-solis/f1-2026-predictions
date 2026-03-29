"""Persist and reconcile saved predictions for later accuracy analysis."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.persistence.artifact_store import ArtifactStore
from src.systems.systematic_learning import SystematicLearningSystem
from src.types.prediction_types import QualifyingGridEntry
from src.utils.accuracy_targets import (
    explicit_target_actuals,
    explicit_target_predictions,
    fastf1_session_name,
    legacy_target_keys_for_prediction,
    sanitize_actual_rows,
    sanitize_prediction_rows,
    synthesize_legacy_actuals,
    synthesize_legacy_targets,
    target_session_name,
    weekend_format_name,
)

logger = logging.getLogger(__name__)

ActualResultRows = Sequence[Mapping[str, Any]]
FetchedSessionResults = list[QualifyingGridEntry]


class PredictionLogger:
    """Save, load, and enrich race-weekend prediction artifacts."""

    def __init__(self, predictions_dir: str = "data/predictions"):
        """Create a prediction logger rooted at the given predictions directory."""
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

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
        target_predictions: dict[str, dict[str, Any]] | None = None,
    ) -> Path:
        """Save a prediction payload in the configured persistence backends."""
        if not qualifying_prediction or not race_prediction:
            raise ValueError("Predictions cannot be empty")

        for index, prediction in enumerate(qualifying_prediction):
            if "driver" not in prediction or "team" not in prediction:
                raise ValueError(f"Qualifying prediction {index} missing 'driver' or 'team' field")
        for index, prediction in enumerate(race_prediction):
            if "driver" not in prediction or "team" not in prediction:
                raise ValueError(f"Race prediction {index} missing 'driver' or 'team' field")

        valid_weather = ["dry", "rain", "mixed", "wet"]
        if weather not in valid_weather:
            logger.warning("Unusual weather value: %s. Expected one of %s", weather, valid_weather)

        if run_id is None:
            run_id = str(uuid.uuid4())

        normalized_year, normalized_race_name, normalized_session_name = self._prediction_identity(
            year,
            race_name,
            session_name,
        )
        filepath = self._prediction_file_path(
            normalized_year,
            normalized_race_name,
            normalized_session_name,
        )

        normalized_targets = self._normalize_target_predictions(target_predictions)
        metadata_payload = {} if metadata is None else dict(metadata)
        if "weekend_format" not in metadata_payload:
            metadata_payload["weekend_format"] = self._infer_weekend_format(
                session_name=normalized_session_name,
                target_predictions=normalized_targets,
            )

        prediction_data = {
            "metadata": {
                "run_id": run_id,
                "year": normalized_year,
                "race_name": normalized_race_name,
                "session_name": normalized_session_name,
                "predicted_at": datetime.now(UTC).isoformat(),
                "weather": weather,
                "fp_blend_info": fp_blend_info or {},
                **metadata_payload,
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
            "targets": normalized_targets,
            "actuals": {
                "qualifying": None,
                "race": None,
                "targets": {target_key: None for target_key in normalized_targets},
            },
        }

        self._write_prediction_file(filepath, prediction_data)
        artifact_key = self._artifact_key_for_prediction(
            normalized_year,
            normalized_race_name,
            normalized_session_name,
        )
        try:
            self.artifact_store.save_artifact(
                artifact_type="prediction",
                artifact_key=artifact_key,
                data=prediction_data,
                version=1,
                run_id=run_id,
            )
            logger.info("Saved prediction via ArtifactStore and file fallback (run_id=%s)", run_id)
        except Exception as exc:
            logger.warning(
                "ArtifactStore save failed for prediction %s; kept file copy at %s (%s)",
                artifact_key,
                filepath,
                exc,
            )

        return filepath

    def _prediction_file_path(
        self,
        year: int,
        race_name: str,
        session_name: str,
    ) -> Path:
        """Build the normalized file path for one prediction payload."""
        safe_race_name = race_name.lower().replace(" ", "_").replace("'", "")
        return (
            self.predictions_dir
            / str(year)
            / safe_race_name
            / f"{safe_race_name}_{session_name.lower()}.json"
        )

    @staticmethod
    def _write_prediction_file(filepath: Path, prediction_data: dict[str, Any]) -> None:
        """Write one prediction payload to disk with stable formatting."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as file_handle:
            json.dump(prediction_data, file_handle, indent=2)

    def load_prediction(
        self,
        year: int,
        race_name: str,
        session_name: str,
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Load a saved prediction with schema validation."""
        normalized_year, normalized_race_name, normalized_session_name = self._prediction_identity(
            year,
            race_name,
            session_name,
        )
        artifact_key = self._artifact_key_for_prediction(
            normalized_year,
            normalized_race_name,
            normalized_session_name,
        )
        try:
            data = self.artifact_store.load_artifact(
                artifact_type="prediction",
                artifact_key=artifact_key,
                version="latest",
                run_id=run_id,
            )
            if data:
                if self._validate_prediction_schema(data):
                    return data
                logger.error("Invalid prediction schema from DB for %s", artifact_key)
        except Exception as exc:
            logger.warning("ArtifactStore load failed: %s, trying file fallback", exc)

        filepath = self._prediction_file_path(
            normalized_year,
            normalized_race_name,
            normalized_session_name,
        )
        if not filepath.exists():
            logger.warning("Prediction not found in file: %s", filepath)
            return None

        try:
            with open(filepath) as file_handle:
                data = json.load(file_handle)
            if self._validate_prediction_schema(data):
                return data
            logger.error("Invalid prediction schema in file %s", filepath)
            return None
        except json.JSONDecodeError as exc:
            logger.error("Corrupted JSON file %s: %s", filepath, exc)
            return None
        except Exception as exc:
            logger.error("Failed to load prediction from %s: %s", filepath, exc)
            return None

    def _validate_prediction_schema(self, data: dict[str, Any]) -> bool:
        """Validate the minimum prediction payload shape."""
        required_keys = ["metadata", "qualifying", "race", "actuals"]
        if not all(key in data for key in required_keys):
            return False
        if not isinstance(data["metadata"], dict):
            return False
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
        qualifying_results: ActualResultRows | None = None,
        race_results: ActualResultRows | None = None,
        run_id: str | None = None,
        target_actual_results: dict[str, ActualResultRows | None] | None = None,
    ) -> bool:
        """Attach actual results to a saved prediction."""
        prediction = self.load_prediction(year, race_name, session_name, run_id=run_id)
        if prediction is None:
            return False

        normalized_target_actuals = self._normalize_target_actual_results(target_actual_results)
        if not normalized_target_actuals:
            normalized_target_actuals = self._derive_target_actual_results_from_legacy_inputs(
                prediction,
                qualifying_results=qualifying_results,
                race_results=race_results,
            )

        actuals = prediction.setdefault("actuals", {})
        if not isinstance(actuals, dict):
            actuals = {}
            prediction["actuals"] = actuals
        actuals_targets = actuals.setdefault("targets", {})
        if not isinstance(actuals_targets, dict):
            actuals_targets = {}
            actuals["targets"] = actuals_targets

        for target_key, rows in normalized_target_actuals.items():
            actuals_targets[target_key] = rows

        if qualifying_results is not None and self._should_update_top_level_actual(
            prediction,
            session_type="qualifying",
        ):
            actuals["qualifying"] = [
                {"position": i + 1, "driver": row["driver"], "team": row["team"]}
                for i, row in enumerate(qualifying_results)
            ]

        if race_results is not None and self._should_update_top_level_actual(
            prediction,
            session_type="race",
        ):
            actuals["race"] = [
                {"position": i + 1, "driver": row["driver"], "team": row["team"]}
                for i, row in enumerate(race_results)
            ]

        actual_run_id = run_id or prediction["metadata"].get("run_id")
        normalized_year, normalized_race_name, normalized_session_name = self._prediction_identity(
            year,
            race_name,
            session_name,
        )
        artifact_key = self._artifact_key_for_prediction(
            normalized_year,
            normalized_race_name,
            normalized_session_name,
        )
        filepath = self._prediction_file_path(
            normalized_year,
            normalized_race_name,
            normalized_session_name,
        )
        self._write_prediction_file(filepath, prediction)
        try:
            self.artifact_store.save_artifact(
                artifact_type="prediction",
                artifact_key=artifact_key,
                data=prediction,
                version=1,
                run_id=actual_run_id,
            )
            logger.info(
                "Updated actuals via ArtifactStore and file fallback (run_id=%s)", actual_run_id
            )
        except Exception as exc:
            logger.warning(
                "ArtifactStore save failed while updating actuals for %s; kept file copy at %s (%s)",
                artifact_key,
                filepath,
                exc,
            )

        try:
            learning_summary = self.learning_system.update_from_prediction_record(prediction)
            if learning_summary.get("sessions_updated", 0) > 0:
                logger.info(
                    "Updated adaptive calibration from actual results "
                    "(sessions=%s, drivers=%s, pairs=%s)",
                    learning_summary["sessions_updated"],
                    learning_summary["driver_updates"],
                    learning_summary["pair_updates"],
                )
        except Exception as exc:
            logger.warning("Could not update adaptive calibration from actuals: %s", exc)

        return True

    def get_all_predictions(self, year: int) -> list[dict[str, Any]]:
        """Load season predictions from both storage backends and deduplicate them."""
        target_year = int(year)
        predictions: list[dict[str, Any]] = []

        try:
            artifact_rows = self.artifact_store.list_artifacts("prediction", limit=4096)
        except Exception as exc:
            logger.warning("Could not list prediction artifacts for %s: %s", year, exc)
            artifact_rows = []

        for row in artifact_rows:
            payload = self._prediction_payload_from_listing_row(row)
            if payload is None:
                continue

            metadata = payload.get("metadata", {})
            try:
                payload_year = int(metadata.get("year"))
            except (TypeError, ValueError):
                continue
            if payload_year != target_year:
                continue
            predictions.append(payload)

        predictions.extend(self._load_predictions_from_files(target_year))
        return self._deduplicate_predictions(predictions)

    def reconcile_completed_prediction_actuals(self, year: int) -> int:
        """Attach saved target actuals to completed race weekends for a season."""
        from src.data.actual_results_fetcher import fetch_actual_session_results
        from src.utils.session_detector import SessionDetector
        from src.utils.weekend import is_sprint_weekend

        target_year = int(year)
        predictions = self.get_all_predictions(target_year)
        if not predictions:
            return 0

        detector = SessionDetector()
        updated_predictions = 0
        race_completion_cache: dict[str, bool] = {}
        sprint_cache: dict[str, bool] = {}
        actual_cache: dict[tuple[str, str], FetchedSessionResults | None] = {}

        for prediction in predictions:
            metadata = prediction.get("metadata", {})
            race_name = str(metadata.get("race_name", "")).strip()
            session_name = str(metadata.get("session_name", "")).strip().upper()
            if not race_name or not session_name:
                continue

            if race_name not in race_completion_cache:
                race_completion_cache[race_name] = (
                    detector.get_session_completion_state(target_year, race_name, "R")
                    == "completed"
                )
            if not race_completion_cache[race_name]:
                continue

            if race_name not in sprint_cache:
                try:
                    sprint_cache[race_name] = bool(is_sprint_weekend(target_year, race_name))
                except Exception as exc:
                    logger.warning(
                        "Could not determine weekend type while reconciling actuals for %s %s: %s",
                        target_year,
                        race_name,
                        exc,
                    )
                    sprint_cache[race_name] = False

            if self._prediction_has_full_actuals(
                prediction,
                is_sprint=sprint_cache[race_name],
            ):
                continue

            qualifying_session, race_session = self._actual_sessions_for_checkpoint(
                session_name=session_name,
                is_sprint=sprint_cache[race_name],
            )
            target_predictions = self._prediction_targets(
                prediction,
                is_sprint=sprint_cache[race_name],
            )
            target_actual_results: dict[str, ActualResultRows | None] = {}

            for actual_session in {qualifying_session, race_session}:
                cache_key = (race_name, actual_session)
                if cache_key not in actual_cache:
                    actual_cache[cache_key] = fetch_actual_session_results(
                        target_year,
                        race_name,
                        fastf1_session_name(actual_session),
                    )

            for target_key, payload in target_predictions.items():
                target_session = str(
                    payload.get("target_session") or target_session_name(target_key)
                )
                cache_key = (race_name, target_session)
                if cache_key not in actual_cache:
                    actual_cache[cache_key] = fetch_actual_session_results(
                        target_year,
                        race_name,
                        fastf1_session_name(target_session),
                    )
                target_actual_results[target_key] = actual_cache.get(cache_key)

            qualifying_results = actual_cache.get((race_name, qualifying_session))
            race_results = actual_cache.get((race_name, race_session))
            has_target_results = any(rows is not None for rows in target_actual_results.values())
            if qualifying_results is None and race_results is None and not has_target_results:
                continue

            updated = self.update_actuals(
                year=target_year,
                race_name=race_name,
                session_name=session_name,
                qualifying_results=qualifying_results,
                race_results=race_results,
                target_actual_results=target_actual_results,
            )
            if updated:
                updated_predictions += 1

        return updated_predictions

    def has_prediction_for_session(self, year: int, race_name: str, session_name: str) -> bool:
        """Check if a prediction already exists for a race checkpoint."""
        normalized_year, normalized_race_name, normalized_session_name = self._prediction_identity(
            year,
            race_name,
            session_name,
        )
        artifact_key = self._artifact_key_for_prediction(
            normalized_year,
            normalized_race_name,
            normalized_session_name,
        )
        try:
            data = self.artifact_store.load_artifact(
                artifact_type="prediction",
                artifact_key=artifact_key,
                version="latest",
            )
            if data:
                return True
        except Exception:
            pass

        filepath = self._prediction_file_path(
            normalized_year,
            normalized_race_name,
            normalized_session_name,
        )
        return filepath.exists()

    def _load_predictions_from_files(self, year: int) -> list[dict[str, Any]]:
        """Load season predictions from local files when listing from storage fails."""
        year_dir = self.predictions_dir / str(year)
        if not year_dir.exists():
            return []

        predictions: list[dict[str, Any]] = []
        for race_dir in year_dir.iterdir():
            if not race_dir.is_dir():
                continue
            for prediction_file in race_dir.glob("*.json"):
                try:
                    with open(prediction_file) as file_handle:
                        payload = json.load(file_handle)
                except Exception as exc:
                    logger.warning("Could not read prediction file %s: %s", prediction_file, exc)
                    continue
                if self._validate_prediction_schema(payload):
                    predictions.append(payload)

        predictions.sort(key=self._prediction_sort_key)
        return predictions

    def _prediction_payload_from_listing_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Extract a validated payload from an artifact listing row."""
        payload = row.get("data")
        if not isinstance(payload, dict):
            return None
        if not self._validate_prediction_schema(payload):
            return None
        return payload

    def _normalize_target_predictions(
        self,
        target_predictions: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        """Normalize target payloads before they are persisted."""
        normalized: dict[str, dict[str, Any]] = {}
        if not isinstance(target_predictions, dict):
            return normalized

        for target_key, payload in target_predictions.items():
            if not isinstance(payload, dict):
                continue
            predicted_order = sanitize_prediction_rows(payload.get("predicted_order"))
            if not predicted_order:
                continue
            fp_blend_info = payload.get("fp_blend_info")
            normalized[target_key] = {
                "target_session": str(
                    payload.get("target_session", target_session_name(target_key))
                )
                .strip()
                .upper(),
                "predicted_order": predicted_order,
                "result_mode": str(payload.get("result_mode", "PREDICTED")).strip().upper(),
                "grid_source": str(payload.get("grid_source", "PREDICTED")).strip().upper(),
                "fp_blend_info": fp_blend_info if isinstance(fp_blend_info, dict) else {},
                "mean_confidence": payload.get("mean_confidence"),
                "eligible_at_save": bool(payload.get("eligible_at_save", True)),
            }
        return normalized

    @staticmethod
    def _normalize_target_actual_results(
        target_actual_results: dict[str, ActualResultRows | None] | None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Normalize target actual rows keyed by canonical target name."""
        normalized: dict[str, list[dict[str, Any]]] = {}
        if not isinstance(target_actual_results, dict):
            return normalized
        for target_key, rows in target_actual_results.items():
            sanitized = sanitize_actual_rows(rows)
            if sanitized:
                normalized[target_key] = sanitized
        return normalized

    @staticmethod
    def _infer_weekend_format(
        *,
        session_name: str,
        target_predictions: dict[str, dict[str, Any]],
    ) -> str:
        """Infer weekend format when the caller does not provide one."""
        if any("sprint" in target_key for target_key in target_predictions):
            return weekend_format_name(True)
        checkpoint = str(session_name).strip().upper()
        if checkpoint in {"SQ", "SPRINT"}:
            return weekend_format_name(True)
        return weekend_format_name(False)

    @staticmethod
    def _prediction_is_sprint_weekend(
        prediction: dict[str, Any],
        *,
        is_sprint: bool | None = None,
    ) -> bool:
        """Return the weekend format for a saved prediction."""
        metadata = prediction.get("metadata", {})
        if isinstance(metadata, dict):
            weekend_format = str(metadata.get("weekend_format", "")).strip().lower()
            if weekend_format in {"sprint", "normal"}:
                return weekend_format == "sprint"
        return bool(is_sprint)

    def _prediction_targets(
        self,
        prediction: dict[str, Any],
        *,
        is_sprint: bool,
    ) -> dict[str, dict[str, Any]]:
        """Return canonical target predictions for a saved payload."""
        explicit_targets = explicit_target_predictions(prediction)
        if explicit_targets:
            return explicit_targets
        return synthesize_legacy_targets(prediction, is_sprint=is_sprint)

    def _prediction_actual_targets(
        self,
        prediction: dict[str, Any],
        *,
        is_sprint: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return canonical target actuals for a saved payload."""
        explicit_targets = explicit_target_actuals(prediction)
        if explicit_targets:
            return explicit_targets
        return synthesize_legacy_actuals(prediction, is_sprint=is_sprint)

    def _derive_target_actual_results_from_legacy_inputs(
        self,
        prediction: dict[str, Any],
        *,
        qualifying_results: ActualResultRows | None,
        race_results: ActualResultRows | None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Map legacy top-level actual inputs onto canonical target keys."""
        metadata = prediction.get("metadata", {})
        checkpoint_session = str(metadata.get("session_name", "")).strip().upper()
        is_sprint = self._prediction_is_sprint_weekend(prediction)
        prediction_targets = self._prediction_targets(prediction, is_sprint=is_sprint)

        derived: dict[str, list[dict[str, Any]]] = {}
        for target_key, payload in prediction_targets.items():
            target_session = str(payload.get("target_session", target_session_name(target_key)))
            if target_session in {"Q", "SQ"} and qualifying_results is not None:
                sanitized = sanitize_actual_rows(qualifying_results)
            elif target_session in {"R", "SPRINT"} and race_results is not None:
                sanitized = sanitize_actual_rows(race_results)
            else:
                sanitized = []
            if sanitized:
                derived[target_key] = sanitized

        if derived:
            return derived

        qualifying_target, race_target = legacy_target_keys_for_prediction(
            checkpoint_session,
            is_sprint=is_sprint,
        )
        if qualifying_target is not None and qualifying_results is not None:
            sanitized = sanitize_actual_rows(qualifying_results)
            if sanitized:
                derived[qualifying_target] = sanitized
        if race_target is not None and race_results is not None:
            sanitized = sanitize_actual_rows(race_results)
            if sanitized:
                derived[race_target] = sanitized
        return derived

    @staticmethod
    def _should_update_top_level_actual(
        prediction: dict[str, Any],
        *,
        session_type: str,
    ) -> bool:
        """Return True when legacy top-level actuals should be written."""
        metadata = prediction.get("metadata", {})
        if not isinstance(metadata, dict):
            return True
        metadata_key = f"top_level_{session_type}_eligible_at_save"
        return bool(metadata.get(metadata_key, True))

    @staticmethod
    def _normalize_race_name(value: Any) -> str:
        """Normalize race names so whitespace drift cannot create a new storage key."""
        return " ".join(str(value).split()).strip()

    @staticmethod
    def _normalize_session_name(value: Any) -> str:
        """Normalize checkpoint session identifiers for storage and lookup."""
        return str(value).strip().upper()

    @classmethod
    def _prediction_identity(
        cls, year: Any, race_name: Any, session_name: Any
    ) -> tuple[int, str, str]:
        """Return the canonical storage identity for one saved prediction."""
        return (
            int(year),
            cls._normalize_race_name(race_name),
            cls._normalize_session_name(session_name),
        )

    @classmethod
    def _artifact_key_for_prediction(cls, year: Any, race_name: Any, session_name: Any) -> str:
        """Build the canonical artifact key for one saved prediction."""
        normalized_year, normalized_race_name, normalized_session_name = cls._prediction_identity(
            year,
            race_name,
            session_name,
        )
        return f"{normalized_year}::{normalized_race_name}::{normalized_session_name}"

    def _deduplicate_predictions(self, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse duplicate checkpoint payloads and keep the newest one per state."""
        deduped_by_identity: dict[tuple[int, str, str], dict[str, Any]] = {}
        duplicate_count = 0

        for prediction in predictions:
            metadata = prediction.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            try:
                identity = self._prediction_identity(
                    metadata.get("year"),
                    metadata.get("race_name"),
                    metadata.get("session_name"),
                )
            except (TypeError, ValueError):
                continue

            existing = deduped_by_identity.get(identity)
            if existing is None:
                deduped_by_identity[identity] = prediction
                continue

            duplicate_count += 1
            if self._prediction_sort_key(prediction) >= self._prediction_sort_key(existing):
                deduped_by_identity[identity] = prediction

        deduped_predictions = list(deduped_by_identity.values())
        deduped_predictions.sort(key=self._prediction_sort_key)
        if duplicate_count > 0:
            logger.warning(
                "Collapsed %s duplicate prediction artifact(s) while loading history",
                duplicate_count,
            )
        return deduped_predictions

    @staticmethod
    def _prediction_sort_key(prediction: dict[str, Any]) -> tuple[datetime, str, str]:
        """Build a stable sort key for prediction-history rendering."""
        metadata = prediction.get("metadata", {})
        predicted_at = metadata.get("predicted_at") if isinstance(metadata, dict) else None
        race_name = (
            PredictionLogger._normalize_race_name(metadata.get("race_name", ""))
            if isinstance(metadata, dict)
            else ""
        )
        session_name = (
            PredictionLogger._normalize_session_name(metadata.get("session_name", ""))
            if isinstance(metadata, dict)
            else ""
        )
        return (
            PredictionLogger._parse_prediction_timestamp(predicted_at),
            race_name,
            session_name,
        )

    @staticmethod
    def _parse_prediction_timestamp(value: Any) -> datetime:
        """Parse a saved prediction timestamp with a safe minimum fallback."""
        if not isinstance(value, str):
            return datetime.min.replace(tzinfo=UTC)
        candidate = value.strip()
        if not candidate:
            return datetime.min.replace(tzinfo=UTC)
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return datetime.min.replace(tzinfo=UTC)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _prediction_has_full_actuals(
        self,
        prediction: dict[str, Any],
        *,
        is_sprint: bool | None = None,
    ) -> bool:
        """Return True when every stored target already has actual results."""
        sprint_flag = self._prediction_is_sprint_weekend(prediction, is_sprint=is_sprint)
        target_predictions = self._prediction_targets(prediction, is_sprint=sprint_flag)
        if target_predictions:
            target_actuals = self._prediction_actual_targets(prediction, is_sprint=sprint_flag)
            return all(bool(target_actuals.get(target_key)) for target_key in target_predictions)

        actuals = prediction.get("actuals", {})
        if not isinstance(actuals, dict):
            return False
        return actuals.get("qualifying") is not None and actuals.get("race") is not None

    @staticmethod
    def _actual_sessions_for_checkpoint(
        *,
        session_name: str,
        is_sprint: bool,
    ) -> tuple[str, str]:
        """Return legacy top-level actual sessions for a checkpoint prediction."""
        checkpoint_upper = str(session_name).strip().upper()
        if is_sprint and checkpoint_upper in {"PRE", "FP1", "SQ", "SPRINT"}:
            return "SQ", "SPRINT"
        return "Q", "R"

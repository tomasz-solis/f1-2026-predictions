"""Helpers for reconstructing checkpoint predictions from stored session snapshots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from src.dashboard.live_prediction_flow import prediction_targets_for_checkpoint
from src.persistence.artifact_store import ArtifactStore
from src.predictors.baseline_2026 import Baseline2026Predictor
from src.utils import config_loader
from src.utils.accuracy_snapshots import build_accuracy_snapshot_records
from src.utils.accuracy_targets import (
    eligible_target_keys,
    explicit_target_actuals,
    legacy_target_keys_for_prediction,
    synthesize_legacy_actuals,
    target_deadline_session,
    weekend_format_name,
)
from src.utils.car_snapshot_history import (
    SNAPSHOT_ARTIFACT_TYPE,
    snapshot_artifact_key,
    snapshot_sort_timestamp,
    sort_snapshot_payloads,
)
from src.utils.prediction_logger import ActualResultRows, PredictionLogger
from src.utils.prediction_metrics import PredictionMetrics
from src.utils.race_input_confidence import cap_predicted_main_race_input_confidence
from src.utils.team_mapping import map_team_to_characteristics
from src.utils.weekend import is_sprint_weekend


def _copy_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a defensive deep copy when a payload is available."""
    if not isinstance(payload, dict):
        return None
    return deepcopy(payload)


class SnapshotOverlayArtifactStore:
    """Delegate artifact store reads while overriding season car characteristics."""

    def __init__(
        self,
        *,
        base_store: ArtifactStore,
        season_year: int,
        car_characteristics_payload: dict[str, Any],
    ) -> None:
        """Initialize the wrapper with one overlaid car-characteristics payload."""
        self.base_store = base_store
        self.season_year = int(season_year)
        self.car_characteristics_payload = deepcopy(car_characteristics_payload)
        self.data_root = base_store.data_root
        self.storage_mode = getattr(base_store, "storage_mode", "file_only")

    def load_artifact(
        self,
        artifact_type: str,
        artifact_key: str,
        version: str | int = "latest",
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return overlaid season characteristics while delegating every other read."""
        if (
            artifact_type == "car_characteristics"
            and artifact_key == f"{self.season_year}::car_characteristics"
        ):
            return deepcopy(self.car_characteristics_payload)
        return self.base_store.load_artifact(
            artifact_type,
            artifact_key,
            version=version,
            run_id=run_id,
        )

    def save_artifact(
        self,
        artifact_type: str,
        artifact_key: str,
        data: dict[str, Any],
        version: int | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Delegate writes to the wrapped artifact store."""
        return self.base_store.save_artifact(
            artifact_type,
            artifact_key,
            data,
            version=version,
            run_id=run_id,
        )

    def list_artifacts(
        self,
        artifact_type: str,
        key_prefix: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Delegate listing operations to the wrapped store."""
        return self.base_store.list_artifacts(
            artifact_type,
            key_prefix=key_prefix,
            limit=limit,
        )

    def get_latest_version(self, artifact_type: str, artifact_key: str) -> int:
        """Delegate version checks to the wrapped store."""
        return self.base_store.get_latest_version(artifact_type, artifact_key)


@dataclass
class ReconstructionSummary:
    """Summary returned after one reconstructed checkpoint is persisted."""

    year: int
    race_name: str
    checkpoint_session: str
    weather: str
    is_sprint: bool
    target_keys: list[str]
    prediction_path: Path
    snapshot_records_written: int
    actuals_source: str
    information_cutoff_at: str | None


def load_checkpoint_snapshot_payload(
    *,
    store: ArtifactStore,
    year: int,
    race_name: str,
    checkpoint_session: str,
) -> dict[str, Any]:
    """Load one stored car-characteristics snapshot for a race checkpoint."""
    checkpoint_session_upper = str(checkpoint_session).strip().upper()
    artifact_key = snapshot_artifact_key(year, race_name, checkpoint_session_upper)
    payload = store.load_artifact(SNAPSHOT_ARTIFACT_TYPE, artifact_key)
    if isinstance(payload, dict):
        return payload

    if checkpoint_session_upper == "PRE":
        fallback_payload = _load_latest_snapshot_before_pre_checkpoint(
            store=store,
            year=year,
            race_name=race_name,
            is_sprint=is_sprint_weekend(year, race_name),
        )
        if isinstance(fallback_payload, dict):
            return fallback_payload

    raise FileNotFoundError(
        f"Missing car-characteristics snapshot for {race_name} {year} {checkpoint_session_upper}"
    )


def _load_latest_snapshot_before_pre_checkpoint(
    *,
    store: ArtifactStore,
    year: int,
    race_name: str,
    is_sprint: bool,
) -> dict[str, Any] | None:
    """Return the newest stored snapshot that predates the weekend PRE deadline."""
    from src.utils.accuracy_targets import _scheduled_session_start

    deadline_starts: list[datetime] = []
    weekend_format = weekend_format_name(is_sprint)
    for target_key in eligible_target_keys("PRE", is_sprint):
        deadline_session = target_deadline_session(target_key, weekend_format, "PRE")
        if not deadline_session:
            continue
        deadline_start = _scheduled_session_start(
            year=year,
            race_name=race_name,
            session_name=deadline_session,
        )
        if deadline_start is not None:
            deadline_starts.append(deadline_start)

    if not deadline_starts:
        return None

    pre_deadline = min(deadline_starts)
    snapshot_rows = store.list_artifacts(
        SNAPSHOT_ARTIFACT_TYPE,
        key_prefix=f"{int(year)}::",
        limit=8192,
    )
    candidate_payloads: list[dict[str, Any]] = []
    for row in snapshot_rows:
        payload = row.get("data")
        if not isinstance(payload, dict):
            continue
        if snapshot_sort_timestamp(payload) >= pre_deadline:
            continue
        candidate_payloads.append(payload)

    if not candidate_payloads:
        return None

    return sort_snapshot_payloads(candidate_payloads)[-1]


def build_snapshot_overlay_car_characteristics(
    *,
    base_car_payload: dict[str, Any],
    snapshot_payload: dict[str, Any],
) -> dict[str, Any]:
    """Overlay snapshot practice profiles onto the base season characteristics payload."""
    merged_payload = deepcopy(base_car_payload)
    base_teams = merged_payload.setdefault("teams", {})
    if not isinstance(base_teams, dict):
        raise ValueError("Base car-characteristics payload is missing a valid `teams` mapping")

    known_teams = set(base_teams.keys())
    snapshot_teams = snapshot_payload.get("teams", {})
    if not isinstance(snapshot_teams, dict) or not snapshot_teams:
        raise ValueError("Snapshot payload does not contain any team profiles")

    for raw_team_name, raw_team_payload in snapshot_teams.items():
        if not isinstance(raw_team_payload, dict):
            continue
        profiles = raw_team_payload.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            continue

        mapped_name = map_team_to_characteristics(str(raw_team_name), known_teams=known_teams)
        team_name = mapped_name if mapped_name else str(raw_team_name)
        team_payload = base_teams.setdefault(team_name, {})
        if not isinstance(team_payload, dict):
            team_payload = {}
            base_teams[team_name] = team_payload

        team_payload["testing_characteristics_profiles"] = deepcopy(profiles)
        balanced_profile = profiles.get("balanced")
        if isinstance(balanced_profile, dict) and balanced_profile:
            team_payload["testing_characteristics"] = deepcopy(balanced_profile)

    merged_payload["checkpoint_snapshot"] = {
        "event_name": str(snapshot_payload.get("event_name", "")).strip(),
        "session_name": str(snapshot_payload.get("session_name", "")).strip().upper(),
        "source": str(snapshot_payload.get("source", "")).strip(),
        "captured_at": snapshot_payload.get("captured_at"),
        "session_started_at": snapshot_payload.get("session_started_at"),
    }
    return merged_payload


def _resolve_dashboard_simulation_count(kind: str) -> int:
    """Resolve default dashboard simulation counts without importing rendering code."""
    if str(kind).strip().lower() == "qualifying":
        return int(config_loader.get("prediction_precompute.qualifying_n_simulations", 100))
    return int(config_loader.get("prediction_precompute.race_n_simulations", 100))


def _derive_race_input_confidence(
    qualifying_result: dict[str, Any],
    *,
    grid_source: str,
) -> float:
    """Match the dashboard race-input confidence heuristic for reconstructed runs."""
    if str(grid_source).strip().upper() == "ACTUAL":
        return 1.0

    try:
        base_confidence = float(qualifying_result.get("data_confidence_score", 0.5))
    except (TypeError, ValueError):
        base_confidence = 0.5
    base_confidence = max(0.0, min(base_confidence, 1.0))

    data_source = str(qualifying_result.get("data_source", "")).lower()
    source_adjustment = 0.0
    if "model-only" in data_source:
        source_adjustment = -0.10
    elif "testing short-run profile blend" in data_source:
        source_adjustment = -0.05

    grid_adjustment = 0.20 if str(grid_source).strip().upper() == "ACTUAL" else 0.0
    return max(0.0, min(base_confidence + source_adjustment + grid_adjustment, 1.0))


def _mark_predicted_qualifying_section(section: dict[str, Any]) -> dict[str, Any]:
    """Return a qualifying-style section tagged as a predicted reconstruction."""
    marked = deepcopy(section)
    marked["result_mode"] = "PREDICTED"
    marked["grid_source"] = "PREDICTED"
    return marked


def _mark_predicted_race_section(
    section: dict[str, Any],
    *,
    input_confidence: float,
) -> dict[str, Any]:
    """Return a race-style section tagged as a predicted reconstruction."""
    marked = deepcopy(section)
    marked["result_mode"] = "PREDICTED"
    marked["grid_source"] = "PREDICTED"
    marked["input_confidence"] = round(float(input_confidence), 3)
    return marked


def build_reconstructed_prediction_results(
    *,
    year: int,
    race_name: str,
    weather: str,
    checkpoint_session: str,
    artifact_store: ArtifactStore,
    qualifying_n_simulations: int | None = None,
    race_n_simulations: int | None = None,
) -> tuple[dict[str, Any], bool]:
    """Build retrospective prediction results from a stored checkpoint snapshot."""
    checkpoint_session_upper = str(checkpoint_session).strip().upper()
    is_sprint = bool(is_sprint_weekend(year, race_name))

    base_car_payload = artifact_store.load_artifact(
        "car_characteristics",
        f"{int(year)}::car_characteristics",
    )
    if not isinstance(base_car_payload, dict):
        raise FileNotFoundError(f"Missing base car characteristics for season {year}")

    snapshot_payload = load_checkpoint_snapshot_payload(
        store=artifact_store,
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint_session_upper,
    )
    overlay_payload = build_snapshot_overlay_car_characteristics(
        base_car_payload=base_car_payload,
        snapshot_payload=snapshot_payload,
    )
    overlay_store = SnapshotOverlayArtifactStore(
        base_store=artifact_store,
        season_year=year,
        car_characteristics_payload=overlay_payload,
    )

    predictor = Baseline2026Predictor(
        season_year=year,
        artifact_store=cast(ArtifactStore, overlay_store),
    )
    qualifying_runs = (
        int(qualifying_n_simulations)
        if qualifying_n_simulations is not None
        else _resolve_dashboard_simulation_count("qualifying")
    )
    race_runs = (
        int(race_n_simulations)
        if race_n_simulations is not None
        else _resolve_dashboard_simulation_count("race")
    )

    if is_sprint:
        sprint_quali = _mark_predicted_qualifying_section(
            predictor.predict_qualifying(
                year=year,
                race_name=race_name,
                n_simulations=qualifying_runs,
                qualifying_stage="sprint",
                practice_signal_mode="stored_profiles",
                checkpoint_session_name=checkpoint_session_upper,
            )
        )
        sprint_input_confidence = _derive_race_input_confidence(
            sprint_quali,
            grid_source=sprint_quali["grid_source"],
        )
        sprint_race = _mark_predicted_race_section(
            predictor.predict_sprint_race(
                sprint_quali_grid=sprint_quali["grid"],
                weather=weather,
                race_name=race_name,
                n_simulations=race_runs,
                input_confidence=sprint_input_confidence,
            ),
            input_confidence=sprint_input_confidence,
        )
        main_quali = _mark_predicted_qualifying_section(
            predictor.predict_qualifying(
                year=year,
                race_name=race_name,
                n_simulations=qualifying_runs,
                qualifying_stage="main",
                practice_signal_mode="stored_profiles",
                checkpoint_session_name=checkpoint_session_upper,
            )
        )
        main_race_input_confidence = _derive_race_input_confidence(
            main_quali,
            grid_source=main_quali["grid_source"],
        )
        main_race_input_confidence = cap_predicted_main_race_input_confidence(
            main_race_input_confidence,
            qualifying_result=main_quali,
            grid_source=main_quali["grid_source"],
            is_sprint_weekend=True,
            boundary_session_name=checkpoint_session_upper,
        )
        main_race = _mark_predicted_race_section(
            predictor.predict_race(
                qualifying_grid=main_quali["grid"],
                weather=weather,
                race_name=race_name,
                n_simulations=race_runs,
                year=year,
                input_confidence=main_race_input_confidence,
            ),
            input_confidence=main_race_input_confidence,
        )
        prediction_results = {
            "sprint_quali": sprint_quali,
            "sprint_race": sprint_race,
            "main_quali": main_quali,
            "main_race": main_race,
        }
    else:
        qualifying = _mark_predicted_qualifying_section(
            predictor.predict_qualifying(
                year=year,
                race_name=race_name,
                n_simulations=qualifying_runs,
                qualifying_stage="main",
                practice_signal_mode="stored_profiles",
                checkpoint_session_name=checkpoint_session_upper,
            )
        )
        race_input_confidence = _derive_race_input_confidence(
            qualifying,
            grid_source=qualifying["grid_source"],
        )
        race = _mark_predicted_race_section(
            predictor.predict_race(
                qualifying_grid=qualifying["grid"],
                weather=weather,
                race_name=race_name,
                n_simulations=race_runs,
                year=year,
                input_confidence=race_input_confidence,
            ),
            input_confidence=race_input_confidence,
        )
        prediction_results = {"qualifying": qualifying, "race": race}

    prediction_results["_prediction_context"] = {
        "boundary_session_name": checkpoint_session_upper,
        "reconstruction_source": "car_characteristics_snapshot",
    }
    return prediction_results, is_sprint


def collect_saved_target_actuals(
    *,
    prediction_logger: PredictionLogger,
    year: int,
    race_name: str,
    is_sprint: bool,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Collect actual target rows for one race from already-saved prediction artifacts."""
    actuals_by_target: dict[str, list[dict[str, Any]]] = {}
    source_session = ""

    for prediction in prediction_logger.get_all_predictions(year):
        metadata = prediction.get("metadata", {})
        if str(metadata.get("race_name", "")).strip() != race_name:
            continue

        actuals = explicit_target_actuals(prediction)
        if not actuals:
            actuals = synthesize_legacy_actuals(prediction, is_sprint=is_sprint)
        for target_key, rows in actuals.items():
            if target_key in actuals_by_target:
                continue
            if isinstance(rows, list) and rows:
                actuals_by_target[target_key] = deepcopy(rows)
                if not source_session:
                    source_session = str(metadata.get("session_name", "")).strip().upper()

        expected_targets = {
            target_key
            for target_key in eligible_target_keys("PRE" if is_sprint else "FP1", is_sprint)
        }
        if expected_targets and expected_targets.issubset(actuals_by_target):
            break

    if not actuals_by_target:
        raise FileNotFoundError(
            f"Could not find saved actual target rows for {race_name} {year} in local predictions"
        )
    return actuals_by_target, (source_session or "saved_prediction")


def compute_information_cutoff_at(
    *,
    year: int,
    race_name: str,
    checkpoint_session: str,
    is_sprint: bool,
) -> str | None:
    """Return the latest clean timestamp for a retrospective checkpoint reconstruction."""
    from src.utils.accuracy_targets import _scheduled_session_start

    weekend_format = weekend_format_name(is_sprint)
    checkpoint = str(checkpoint_session).strip().upper()
    deadline_starts: list[datetime] = []
    for target_key in eligible_target_keys(checkpoint, is_sprint):
        deadline_session = target_deadline_session(target_key, weekend_format, checkpoint)
        if not deadline_session:
            continue
        deadline_start = _scheduled_session_start(
            year=year,
            race_name=race_name,
            session_name=deadline_session,
        )
        if deadline_start is not None:
            deadline_starts.append(deadline_start)

    if not deadline_starts:
        return None

    return (min(deadline_starts) - timedelta(seconds=1)).astimezone(UTC).isoformat()


def infer_saved_weather(
    *,
    prediction_logger: PredictionLogger,
    year: int,
    race_name: str,
) -> str:
    """Infer weather from already-saved checkpoints for the same race."""
    for prediction in prediction_logger.get_all_predictions(year):
        metadata = prediction.get("metadata", {})
        if str(metadata.get("race_name", "")).strip() != race_name:
            continue
        weather = str(metadata.get("weather", "")).strip().lower()
        if weather:
            return weather
    return "dry"


def reconstruct_checkpoint_prediction(
    *,
    year: int,
    race_name: str,
    checkpoint_session: str,
    weather: str | None = None,
    data_root: str | Path = "data",
    overwrite: bool = False,
    qualifying_n_simulations: int | None = None,
    race_n_simulations: int | None = None,
) -> ReconstructionSummary:
    """Rebuild, save, and score one retrospective checkpoint prediction."""
    checkpoint_session_upper = str(checkpoint_session).strip().upper()
    artifact_store = ArtifactStore(data_root=data_root)
    prediction_logger = PredictionLogger(
        predictions_dir=str(Path(data_root) / "predictions"),
    )
    if not overwrite and prediction_logger.has_prediction_for_session(
        year,
        race_name,
        checkpoint_session_upper,
    ):
        raise FileExistsError(
            f"Prediction already exists for {race_name} {year} {checkpoint_session_upper}"
        )

    resolved_weather = (
        str(weather).strip().lower()
        if isinstance(weather, str) and weather.strip()
        else infer_saved_weather(
            prediction_logger=prediction_logger,
            year=year,
            race_name=race_name,
        )
    )
    prediction_results, is_sprint = build_reconstructed_prediction_results(
        year=year,
        race_name=race_name,
        weather=resolved_weather,
        checkpoint_session=checkpoint_session_upper,
        artifact_store=artifact_store,
        qualifying_n_simulations=qualifying_n_simulations,
        race_n_simulations=race_n_simulations,
    )
    target_predictions = prediction_targets_for_checkpoint(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        session_name=checkpoint_session_upper,
    )
    if not target_predictions:
        raise ValueError(
            f"No eligible targets could be reconstructed for {race_name} {checkpoint_session_upper}"
        )

    qualifying_target, race_target = legacy_target_keys_for_prediction(
        checkpoint_session_upper,
        is_sprint=is_sprint,
    )
    information_cutoff_at = compute_information_cutoff_at(
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint_session_upper,
        is_sprint=is_sprint,
    )
    generated_at = datetime.now(UTC).isoformat()

    if is_sprint:
        checkpoint_payload = (
            prediction_results["sprint_quali"]
            if checkpoint_session_upper in {"PRE", "FP1", "SQ", "SPRINT"}
            else prediction_results["main_quali"]
        )
        race_payload = (
            prediction_results["sprint_race"]
            if checkpoint_session_upper in {"PRE", "FP1", "SQ", "SPRINT"}
            else prediction_results["main_race"]
        )
    else:
        checkpoint_payload = prediction_results["qualifying"]
        race_payload = prediction_results["race"]

    prediction_path = prediction_logger.save_prediction(
        year=year,
        race_name=race_name,
        session_name=checkpoint_session_upper,
        qualifying_prediction=checkpoint_payload["grid"],
        race_prediction=race_payload["finish_order"],
        weather=resolved_weather,
        fp_blend_info=checkpoint_payload.get("fp_blend_info", {}),
        target_predictions=target_predictions,
        metadata={
            "source": "checkpoint_reconstruction",
            "weekend_format": weekend_format_name(is_sprint),
            "top_level_qualifying_target": qualifying_target,
            "top_level_race_target": race_target,
            "top_level_qualifying_eligible_at_save": qualifying_target in target_predictions,
            "top_level_race_eligible_at_save": race_target in target_predictions,
            "top_level_qualifying_result_mode": checkpoint_payload.get("result_mode", "PREDICTED"),
            "top_level_race_result_mode": race_payload.get("result_mode", "PREDICTED"),
            "top_level_qualifying_grid_source": checkpoint_payload.get("grid_source", "PREDICTED"),
            "top_level_race_grid_source": race_payload.get("grid_source", "PREDICTED"),
            "reconstructed_at": generated_at,
            "reconstruction_source": "car_characteristics_snapshot",
            "information_cutoff_at": information_cutoff_at,
        },
    )

    saved_actuals, actuals_source = collect_saved_target_actuals(
        prediction_logger=prediction_logger,
        year=year,
        race_name=race_name,
        is_sprint=is_sprint,
    )
    prediction_logger.update_actuals(
        year=year,
        race_name=race_name,
        session_name=checkpoint_session_upper,
        qualifying_results=saved_actuals.get(qualifying_target) if qualifying_target else None,
        race_results=saved_actuals.get(race_target) if race_target else None,
        target_actual_results=cast(dict[str, ActualResultRows | None], saved_actuals),
    )

    saved_prediction = prediction_logger.load_prediction(year, race_name, checkpoint_session_upper)
    if saved_prediction is None:
        raise RuntimeError(
            f"Could not reload reconstructed prediction for {race_name} {checkpoint_session_upper}"
        )

    metrics_calculator = PredictionMetrics()
    snapshot_records = build_accuracy_snapshot_records(
        prediction_data=saved_prediction,
        is_sprint=is_sprint,
        metrics_calculator=metrics_calculator,
        generated_by="checkpoint_reconstruction",
    )
    for record in snapshot_records:
        artifact_store.save_artifact(
            artifact_type="accuracy_snapshot",
            artifact_key=record["artifact_key"],
            data=record["data"],
            version=1,
        )

    return ReconstructionSummary(
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint_session_upper,
        weather=resolved_weather,
        is_sprint=is_sprint,
        target_keys=sorted(target_predictions.keys()),
        prediction_path=prediction_path,
        snapshot_records_written=len(snapshot_records),
        actuals_source=actuals_source,
        information_cutoff_at=information_cutoff_at,
    )

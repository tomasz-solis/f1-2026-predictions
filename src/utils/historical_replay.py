"""Build checkpoint-by-checkpoint historical forecasts without leaking future data."""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from src.dashboard.checkpoint_predictor import build_checkpoint_overlay_predictor
from src.dashboard.precomputed_predictions import get_prediction_precompute_config
from src.dashboard.prediction_checkpointing import (
    prediction_model_diagnostics_for_sections,
    prediction_payload_for_session,
    prediction_sections_for_session,
    prediction_targets_for_checkpoint,
)
from src.dashboard.prediction_flow import (
    build_actual_qualifying_section,
    build_actual_race_section,
    build_starting_grid_note,
)
from src.dashboard.race_context import attach_starting_grid_context
from src.persistence.artifact_store import ArtifactStore
from src.predictors.baseline_2026 import Baseline2026Predictor
from src.systems.testing_updater import _season_snapshot_plan, update_from_testing_sessions
from src.systems.updater import update_from_race, update_from_sprint_race
from src.types.prediction_types import QualifyingGridEntry
from src.utils.accuracy_snapshots import build_accuracy_snapshot_records
from src.utils.accuracy_targets import (
    TARGET_SPRINT_QUALIFYING,
    explicit_target_predictions,
    fastf1_session_name,
    legacy_target_keys_for_prediction,
    target_session_name,
    weekend_format_name,
)
from src.utils.checkpoint_reconstruction import compute_information_cutoff_at
from src.utils.config_loader import Config
from src.utils.grid_validation import validate_qualifying_grid
from src.utils.lineups import get_lineups
from src.utils.prediction_context import build_historical_prediction_context
from src.utils.prediction_logger import ActualResultRows, PredictionLogger
from src.utils.prediction_metrics import PredictionMetrics
from src.utils.race_input_confidence import (
    cap_predicted_main_race_input_confidence,
    derive_race_input_confidence,
)
from src.utils.weekend import is_sprint_weekend

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDED_SCORING_TARGETS = frozenset({TARGET_SPRINT_QUALIFYING})
_NORMAL_REPLAY_CHECKPOINTS = ("PRE", "FP1", "FP2", "FP3")
_SPRINT_REPLAY_CHECKPOINTS = ("PRE", "FP1", "SQ")
# A practice session can be genuinely unusable: FastF1 publishes 2026 Barcelona FP1 with
# laps but no team names, so no team profile can be extracted from it. A weekend like that
# still happened, and the checkpoint after it is simply the state that preceded it. Only
# practice degrades this way. Testing days ("Day 1"..) and competitive sessions still fail
# closed, because a season seed or a scored result built on missing data is not a replay.
_DEGRADABLE_SESSIONS = frozenset({"FP1", "FP2", "FP3"})


@dataclass
class ReplayCheckpointRecord:
    """Summary for one saved checkpoint forecast."""

    year: int
    race_name: str
    checkpoint_session: str
    weekend_format: str
    prediction_path: str
    state_summary_path: str
    scored_targets: list[str] = field(default_factory=list)
    excluded_targets: list[str] = field(default_factory=list)
    metrics_by_target: dict[str, dict[str, Any]] = field(default_factory=dict)
    information_cutoff_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return asdict(self)


@dataclass
class HistoricalReplaySummary:
    """Summary for one historical replay run."""

    year: int
    output_root: str
    processed_data_dir: str
    excluded_scoring_targets: list[str]
    testing_sessions_replayed: list[str] = field(default_factory=list)
    weekend_sessions_replayed: list[str] = field(default_factory=list)
    skipped_sessions: list[str] = field(default_factory=list)
    race_updates: list[str] = field(default_factory=list)
    checkpoints: list[ReplayCheckpointRecord] = field(default_factory=list)
    driver_update_trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        payload = asdict(self)
        payload["checkpoints"] = [record.to_dict() for record in self.checkpoints]
        return payload


class ReplayConfigOverride:
    """Small config wrapper that enforces replay-specific safety overrides."""

    def __init__(
        self,
        *,
        base_config: Any,
        overrides: dict[str, Any],
    ) -> None:
        """Store the wrapped config object plus key-level overrides."""
        self._base_config = base_config
        self._overrides = dict(overrides)

    def get(self, key: str, default: Any = None) -> Any:
        """Return an overridden value first, then fall back to the wrapped config."""
        if key in self._overrides:
            return self._overrides[key]
        base_getter = getattr(self._base_config, "get", None)
        if callable(base_getter):
            return base_getter(key, default)
        return default

    def __getattr__(self, name: str) -> Any:
        """Delegate any non-`get` attribute access to the wrapped config object."""
        return getattr(self._base_config, name)


def checkpoint_sequence_for_weekend(is_sprint: bool) -> tuple[str, ...]:
    """Return the forecast checkpoints we want to materialize for a weekend."""
    return _SPRINT_REPLAY_CHECKPOINTS if bool(is_sprint) else _NORMAL_REPLAY_CHECKPOINTS


def session_is_available_at_checkpoint(checkpoint_session: str, target_session: str) -> bool:
    """Return True when a target session should be treated as already completed."""
    checkpoint = str(checkpoint_session or "").strip().upper()
    target = str(target_session or "").strip().upper()
    if not checkpoint or checkpoint == "PRE" or not target:
        return False

    session_order = {
        "FP1": 1,
        "FP2": 2,
        "FP3": 3,
        "SQ": 4,
        "SPRINT": 5,
        "Q": 6,
        "R": 7,
    }
    checkpoint_order = session_order.get(checkpoint, -1)
    target_order = session_order.get(target, 99)
    return checkpoint_order >= 0 and target_order <= checkpoint_order


def apply_target_scoring_policy(
    target_predictions: dict[str, dict[str, Any]],
    *,
    excluded_scoring_targets: set[str] | frozenset[str],
) -> dict[str, dict[str, Any]]:
    """Mark configured targets as saved-but-not-scored."""
    adjusted = deepcopy(target_predictions)
    for target_key in excluded_scoring_targets:
        payload = adjusted.get(target_key)
        if not isinstance(payload, dict):
            continue
        payload["eligible_at_save"] = False
    return adjusted


@contextmanager
def _force_file_only_storage() -> Any:
    """Temporarily force file-backed persistence for a sidecar replay run."""
    previous_mode = os.environ.get("USE_DB_STORAGE")
    os.environ["USE_DB_STORAGE"] = "file_only"
    try:
        yield
    finally:
        if previous_mode is None:
            os.environ.pop("USE_DB_STORAGE", None)
        else:
            os.environ["USE_DB_STORAGE"] = previous_mode


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")


def _slugify_race_name(race_name: str) -> str:
    """Return a stable filesystem-safe race label."""
    return str(race_name).strip().lower().replace(" ", "_").replace("'", "")


def _resolve_simulation_count(kind: str) -> int:
    """Resolve simulation counts from the existing dashboard config."""
    settings = get_prediction_precompute_config()
    if str(kind).strip().lower() == "qualifying":
        return int(settings.get("qualifying_n_simulations", 100))
    return int(settings.get("race_n_simulations", 100))


def _build_replay_predictor_config() -> ReplayConfigOverride:
    """Build a predictor config that blocks prediction-log feedback during replay."""
    return ReplayConfigOverride(
        base_config=Config(),
        overrides={
            "baseline_predictor.current_season_form.infer_from_saved_actuals": False,
        },
    )


def _reset_replay_artifacts(processed_dir: Path, year: int) -> None:
    """Reset season artifacts to the pre-race baseline used by the replay."""
    from scripts.rebuild_2026_race_artifacts import (
        _read_json,
        _reset_car_artifact,
        _reset_driver_artifact,
    )

    car_file = processed_dir / "car_characteristics" / f"{int(year)}_car_characteristics.json"
    driver_file = (
        processed_dir / "driver_characteristics" / f"{int(year)}_driver_characteristics.json"
    )
    # The flat driver_characteristics.json is the committed pre-season snapshot
    # (sessions_observed=0). It is what makes the replay leakage-safe: resetting
    # from the in-season season-scoped file instead would seed race-1 forecasts
    # with end-of-season ratings.
    baseline_driver_file = processed_dir / "driver_characteristics.json"
    if not baseline_driver_file.exists():
        raise FileNotFoundError(
            f"Replay pre-season driver baseline not found: {baseline_driver_file}. "
            "This flat driver_characteristics.json is a generated, git-ignored "
            "pre-season snapshot (sessions_observed=0) that the source processed dir "
            "is copied from. Regenerate it with "
            "`uv run python scripts/extract_driver_characteristics.py "
            "--output data/processed/driver_characteristics.json`, or for the exact "
            "last-committed snapshot run "
            "`git show ff9197a0^:data/processed/driver_characteristics.json "
            "> data/processed/driver_characteristics.json`, then re-run."
        )

    _reset_car_artifact(car_file, year=int(year))
    _reset_driver_artifact(
        driver_file,
        baseline_payload=_read_json(baseline_driver_file),
        year=int(year),
    )


def _prepare_output_root(
    *,
    source_processed_dir: Path,
    output_root: Path,
    year: int,
    overwrite: bool,
) -> Path:
    """Create a clean replay data root and seed it with processed inputs."""
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Replay output root already exists: {output_root}. Pass overwrite=True to rebuild."
            )
        shutil.rmtree(output_root)

    processed_dir = output_root / "processed"
    shutil.copytree(source_processed_dir, processed_dir)
    _reset_replay_artifacts(processed_dir, year=int(year))
    return processed_dir


def _fetch_actual_session_results(
    *,
    year: int,
    race_name: str,
    session_name: str,
    actual_cache: dict[tuple[int, str, str], list[QualifyingGridEntry]],
) -> list[QualifyingGridEntry]:
    """Load one actual competitive-session classification with a small in-memory cache."""
    from src.data.actual_results_fetcher import fetch_actual_session_results

    normalized_session = str(session_name).strip().upper()
    cache_key = (int(year), str(race_name), normalized_session)
    if cache_key not in actual_cache:
        loaded = fetch_actual_session_results(
            int(year),
            str(race_name),
            fastf1_session_name(normalized_session),
        )
        if not loaded:
            raise FileNotFoundError(
                f"Could not load actual {normalized_session} results for {race_name} {year}"
            )
        actual_cache[cache_key] = deepcopy(loaded)
    return deepcopy(actual_cache[cache_key])


def _fetch_actual_starting_grid(
    *,
    year: int,
    race_name: str,
    actual_cache: dict[tuple[int, str, str], list[QualifyingGridEntry]],
) -> list[QualifyingGridEntry]:
    """Load the grid a completed race actually started from, penalties included."""
    from src.data.actual_results_fetcher import fetch_actual_starting_grid

    cache_key = (int(year), str(race_name), "R_STARTING_GRID")
    if cache_key not in actual_cache:
        loaded = fetch_actual_starting_grid(int(year), str(race_name))
        if not loaded:
            raise FileNotFoundError(
                f"Could not load the actual starting grid for {race_name} {year}"
            )
        actual_cache[cache_key] = deepcopy(loaded)
    return deepcopy(actual_cache[cache_key])


def _resolve_qualifying_section_for_replay(
    *,
    predictor: Any,
    year: int,
    race_name: str,
    checkpoint_session: str,
    target_session: str,
    qualifying_stage: str,
    actual_cache: dict[tuple[int, str, str], list[QualifyingGridEntry]],
) -> tuple[dict[str, Any], list[QualifyingGridEntry], str]:
    """Return the checkpoint-appropriate qualifying payload for one target session."""
    if session_is_available_at_checkpoint(checkpoint_session, target_session):
        actual_grid = _fetch_actual_session_results(
            year=year,
            race_name=race_name,
            session_name=target_session,
            actual_cache=actual_cache,
        )
        section = build_actual_qualifying_section(actual_grid, session_name=target_session)
        return section, actual_grid, "ACTUAL"

    section = predictor.predict_qualifying(
        year=year,
        race_name=race_name,
        qualifying_stage=qualifying_stage,
        n_simulations=_resolve_simulation_count("qualifying"),
        practice_signal_mode="stored_profiles",
        checkpoint_session_name=checkpoint_session,
        prediction_context=build_historical_prediction_context(
            year=year,
            race_name=race_name,
            target_session_name=target_session,
        ),
    )
    section = deepcopy(section)
    section["grid_source"] = "PREDICTED"
    section["result_mode"] = "PREDICTED"
    predicted_grid = validate_qualifying_grid(list(section.get("grid", [])))
    return section, predicted_grid, "PREDICTED"


def _resolve_race_section_for_replay(
    *,
    predictor: Any,
    year: int,
    race_name: str,
    checkpoint_session: str,
    target_session: str,
    qualifying_grid: list[QualifyingGridEntry],
    qualifying_grid_source: str,
    grid_session_name: str,
    weather: str,
    input_confidence: float,
    actual_cache: dict[tuple[int, str, str], list[QualifyingGridEntry]],
) -> dict[str, Any]:
    """Return the checkpoint-appropriate race payload for one target session."""
    # Qualifying classification is not the starting grid. Once qualifying is inside the
    # checkpoint the penalties are known too, so the race replays from the grid the cars
    # actually lined up on. Before qualifying the grid is genuinely being predicted, and
    # substituting the post-penalty order there would leak the future into the replay.
    if str(target_session).strip().upper() == "R" and qualifying_grid_source == "ACTUAL":
        qualifying_grid = _fetch_actual_starting_grid(
            year=year,
            race_name=race_name,
            actual_cache=actual_cache,
        )

    if session_is_available_at_checkpoint(checkpoint_session, target_session):
        actual_results = _fetch_actual_session_results(
            year=year,
            race_name=race_name,
            session_name=target_session,
            actual_cache=actual_cache,
        )
        section = build_actual_race_section(actual_results, session_name=target_session)
        section["grid_source"] = qualifying_grid_source
        attach_starting_grid_context(section, qualifying_grid, grid_session_name)
        if qualifying_grid_source == "ACTUAL":
            section["starting_grid_note"] = build_starting_grid_note(grid_session_name)
        return section

    if str(target_session).strip().upper() == "SPRINT":
        section = predictor.predict_sprint_race(
            sprint_quali_grid=qualifying_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=_resolve_simulation_count("race"),
            input_confidence=input_confidence,
            prediction_context=build_historical_prediction_context(
                year=year,
                race_name=race_name,
                target_session_name=target_session,
            ),
        )
    else:
        section = predictor.predict_race(
            qualifying_grid=qualifying_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=_resolve_simulation_count("race"),
            year=year,
            input_confidence=input_confidence,
            prediction_context=build_historical_prediction_context(
                year=year,
                race_name=race_name,
                target_session_name=target_session,
            ),
        )
    section = deepcopy(section)
    section["grid_source"] = qualifying_grid_source
    section["result_mode"] = "PREDICTED"
    section["input_confidence"] = round(float(input_confidence), 3)
    attach_starting_grid_context(section, qualifying_grid, grid_session_name)
    if qualifying_grid_source == "ACTUAL":
        section["starting_grid_note"] = build_starting_grid_note(grid_session_name)
    return section


def build_checkpoint_prediction_results(
    *,
    predictor: Any,
    year: int,
    race_name: str,
    checkpoint_session: str,
    weather: str,
    is_sprint: bool,
    actual_cache: dict[tuple[int, str, str], list[QualifyingGridEntry]],
) -> dict[str, Any]:
    """Build all still-relevant prediction sections for one replay checkpoint."""
    checkpoint = str(checkpoint_session).strip().upper()

    if is_sprint:
        sprint_quali, sprint_grid, sprint_grid_source = _resolve_qualifying_section_for_replay(
            predictor=predictor,
            year=year,
            race_name=race_name,
            checkpoint_session=checkpoint,
            target_session="SQ",
            qualifying_stage="sprint",
            actual_cache=actual_cache,
        )
        sprint_input_confidence = derive_race_input_confidence(
            sprint_quali,
            grid_source=sprint_grid_source,
        )
        sprint_race = _resolve_race_section_for_replay(
            predictor=predictor,
            year=year,
            race_name=race_name,
            checkpoint_session=checkpoint,
            target_session="SPRINT",
            qualifying_grid=sprint_grid,
            qualifying_grid_source=sprint_grid_source,
            grid_session_name="SQ",
            weather=weather,
            input_confidence=sprint_input_confidence,
            actual_cache=actual_cache,
        )

        main_quali, main_grid, main_grid_source = _resolve_qualifying_section_for_replay(
            predictor=predictor,
            year=year,
            race_name=race_name,
            checkpoint_session=checkpoint,
            target_session="Q",
            qualifying_stage="main",
            actual_cache=actual_cache,
        )
        main_race_input_confidence = derive_race_input_confidence(
            main_quali,
            grid_source=main_grid_source,
        )
        main_race_input_confidence = cap_predicted_main_race_input_confidence(
            main_race_input_confidence,
            qualifying_result=main_quali,
            grid_source=main_grid_source,
            is_sprint_weekend=True,
            boundary_session_name=checkpoint,
        )
        main_race = _resolve_race_section_for_replay(
            predictor=predictor,
            year=year,
            race_name=race_name,
            checkpoint_session=checkpoint,
            target_session="R",
            qualifying_grid=main_grid,
            qualifying_grid_source=main_grid_source,
            grid_session_name="Q",
            weather=weather,
            input_confidence=main_race_input_confidence,
            actual_cache=actual_cache,
        )
        return {
            "sprint_quali": sprint_quali,
            "sprint_race": sprint_race,
            "main_quali": main_quali,
            "main_race": main_race,
            "_prediction_context": {
                "boundary_session_name": checkpoint,
                "reconstruction_source": "historical_replay",
            },
        }

    qualifying, qualifying_grid, grid_source = _resolve_qualifying_section_for_replay(
        predictor=predictor,
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint,
        target_session="Q",
        qualifying_stage="main",
        actual_cache=actual_cache,
    )
    race_input_confidence = derive_race_input_confidence(
        qualifying,
        grid_source=grid_source,
    )
    race = _resolve_race_section_for_replay(
        predictor=predictor,
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint,
        target_session="R",
        qualifying_grid=qualifying_grid,
        qualifying_grid_source=grid_source,
        grid_session_name="Q",
        weather=weather,
        input_confidence=race_input_confidence,
        actual_cache=actual_cache,
    )
    return {
        "qualifying": qualifying,
        "race": race,
        "_prediction_context": {
            "boundary_session_name": checkpoint,
            "reconstruction_source": "historical_replay",
        },
    }


def _build_target_actual_results(
    *,
    year: int,
    race_name: str,
    target_predictions: dict[str, dict[str, Any]],
    actual_cache: dict[tuple[int, str, str], list[QualifyingGridEntry]],
) -> dict[str, ActualResultRows | None]:
    """Load actual rows for every stored target."""
    target_actuals: dict[str, ActualResultRows | None] = {}
    for target_key, payload in target_predictions.items():
        target_session = str(payload.get("target_session") or target_session_name(target_key))
        target_actuals[target_key] = _fetch_actual_session_results(
            year=year,
            race_name=race_name,
            session_name=target_session,
            actual_cache=actual_cache,
        )
    return target_actuals


def _target_eligibility_map(saved_prediction: dict[str, Any]) -> dict[str, bool]:
    """Return per-target save eligibility from the stored payload."""
    eligibility: dict[str, bool] = {}
    for target_key, payload in explicit_target_predictions(saved_prediction).items():
        eligibility[target_key] = bool(payload.get("eligible_at_save", True))
    return eligibility


def _build_state_summary_payload(
    *,
    predictor: Any,
    year: int,
    race_name: str,
    checkpoint_session: str,
    is_sprint: bool,
    scored_targets: list[str],
    excluded_targets: list[str],
    metrics_by_target: dict[str, dict[str, Any]],
    prediction_path: Path,
) -> dict[str, Any]:
    """Build one checkpoint state summary for later inspection."""
    try:
        lineups = get_lineups(year, race_name)
    except Exception:
        lineups = {}

    driver_team_map = {
        str(driver_code): str(team_name)
        for team_name, drivers in lineups.items()
        for driver_code in drivers
    }

    team_state: dict[str, Any] = {}
    for team_name, team_data in sorted(getattr(predictor, "teams", {}).items()):
        if not isinstance(team_data, dict):
            continue
        team_state[str(team_name)] = {
            "overall_performance": team_data.get("overall_performance"),
            "uncertainty": team_data.get("uncertainty"),
            "races_completed": team_data.get("races_completed"),
            "current_season_performance": deepcopy(team_data.get("current_season_performance", [])),
            "directionality": deepcopy(team_data.get("directionality", {})),
            "testing_characteristics": deepcopy(team_data.get("testing_characteristics", {})),
            "testing_characteristics_profiles": deepcopy(
                team_data.get("testing_characteristics_profiles", {})
            ),
            "checkpoint_driver_deltas_seconds": deepcopy(
                team_data.get("checkpoint_driver_deltas_seconds", {})
            ),
        }

    driver_state: dict[str, Any] = {}
    for driver_code, driver_data in sorted(getattr(predictor, "drivers", {}).items()):
        if not isinstance(driver_data, dict):
            continue
        team_name = driver_team_map.get(str(driver_code), "")
        checkpoint_delta = None
        checkpoint_delta_getter = getattr(predictor, "_get_checkpoint_driver_delta_seconds", None)
        if team_name and callable(checkpoint_delta_getter):
            checkpoint_delta = checkpoint_delta_getter(
                team_name,
                str(driver_code),
            )
        driver_state[str(driver_code)] = {
            "team": team_name,
            "pace": deepcopy(driver_data.get("pace", {})),
            "racecraft": deepcopy(driver_data.get("racecraft", {})),
            "experience": deepcopy(driver_data.get("experience", {})),
            "bayesian": deepcopy(driver_data.get("bayesian", {})),
            "checkpoint_driver_delta_seconds": checkpoint_delta,
        }

    return {
        "year": int(year),
        "race_name": str(race_name),
        "checkpoint_session": str(checkpoint_session).strip().upper(),
        "weekend_format": weekend_format_name(is_sprint),
        "prediction_path": str(prediction_path),
        "car_characteristics_snapshot": deepcopy(
            getattr(predictor, "car_characteristics_snapshot", {})
        ),
        "scored_targets": list(scored_targets),
        "excluded_targets": list(excluded_targets),
        "metrics_by_target": deepcopy(metrics_by_target),
        "teams": team_state,
        "drivers": driver_state,
    }


def _write_state_summary(
    *,
    output_root: Path,
    state_payload: dict[str, Any],
    race_name: str,
    checkpoint_session: str,
) -> Path:
    """Persist one human-inspectable checkpoint state summary."""
    path = (
        output_root
        / "reports"
        / "checkpoints"
        / str(state_payload["year"])
        / _slugify_race_name(race_name)
        / f"{str(checkpoint_session).strip().lower()}.json"
    )
    _write_json(path, state_payload)
    return path


def _write_summary_markdown(summary: HistoricalReplaySummary, output_root: Path) -> None:
    """Write a short markdown index for the generated replay artifacts."""
    lines = [
        f"# Historical Replay {summary.year}",
        "",
        f"- Output root: `{summary.output_root}`",
        f"- Processed data dir: `{summary.processed_data_dir}`",
        f"- Excluded scoring targets: {', '.join(summary.excluded_scoring_targets) or 'none'}",
        "",
        "## Checkpoints",
    ]

    for record in summary.checkpoints:
        scored_targets = ", ".join(record.scored_targets) or "none"
        excluded_targets = ", ".join(record.excluded_targets) or "none"
        lines.extend(
            [
                f"- {record.race_name} {record.checkpoint_session}:",
                f"  prediction `{record.prediction_path}`",
                f"  state `{record.state_summary_path}`",
                f"  scored `{scored_targets}`",
                f"  excluded `{excluded_targets}`",
            ]
        )

    markdown_path = output_root / "reports" / "summary.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n")


def _build_race_checkpoint_record(
    *,
    year: int,
    race_name: str,
    checkpoint_session: str,
    weather: str,
    processed_dir: Path,
    output_root: Path,
    excluded_scoring_targets: set[str] | frozenset[str],
    actual_cache: dict[tuple[int, str, str], list[QualifyingGridEntry]],
    predictor_config: ReplayConfigOverride,
) -> ReplayCheckpointRecord:
    """Generate, save, score, and summarize one replay checkpoint."""
    is_sprint = bool(is_sprint_weekend(year, race_name))
    artifact_store = ArtifactStore(data_root=output_root)
    base_predictor = Baseline2026Predictor(
        data_dir=str(processed_dir),
        season_year=year,
        artifact_store=artifact_store,
        config=cast(Config, predictor_config),
    )
    predictor = build_checkpoint_overlay_predictor(
        base_predictor=base_predictor,
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint_session,
        is_sprint=is_sprint,
    )

    prediction_results = build_checkpoint_prediction_results(
        predictor=predictor,
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint_session,
        weather=weather,
        is_sprint=is_sprint,
        actual_cache=actual_cache,
    )
    target_predictions = prediction_targets_for_checkpoint(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        session_name=checkpoint_session,
    )
    target_predictions = apply_target_scoring_policy(
        target_predictions,
        excluded_scoring_targets=excluded_scoring_targets,
    )

    qualifying_rows, race_rows, fp_blend_info = prediction_payload_for_session(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        session_name=checkpoint_session,
    )
    qualifying_section, race_section = prediction_sections_for_session(
        prediction_results=prediction_results,
        is_sprint=is_sprint,
        session_name=checkpoint_session,
    )
    qualifying_target, race_target = legacy_target_keys_for_prediction(
        checkpoint_session,
        is_sprint=is_sprint,
    )
    information_cutoff_at = compute_information_cutoff_at(
        year=year,
        race_name=race_name,
        checkpoint_session=checkpoint_session,
        is_sprint=is_sprint,
    )

    logger_instance = PredictionLogger(predictions_dir=str(output_root / "predictions"))
    target_actual_results = _build_target_actual_results(
        year=year,
        race_name=race_name,
        target_predictions=target_predictions,
        actual_cache=actual_cache,
    )
    qualifying_actual = (
        target_actual_results.get(qualifying_target) if qualifying_target is not None else None
    )
    race_actual = target_actual_results.get(race_target) if race_target is not None else None

    top_level_qualifying_eligible = bool(
        target_predictions.get(str(qualifying_target), {}).get("eligible_at_save", False)
    )
    top_level_race_eligible = bool(
        target_predictions.get(str(race_target), {}).get("eligible_at_save", False)
    )

    prediction_path = logger_instance.save_prediction(
        year=year,
        race_name=race_name,
        session_name=checkpoint_session,
        qualifying_prediction=qualifying_rows,
        race_prediction=race_rows,
        weather=weather,
        fp_blend_info=fp_blend_info,
        target_predictions=target_predictions,
        metadata={
            "source": "historical_replay",
            "weekend_format": weekend_format_name(is_sprint),
            "top_level_qualifying_target": qualifying_target,
            "top_level_race_target": race_target,
            "top_level_qualifying_eligible_at_save": top_level_qualifying_eligible,
            "top_level_race_eligible_at_save": top_level_race_eligible,
            "top_level_qualifying_result_mode": str(
                qualifying_section.get("result_mode", "PREDICTED")
            )
            .strip()
            .upper(),
            "top_level_race_result_mode": str(race_section.get("result_mode", "PREDICTED"))
            .strip()
            .upper(),
            "top_level_qualifying_grid_source": str(
                qualifying_section.get("grid_source", "PREDICTED")
            )
            .strip()
            .upper(),
            "top_level_race_grid_source": str(race_section.get("grid_source", "PREDICTED"))
            .strip()
            .upper(),
            "information_cutoff_at": information_cutoff_at,
            "excluded_scoring_targets": sorted(excluded_scoring_targets),
            **prediction_model_diagnostics_for_sections(
                qualifying_section=qualifying_section,
                race_section=race_section,
            ),
        },
    )
    logger_instance.update_actuals(
        year=year,
        race_name=race_name,
        session_name=checkpoint_session,
        qualifying_results=qualifying_actual,
        race_results=race_actual,
        target_actual_results=target_actual_results,
    )

    saved_prediction = logger_instance.load_prediction(year, race_name, checkpoint_session)
    if saved_prediction is None:
        raise RuntimeError(
            f"Could not reload saved replay prediction for {race_name} {checkpoint_session}"
        )

    metrics_calculator = PredictionMetrics()
    metrics_by_target = metrics_calculator.calculate_prediction_target_metrics(
        saved_prediction,
        is_sprint=is_sprint,
    )
    target_eligibility = _target_eligibility_map(saved_prediction)
    filtered_metrics = {
        target_key: metrics
        for target_key, metrics in metrics_by_target.items()
        if target_eligibility.get(target_key, True)
    }

    snapshot_records = build_accuracy_snapshot_records(
        prediction_data=saved_prediction,
        is_sprint=is_sprint,
        metrics_calculator=metrics_calculator,
        generated_by="historical_replay",
    )
    for record in snapshot_records:
        artifact_store.save_artifact(
            artifact_type="accuracy_snapshot",
            artifact_key=record["artifact_key"],
            data=record["data"],
            version=1,
        )

    state_summary_path = _write_state_summary(
        output_root=output_root,
        state_payload=_build_state_summary_payload(
            predictor=predictor,
            year=year,
            race_name=race_name,
            checkpoint_session=checkpoint_session,
            is_sprint=is_sprint,
            scored_targets=sorted(filtered_metrics),
            excluded_targets=sorted(
                set(target_eligibility)
                - {key for key, eligible in target_eligibility.items() if eligible}
            ),
            metrics_by_target=filtered_metrics,
            prediction_path=prediction_path,
        ),
        race_name=race_name,
        checkpoint_session=checkpoint_session,
    )

    return ReplayCheckpointRecord(
        year=year,
        race_name=race_name,
        checkpoint_session=str(checkpoint_session).strip().upper(),
        weekend_format=weekend_format_name(is_sprint),
        prediction_path=str(prediction_path),
        state_summary_path=str(state_summary_path),
        scored_targets=sorted(filtered_metrics),
        excluded_targets=sorted(
            set(target_eligibility)
            - {key for key, eligible in target_eligibility.items() if eligible}
        ),
        metrics_by_target=filtered_metrics,
        information_cutoff_at=information_cutoff_at,
    )


def _apply_session_update(
    *,
    year: int,
    event_name: str,
    session_name: str,
    cache_dirs: list[str],
    processed_dir: Path,
) -> bool:
    """Replay one cached session into the sidecar season state.

    Returns whether the session was applied. An unusable practice session returns
    ``False`` instead of raising; every other session still fails closed.
    """
    errors: list[str] = []
    for cache_dir in cache_dirs:
        try:
            update_from_testing_sessions(
                year=year,
                events=[event_name],
                data_dir=str(processed_dir),
                sessions=[session_name],
                cache_dir=cache_dir,
            )
            return True
        except ValueError as exc:
            errors.append(f"{cache_dir}: {exc}")

    joined_errors = "; ".join(errors) if errors else "no cache directories were available"
    if str(session_name).strip().upper() in _DEGRADABLE_SESSIONS:
        logger.warning(
            "Skipping unusable practice session %s %s: %s",
            event_name,
            session_name,
            joined_errors,
        )
        return False
    raise ValueError(f"Could not replay {event_name} {session_name}: {joined_errors}")


def run_historical_checkpoint_replay(
    *,
    year: int = 2026,
    source_processed_dir: str | Path = "data/processed",
    output_root: str | Path = "data/historical_replay",
    weather: str = "dry",
    overwrite: bool = False,
    excluded_scoring_targets: set[str] | frozenset[str] | None = None,
    stop_after_race: str | None = None,
) -> HistoricalReplaySummary:
    """Replay testing and race weekends into sidecar checkpoint forecast files."""
    processed_source = Path(source_processed_dir)
    replay_output_root = Path(output_root)
    scoring_exclusions = (
        DEFAULT_EXCLUDED_SCORING_TARGETS
        if excluded_scoring_targets is None
        else frozenset(excluded_scoring_targets)
    )

    if not processed_source.exists():
        raise FileNotFoundError(f"Processed source directory does not exist: {processed_source}")

    with _force_file_only_storage():
        processed_dir = _prepare_output_root(
            source_processed_dir=processed_source,
            output_root=replay_output_root,
            year=year,
            overwrite=overwrite,
        )
        actual_cache: dict[tuple[int, str, str], list[QualifyingGridEntry]] = {}
        driver_update_traces: list[dict[str, Any]] = []
        summary = HistoricalReplaySummary(
            year=year,
            output_root=str(replay_output_root),
            processed_data_dir=str(processed_dir),
            excluded_scoring_targets=sorted(scoring_exclusions),
        )
        predictor_config = _build_replay_predictor_config()

        replay_plan = _season_snapshot_plan(year)
        testing_entries = [
            entry for entry in replay_plan if "testing" in str(entry.get("event_name", "")).lower()
        ]
        race_entries = [
            entry
            for entry in replay_plan
            if "testing" not in str(entry.get("event_name", "")).lower()
        ]

        for plan_entry in testing_entries:
            event_name = str(plan_entry["event_name"])
            cache_dirs = [str(path) for path in plan_entry.get("cache_dirs", [])]
            for session_name in plan_entry.get("sessions", []):
                _apply_session_update(
                    year=year,
                    event_name=event_name,
                    session_name=str(session_name),
                    cache_dirs=cache_dirs,
                    processed_dir=processed_dir,
                )
                summary.testing_sessions_replayed.append(f"{event_name}::{session_name}")

        stop_after_label = str(stop_after_race).strip() if stop_after_race else None
        for plan_entry in race_entries:
            race_name = str(plan_entry["event_name"])
            cache_dirs = [str(path) for path in plan_entry.get("cache_dirs", [])]
            is_sprint = bool(is_sprint_weekend(year, race_name))
            replay_checkpoints = set(checkpoint_sequence_for_weekend(is_sprint))

            summary.checkpoints.append(
                _build_race_checkpoint_record(
                    year=year,
                    race_name=race_name,
                    checkpoint_session="PRE",
                    weather=weather,
                    processed_dir=processed_dir,
                    output_root=replay_output_root,
                    excluded_scoring_targets=scoring_exclusions,
                    actual_cache=actual_cache,
                    predictor_config=predictor_config,
                )
            )

            for session_name in plan_entry.get("sessions", []):
                normalized_session = str(session_name).strip().upper()
                applied = _apply_session_update(
                    year=year,
                    event_name=race_name,
                    session_name=normalized_session,
                    cache_dirs=cache_dirs,
                    processed_dir=processed_dir,
                )
                if applied:
                    summary.weekend_sessions_replayed.append(f"{race_name}::{normalized_session}")
                else:
                    summary.skipped_sessions.append(f"{race_name}::{normalized_session}")

                if normalized_session in replay_checkpoints and normalized_session != "PRE":
                    summary.checkpoints.append(
                        _build_race_checkpoint_record(
                            year=year,
                            race_name=race_name,
                            checkpoint_session=normalized_session,
                            weather=weather,
                            processed_dir=processed_dir,
                            output_root=replay_output_root,
                            excluded_scoring_targets=scoring_exclusions,
                            actual_cache=actual_cache,
                            predictor_config=predictor_config,
                        )
                    )

            if is_sprint:
                update_from_sprint_race(
                    year,
                    race_name,
                    str(processed_dir.parent),
                    trace_rows=driver_update_traces,
                )
            update_from_race(
                year,
                race_name,
                str(processed_dir),
                trace_rows=driver_update_traces,
            )
            summary.race_updates.append(race_name)
            if stop_after_label and race_name == stop_after_label:
                break

        reports_dir = replay_output_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        trace_path = reports_dir / "driver_update_trace.json"
        _write_json(
            trace_path,
            {
                "year": int(year),
                "rows": driver_update_traces,
            },
        )
        summary.driver_update_trace_path = str(trace_path)
        _write_json(reports_dir / "summary.json", summary.to_dict())
        _write_summary_markdown(summary, replay_output_root)
        return summary

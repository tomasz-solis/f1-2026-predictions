"""Dashboard caching and predictor bootstrap."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from src.persistence.config import should_read_db_first
from src.utils.data_paths import resolve_repo_data_path

logger = logging.getLogger(__name__)
_FASTF1_CACHE_DIR = resolve_repo_data_path("data/raw/.fastf1_cache")
_FASTF1_CACHE_ENABLED_FOR: str | None = None
_DEFAULT_SEASON = 2026
_PREDICTION_CODE_FINGERPRINT_FILES = [
    "src/dashboard/checkpoint_predictor.py",
    "src/dashboard/prediction_flow.py",
    "src/dashboard/warmup_prediction_builders.py",
    "src/data/data_generator.py",
    "src/models/bayesian.py",
    "src/models/priors_factory.py",
    "src/models/regulations.py",
    "src/predictors/baseline/data_mixin.py",
    "src/predictors/baseline/data_support.py",
    "src/predictors/baseline/qualifying_mixin.py",
    "src/predictors/baseline/qualifying_preparation.py",
    "src/predictors/baseline/qualifying_simulation.py",
    "src/predictors/baseline/race/grid_uncertainty.py",
    "src/predictors/baseline/race/race_simulation.py",
    "src/predictors/baseline/race/preparation_flow.py",
    "src/predictors/baseline/race/result_processing.py",
    "src/predictors/baseline/team_strength.py",
    "src/predictors/baseline_2026.py",
    "src/systems/testing_updater.py",
    "src/systems/testing_updater_flow.py",
    "src/systems/testing_updater_metrics.py",
    "src/systems/updater.py",
    "src/systems/updater_flow.py",
    "src/utils/checkpoint_reconstruction.py",
    "src/utils/driver_fp_adjustment.py",
    "src/utils/fp_blending.py",
    "src/utils/grid_validation.py",
    "src/utils/race_input_confidence.py",
]
_RUNTIME_PREDICTION_INPUT_FILES = [
    "data/processed/car_characteristics/{year}_car_characteristics.json",
    "data/processed/driver_characteristics/{year}_driver_characteristics.json",
    "data/processed/driver_characteristics.json",
    "data/processed/track_characteristics/{year}_track_characteristics.json",
    "data/systems/practice_characteristics_state.json",
]
_FILE_FINGERPRINT_CACHE: dict[str, tuple[int, int, tuple[int, str]]] = {}


def _fastf1_module() -> Any:
    """Import FastF1 only when a caller needs it."""
    import fastf1 as fastf1_module

    return fastf1_module


def __getattr__(name: str) -> Any:
    """Backwards-compatible lazy access to optional heavy modules."""
    if name == "fastf1":
        module = _fastf1_module()
        globals()[name] = module
        return module
    if name == "ArtifactStore":
        from src.persistence.artifact_store import ArtifactStore as artifact_store_class

        return artifact_store_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _artifact_store_class() -> Any:
    patched = globals().get("ArtifactStore")
    if patched is not None:
        return patched
    from src.persistence.artifact_store import ArtifactStore as artifact_store_class

    return artifact_store_class


def enable_fastf1_cache() -> None:
    """Enable FastF1 project-local cache."""
    global _FASTF1_CACHE_ENABLED_FOR

    _FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir_str = str(_FASTF1_CACHE_DIR)
    if _FASTF1_CACHE_ENABLED_FOR == cache_dir_str:
        return

    try:
        _fastf1_module().Cache.enable_cache(cache_dir_str)
        _FASTF1_CACHE_ENABLED_FOR = cache_dir_str
    except Exception as exc:
        logger.warning("Could not enable FastF1 cache at %s: %s", _FASTF1_CACHE_DIR, exc)


def get_artifact_versions(year: int = _DEFAULT_SEASON) -> dict[str, tuple[int, str]]:
    """Get version and deterministic fingerprint for artifacts."""
    store = _artifact_store_class()(data_root="data")
    versions = {}
    season_year = int(year)

    artifacts_to_track = [
        ("car_characteristics", f"{season_year}::car_characteristics"),
        ("driver_characteristics", f"{season_year}::driver_characteristics"),
        ("track_characteristics", f"{season_year}::track_characteristics"),
    ]

    for artifact_type, artifact_key in artifacts_to_track:
        try:
            data = store.load_artifact(artifact_type, artifact_key)
            if data:
                version = data.get("version", 1)
                updated_at = data.get(
                    "last_updated",
                    data.get("updated_at", data.get("directionality_last_updated", "")),
                )
                versions[f"{artifact_type}::{artifact_key}"] = (version, updated_at)
            else:
                versions[f"{artifact_type}::{artifact_key}"] = (0, "")
        except Exception as e:
            logger.warning("Failed to load version for %s::%s: %s", artifact_type, artifact_key, e)
            versions[f"{artifact_type}::{artifact_key}"] = (0, "")

    # Fold in the most-recent checkpoint snapshot so the prediction cache key changes when
    # a snapshot is (re)written. Checkpoint reconstructions read snapshots at predict time,
    # but the cache key derives from this function on both the warmup-write and dashboard-read
    # sides; without this, a snapshot correction (with no season-artifact change) would keep
    # serving a stale precomputed prediction. Snapshots auto-increment, so the newest row's
    # version + created_at advances on every (re)write. Defensive: never break serving.
    snapshot_fingerprint_key = f"car_characteristics_snapshot::{season_year}"
    try:
        recent_snapshots = store.list_artifacts(
            "car_characteristics_snapshot",
            key_prefix=f"{season_year}::",
            limit=1,
        )
        if recent_snapshots:
            newest = recent_snapshots[0]
            snapshot_version = int(newest.get("version", 0) or 0)
            snapshot_marker = f"{newest.get('artifact_key', '')}|{newest.get('created_at', '')}"
            versions[snapshot_fingerprint_key] = (snapshot_version, snapshot_marker)
        else:
            versions[snapshot_fingerprint_key] = (0, "")
    except Exception as exc:  # noqa: BLE001 - cache fingerprint must never break serving
        logger.warning("Failed to fingerprint car_characteristics_snapshot: %s", exc)
        versions[snapshot_fingerprint_key] = (0, "")

    # In DB-backed modes, ignore mutable local runtime files so hashes remain
    # consistent across web/worker instances (for example Render web + cron).
    file_fingerprints = _get_file_timestamps(
        year=season_year,
        include_runtime_files=not should_read_db_first(),
    )
    versions.update(file_fingerprints)

    return versions


def _get_file_timestamps(
    year: int = _DEFAULT_SEASON,
    *,
    include_runtime_files: bool = True,
) -> dict[str, tuple[int, str]]:
    """Get deterministic file fingerprints for cache-relevant local artifacts."""
    season_year = int(year)
    previous_year = max(season_year - 1, 0)
    static_files = [
        f"data/{previous_year}_pirelli_info.json",
        f"data/{season_year}_pirelli_info.json",
        "config/default.yaml",
        *_PREDICTION_CODE_FINGERPRINT_FILES,
    ]
    runtime_files = [
        file_template.format(year=season_year) for file_template in _RUNTIME_PREDICTION_INPUT_FILES
    ]
    files = static_files + (runtime_files if include_runtime_files else [])

    timestamps: dict[str, tuple[int, str]] = {}
    for file in files:
        path = resolve_repo_data_path(file)
        if path.exists():
            timestamps[file] = _fingerprint_file(path)
        else:
            timestamps[file] = (0, "")

    return timestamps


def _fingerprint_file(path: Path) -> tuple[int, str]:
    """Return a stable content fingerprint, reusing hashes when file metadata is unchanged."""
    try:
        stat_result = path.stat()
    except (AttributeError, OSError):
        stat_result = None

    cache_key = str(path)
    if stat_result is not None:
        cached = _FILE_FINGERPRINT_CACHE.get(cache_key)
        if (
            cached is not None
            and cached[0] == stat_result.st_mtime_ns
            and cached[1] == stat_result.st_size
        ):
            return cached[2]

    try:
        raw = path.read_bytes()
    except OSError:
        return (0, "")

    fingerprint = (len(raw), hashlib.sha1(raw).hexdigest())
    if stat_result is not None:
        _FILE_FINGERPRINT_CACHE[cache_key] = (
            stat_result.st_mtime_ns,
            stat_result.st_size,
            fingerprint,
        )
    return fingerprint


@st.cache_resource(show_spinner=False)
def get_predictor(_artifact_versions: dict[str, tuple[int, str]], year: int = _DEFAULT_SEASON):
    """Load and cache predictor (invalidates when artifacts change)."""
    from src.predictors.baseline_2026 import Baseline2026Predictor
    from src.utils.config_loader import Config

    canonical_logger = logging.getLogger("src.data.data_generator")
    original_canonical_level = canonical_logger.level
    canonical_logger.setLevel(logging.WARNING)

    # Refresh singleton config so cache invalidation on config/default.yaml
    # actually propagates into newly created predictors.
    try:
        Config().reload()
    except Exception as exc:
        logger.warning("Failed to reload config before predictor bootstrap: %s", exc)

    predictor = Baseline2026Predictor(season_year=year)

    canonical_logger.setLevel(original_canonical_level)

    return predictor

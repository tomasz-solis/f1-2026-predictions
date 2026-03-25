"""Dashboard caching and predictor bootstrap."""

import hashlib
import logging
from pathlib import Path

import fastf1
import streamlit as st

from src.persistence.artifact_store import ArtifactStore
from src.persistence.config import should_read_db_first

logger = logging.getLogger(__name__)
_FASTF1_CACHE_DIR = Path("data/raw/.fastf1_cache")
_DEFAULT_SEASON = 2026


def enable_fastf1_cache() -> None:
    """Enable FastF1 project-local cache."""
    _FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(str(_FASTF1_CACHE_DIR))
    except Exception as exc:
        logger.warning(f"Could not enable FastF1 cache at {_FASTF1_CACHE_DIR}: {exc}")


def get_artifact_versions(year: int = _DEFAULT_SEASON) -> dict[str, tuple[int, str]]:
    """Get version and deterministic fingerprint for artifacts."""
    store = ArtifactStore(data_root="data")
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
            logger.warning(f"Failed to load version for {artifact_type}::{artifact_key}: {e}")
            versions[f"{artifact_type}::{artifact_key}"] = (0, "")

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
    """Get deterministic file fingerprints for non-DB artifacts."""
    season_year = int(year)
    previous_year = max(season_year - 1, 0)
    static_files = [
        f"data/{previous_year}_pirelli_info.json",
        f"data/{season_year}_pirelli_info.json",
        "config/default.yaml",
        "src/predictors/baseline_2026.py",
        "src/predictors/baseline/qualifying_mixin.py",
        "src/predictors/baseline/qualifying_preparation.py",
        "src/predictors/baseline/qualifying_simulation.py",
        "src/predictors/baseline/race/preparation_flow.py",
        "src/predictors/baseline/race/prediction_flow.py",
        "src/utils/driver_fp_adjustment.py",
        "src/utils/fp_blending.py",
    ]
    runtime_files = [
        f"data/processed/car_characteristics/{season_year}_car_characteristics.json",
        f"data/processed/driver_characteristics/{season_year}_driver_characteristics.json",
        "data/processed/driver_characteristics.json",
        f"data/processed/track_characteristics/{season_year}_track_characteristics.json",
        "data/systems/practice_characteristics_state.json",
    ]
    files = static_files + (runtime_files if include_runtime_files else [])

    timestamps: dict[str, tuple[int, str]] = {}
    for file in files:
        path = Path(file)
        if path.exists():
            try:
                raw = path.read_bytes()
            except OSError:
                timestamps[file] = (0, "")
                continue
            digest = hashlib.sha1(raw).hexdigest()
            timestamps[file] = (len(raw), digest)
        else:
            timestamps[file] = (0, "")

    return timestamps


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
        logger.warning(f"Failed to reload config before predictor bootstrap: {exc}")

    predictor = Baseline2026Predictor(season_year=year)

    canonical_logger.setLevel(original_canonical_level)

    return predictor

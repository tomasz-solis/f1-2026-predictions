"""Dashboard caching and predictor bootstrap."""

import logging
import os
from pathlib import Path

import fastf1
import streamlit as st

from src.persistence.artifact_store import ArtifactStore

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
    """Get version and timestamp for artifacts (DB-backed and file-based)."""
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

    file_timestamps = _get_file_timestamps(year=season_year)
    versions.update(file_timestamps)

    return versions


def _get_file_timestamps(year: int = _DEFAULT_SEASON) -> dict[str, tuple[int, str]]:
    """Get timestamps for non-DB artifacts (config, code, Pirelli info)."""
    season_year = int(year)
    previous_year = max(season_year - 1, 0)
    files = [
        f"data/processed/car_characteristics/{season_year}_car_characteristics.json",
        f"data/processed/driver_characteristics/{season_year}_driver_characteristics.json",
        "data/processed/driver_characteristics.json",
        f"data/processed/track_characteristics/{season_year}_track_characteristics.json",
        "data/systems/practice_characteristics_state.json",
        f"data/{previous_year}_pirelli_info.json",
        f"data/{season_year}_pirelli_info.json",
        "config/default.yaml",
        "src/predictors/baseline_2026.py",
        "src/predictors/baseline/qualifying_mixin.py",
    ]

    timestamps = {}
    for file in files:
        path = Path(file)
        if path.exists():
            mtime = os.path.getmtime(path)
            timestamps[file] = (int(mtime), str(mtime))
        else:
            timestamps[file] = (0, "")

    return timestamps


@st.cache_resource(show_spinner=False)
def get_predictor(_artifact_versions: dict[str, tuple[int, str]], year: int = _DEFAULT_SEASON):
    """Load and cache predictor (invalidates when artifacts change)."""
    from src.predictors.baseline_2026 import Baseline2026Predictor
    from src.utils.config_loader import Config

    original_level = logging.getLogger("src.utils.data_generator").level
    logging.getLogger("src.utils.data_generator").setLevel(logging.WARNING)

    # Refresh singleton config so cache invalidation on config/default.yaml
    # actually propagates into newly created predictors.
    try:
        Config().reload()
    except Exception as exc:
        logger.warning(f"Failed to reload config before predictor bootstrap: {exc}")

    predictor = Baseline2026Predictor(season_year=year)

    logging.getLogger("src.utils.data_generator").setLevel(original_level)

    return predictor

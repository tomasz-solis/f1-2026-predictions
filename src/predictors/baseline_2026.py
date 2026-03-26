"""Entry point for the 2026 baseline predictor."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.data.data_generator import create_baseline_if_missing
from src.persistence.artifact_store import ArtifactStore
from src.predictors.baseline import (
    BaselineDataMixin,
    BaselineQualifyingMixin,
    BaselineRaceMixin,
)
from src.systems.systematic_learning import SystematicLearningSystem
from src.utils.config_loader import Config

logger = logging.getLogger(__name__)


class Baseline2026Predictor(
    BaselineDataMixin,
    BaselineQualifyingMixin,
    BaselineRaceMixin,
):
    """Baseline predictor used by the dashboard and command-line tools."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        seed: int = 42,
        season_year: int = 2026,
        config: Config | None = None,
        artifact_store: ArtifactStore | None = None,
    ):
        """Initialize predictor state and load the baseline data files."""
        BaselineDataMixin.__init__(self)

        self.seed = seed
        self.season_year = int(season_year)
        self.year = self.season_year

        data_dir_path = Path(data_dir)
        if not data_dir_path.is_absolute():
            env_data_dir = os.getenv("F1_DATA_DIR")
            if env_data_dir:
                self.data_dir = (
                    Path(env_data_dir) / data_dir
                    if data_dir != "data/processed"
                    else Path(env_data_dir)
                )
            else:
                self.data_dir = Path.cwd() / data_dir
        else:
            self.data_dir = data_dir_path

        logger.info("Ensuring baseline data is ready...")
        create_baseline_if_missing(self.data_dir)

        self.artifact_store = artifact_store or ArtifactStore(
            data_root=self.data_dir.parent if self.data_dir.name == "processed" else self.data_dir
        )
        self.calibration_system = SystematicLearningSystem(
            state_file=self.artifact_store.data_root / "learning_state.json"
        )
        self.config = config or Config()

        self.load_data()

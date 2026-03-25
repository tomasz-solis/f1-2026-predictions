"""Entry point for the 2026 baseline predictor."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.data.data_generator import create_baseline_if_missing
from src.persistence.artifact_store import ArtifactStore
from src.predictors.baseline import (
    BaselineDataMixin,
    BaselineQualifyingMixin,
    BaselineRaceMixin,
)
from src.predictors.baseline.components import (
    BaselineDataLoader,
    BaselineQualifyingEngine,
    BaselineRaceEngine,
    BaselineStrengthCalculator,
)
from src.systems.systematic_learning import SystematicLearningSystem
from src.types.prediction_types import QualifyingGridEntry
from src.utils.config_loader import Config

logger = logging.getLogger(__name__)


class Baseline2026Predictor(
    BaselineDataMixin,
    BaselineQualifyingMixin,
    BaselineRaceMixin,
):
    """Wire together the baseline predictor components."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        seed: int = 42,
        season_year: int = 2026,
        config: Config | None = None,
        artifact_store: ArtifactStore | None = None,
    ):
        """Initialize predictor state and supporting components."""
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

        self.data_loader = BaselineDataLoader(self)
        self.strength_calculator = BaselineStrengthCalculator(self)
        self.qualifying_engine = BaselineQualifyingEngine(self)
        self.race_engine = BaselineRaceEngine(self)

        self.load_data()

    def load_data(self) -> None:
        """Load team, driver, and track data."""
        self.data_loader.load_data()

    def calculate_track_suitability(self, team: str, race_name: str) -> float:
        """Return track suitability for a team at a given race."""
        return self.strength_calculator.calculate_track_suitability(team=team, race_name=race_name)

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        """Return the blended team strength for a given race."""
        return self.strength_calculator.get_blended_team_strength(team=team, race_name=race_name)

    def _select_race_compound(self, race_name: str) -> str:
        """Return the likely primary race compound."""
        return self.strength_calculator.select_race_compound(race_name=race_name)

    def get_compound_adjusted_team_strength(
        self,
        team: str,
        race_name: str,
        compound: str = "MEDIUM",
    ) -> float:
        """Return team strength adjusted for the selected compound."""
        return self.strength_calculator.get_compound_adjusted_team_strength(
            team=team,
            race_name=race_name,
            compound=compound,
        )

    def predict_qualifying(
        self,
        year: int,
        race_name: str,
        n_simulations: int = 50,
        qualifying_stage: str = "auto",
        practice_signal_mode: str = "auto",
        checkpoint_session_name: str | None = None,
    ) -> dict[str, Any]:
        """Predict the qualifying order for a race weekend."""
        return self.qualifying_engine.predict(
            year=year,
            race_name=race_name,
            n_simulations=n_simulations,
            qualifying_stage=qualifying_stage,
            practice_signal_mode=practice_signal_mode,
            checkpoint_session_name=checkpoint_session_name,
        )

    def predict_sprint_race(
        self,
        sprint_quali_grid: list[dict],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 50,
        input_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Delegate sprint race prediction to the qualifying engine."""
        return self.qualifying_engine.predict_sprint_race(
            sprint_quali_grid=sprint_quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=n_simulations,
            input_confidence=input_confidence,
        )

    def predict_race(
        self,
        qualifying_grid: list[QualifyingGridEntry],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 50,
        is_sprint: bool = False,
        race_compound: str = "MEDIUM",
        year: int | None = None,
        input_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Delegate race prediction to the race engine."""
        return self.race_engine.predict(
            qualifying_grid=qualifying_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=n_simulations,
            is_sprint=is_sprint,
            race_compound=race_compound,
            year=year,
            input_confidence=input_confidence,
        )

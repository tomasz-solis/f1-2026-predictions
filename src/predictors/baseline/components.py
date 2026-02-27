"""Component wrappers used by Baseline2026Predictor facade."""

from __future__ import annotations

from typing import Any

from src.predictors.baseline.data_mixin import BaselineDataMixin
from src.predictors.baseline.qualifying_mixin import BaselineQualifyingMixin
from src.predictors.baseline.race_mixin import BaselineRaceMixin
from src.types.prediction_types import QualifyingGridEntry


class BaselineDataLoader:
    """Loads baseline artifacts into predictor state."""

    def __init__(self, predictor: Any):
        self._predictor = predictor

    def load_data(self) -> None:
        """Load team, driver, and track characteristics."""
        BaselineDataMixin.load_data(self._predictor)


class BaselineStrengthCalculator:
    """Computes team-strength signals and compound-adjusted pace."""

    def __init__(self, predictor: Any):
        self._predictor = predictor

    def calculate_track_suitability(self, team: str, race_name: str) -> float:
        """Return track suitability modifier for a team at a circuit."""
        return BaselineDataMixin.calculate_track_suitability(self._predictor, team, race_name)

    def get_blended_team_strength(self, team: str, race_name: str) -> float:
        """Return blended team strength for the current race context."""
        return BaselineDataMixin.get_blended_team_strength(self._predictor, team, race_name)

    def select_race_compound(self, race_name: str) -> str:
        """Return default race compound selection for a circuit."""
        return BaselineDataMixin._select_race_compound(self._predictor, race_name)

    def get_compound_adjusted_team_strength(
        self,
        team: str,
        race_name: str,
        compound: str = "MEDIUM",
    ) -> float:
        """Return compound-adjusted team strength for race simulation."""
        return BaselineDataMixin.get_compound_adjusted_team_strength(
            self._predictor,
            team,
            race_name,
            compound,
        )


class BaselineQualifyingEngine:
    """Runs qualifying-stage simulations."""

    def __init__(self, predictor: Any):
        self._predictor = predictor

    def predict(
        self,
        year: int,
        race_name: str,
        n_simulations: int = 50,
        qualifying_stage: str = "auto",
    ) -> dict[str, Any]:
        """Predict qualifying grid and confidence ranges."""
        return BaselineQualifyingMixin.predict_qualifying(
            self._predictor,
            year=year,
            race_name=race_name,
            n_simulations=n_simulations,
            qualifying_stage=qualifying_stage,
        )

    def predict_sprint_race(
        self,
        sprint_quali_grid: list[dict],
        weather: str = "dry",
        race_name: str | None = None,
        n_simulations: int = 50,
    ) -> dict[str, Any]:
        """Predict sprint race results from sprint qualifying grid."""
        return BaselineQualifyingMixin.predict_sprint_race(
            self._predictor,
            sprint_quali_grid=sprint_quali_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=n_simulations,
        )


class BaselineRaceEngine:
    """Runs race-stage simulations."""

    def __init__(self, predictor: Any):
        self._predictor = predictor

    def predict(
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
        """Predict race finishing order and uncertainty metrics."""
        return BaselineRaceMixin.predict_race(
            self._predictor,
            qualifying_grid=qualifying_grid,
            weather=weather,
            race_name=race_name,
            n_simulations=n_simulations,
            is_sprint=is_sprint,
            race_compound=race_compound,
            year=year,
            input_confidence=input_confidence,
        )

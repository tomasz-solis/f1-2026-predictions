"""Regression tests for season-state accumulation across the two writer paths.

Both cover the same production failure: practice capture read car characteristics
from the local file while the race-learning path read and wrote the artifact store,
so every Friday practice run overwrote the accumulated season history with the
snapshot baked into the deployment image. `races_completed` sat at 6 from June to
late July while all 11 rounds were being learned.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from src.systems.testing_updater_flow import load_characteristics_payload
from src.systems.updater_flow import (
    _apply_team_performance_updates,
    _recency_weighted_mean,
)


def _write_car_characteristics(data_root, *, year: int, entries: list[float]) -> None:
    """Write a minimal on-disk car-characteristics payload for one team."""
    target = data_root / "processed" / "car_characteristics"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "year": year,
        "races_completed": len(entries),
        "teams": {
            "Aston Martin": {
                "uncertainty": 0.2,
                "overall_performance": 0.14,
                "current_season_performance": list(entries),
            }
        },
    }
    (target / f"{year}_car_characteristics.json").write_text(json.dumps(payload))


class TestLoadCharacteristicsPayloadPrefersStore:
    """`load_characteristics_payload` must not bypass the artifact store."""

    def test_prefers_store_payload_over_stale_file(self, tmp_path, monkeypatch):
        """A store payload ahead of the file wins, so history is not rolled back."""
        _write_car_characteristics(tmp_path, year=2026, entries=[0.1] * 5)
        fresh = {
            "year": 2026,
            "races_completed": 11,
            "teams": {"Aston Martin": {"current_season_performance": [0.1] * 11}},
        }
        monkeypatch.setattr(
            "src.systems.testing_updater_flow.ArtifactStore.load_artifact",
            lambda self, **kwargs: fresh,
        )

        path, payload = load_characteristics_payload(
            str(tmp_path / "processed"),
            2026,
        )

        assert path.name == "2026_car_characteristics.json"
        assert payload["races_completed"] == 11
        assert len(payload["teams"]["Aston Martin"]["current_season_performance"]) == 11

    def test_falls_back_to_file_when_store_empty(self, tmp_path, monkeypatch):
        """File-only deployments keep working when the store has nothing."""
        _write_car_characteristics(tmp_path, year=2026, entries=[0.1, 0.2])
        monkeypatch.setattr(
            "src.systems.testing_updater_flow.ArtifactStore.load_artifact",
            lambda self, **kwargs: None,
        )

        with pytest.raises(FileNotFoundError):
            load_characteristics_payload(str(tmp_path / "processed"), 2026)

    def test_reads_real_file_through_store(self, tmp_path):
        """End to end with no monkeypatching: the store resolves the same file."""
        _write_car_characteristics(tmp_path, year=2026, entries=[0.3, 0.4])

        _, payload = load_characteristics_payload(str(tmp_path / "processed"), 2026)

        assert payload["teams"]["Aston Martin"]["current_season_performance"] == [0.3, 0.4]

    def test_rejects_payload_without_teams(self, tmp_path, monkeypatch):
        """A malformed payload still raises rather than silently updating nothing."""
        _write_car_characteristics(tmp_path, year=2026, entries=[0.1])
        monkeypatch.setattr(
            "src.systems.testing_updater_flow.ArtifactStore.load_artifact",
            lambda self, **kwargs: {"year": 2026},
        )

        with pytest.raises(ValueError, match="missing 'teams'"):
            load_characteristics_payload(str(tmp_path / "processed"), 2026)


class TestRecencyWeightedMean:
    """The season mean must let a recent weekend outweigh stale rounds."""

    def test_weights_later_races_more(self):
        """Aston's real array: a strong last round must beat the flat mean of 0.1."""
        observations = [0.1, 0.1, 0.0, 0.1, 0.0, 0.3]

        weighted = _recency_weighted_mean(observations, recency_exponent=1.8)

        assert weighted > sum(observations) / len(observations)

    def test_flat_mean_when_exponent_zero(self):
        """Exponent 0 preserves the old unweighted behaviour exactly."""
        observations = [0.1, 0.9]

        assert _recency_weighted_mean(observations, recency_exponent=0.0) == pytest.approx(0.5)

    def test_single_observation_returns_itself(self):
        """One race cannot be recency-weighted against anything."""
        assert _recency_weighted_mean([0.42], recency_exponent=1.8) == pytest.approx(0.42)

    def test_empty_returns_zero(self):
        """No observations must not raise or produce nan."""
        assert _recency_weighted_mean([], recency_exponent=1.8) == 0.0


class TestApplyTeamPerformanceUpdates:
    """The baseline must move toward a strong recent round, not away from it."""

    @staticmethod
    def _apply(*, recency_exponent: float) -> dict:
        """Apply Aston's real Hungary round (0.3) on their real May history."""
        char_data = {
            "teams": {
                "Aston Martin": {
                    "uncertainty": 0.2,
                    "overall_performance": 0.14,
                    "current_season_performance": [0.1, 0.1, 0.0, 0.1, 0.0],
                }
            }
        }
        config = {
            "baseline_predictor.baseline_learning_rate": 0.3,
            "baseline_predictor.current_season_form.recency_exponent": recency_exponent,
        }

        _apply_team_performance_updates(
            char_data=char_data,
            race_pace={"Aston Martin": 0.3},
            config_get_fn=lambda key, default=None: config.get(key, default),
            logger=logging.getLogger(__name__),
            now_iso="2026-07-26T16:01:55",
        )
        return char_data["teams"]["Aston Martin"]

    def test_recent_round_counts_more_than_under_flat_mean(self):
        """A best-of-season round must land the baseline above the flat-mean result.

        Exponent 0 reproduces the old behaviour, so this pins the delta the fix buys
        rather than asserting one good weekend flips a genuinely slow car.
        """
        weighted = self._apply(recency_exponent=1.8)["overall_performance"]
        flat = self._apply(recency_exponent=0.0)["overall_performance"]

        assert weighted > flat

    def test_appends_round_and_preserves_preseason_baseline(self):
        """Bookkeeping the reset bug destroyed: one entry per race, preseason kept."""
        team = self._apply(recency_exponent=1.8)

        assert team["races_completed"] == 6
        assert team["current_season_performance"][-1] == 0.3
        assert team["preseason_overall_performance"] == 0.14


class TestTeammateRelativeSkipsUnpairedDrivers:
    """A driver whose teammate is absent has no within-team residual to observe.

    Before the fix the survivor fell through to the raw absolute 1..grid_size rating,
    mixing that scale into a model centred on the field mean. Over the 2026 replay set
    that produced 32 contaminated observations, including ANT at the maximum 22.00 from
    the one race RUS retired from.
    """

    @staticmethod
    def _model():
        """A full 11x2 grid, all seeded at the neutral field rating.

        The field must be full: `field_mean` is the mean over drivers actually in the
        session, so a toy 4-driver field shifts the whole centred scale and makes any
        absolute-range assertion meaningless.
        """
        from src.models.bayesian import BayesianDriverRanking, DriverPrior

        lineups = {"Mercedes": ["RUS", "ANT"], "Ferrari": ["LEC", "HAM"]}
        lineups.update({f"Team{i}": [f"D{i}A", f"D{i}B"] for i in range(9)})
        priors = {
            driver: DriverPrior(
                driver_number=driver,
                driver_code=driver,
                team=team,
                team_tier="top",
                mu=11.5,
                sigma=3.0,
            )
            for team, drivers in lineups.items()
            for driver in drivers
        }
        return BayesianDriverRanking(priors=priors, grid_size=22), lineups

    @staticmethod
    def _full_field(lineups, *, leading: list[str]) -> dict[str, int]:
        """Grid positions for every driver, with `leading` taking the front rows."""
        rest = [d for ds in lineups.values() for d in ds if d not in leading]
        return {driver: pos for pos, driver in enumerate([*leading, *rest], start=1)}

    def test_unpaired_driver_is_not_updated(self):
        """RUS retired, so ANT's P1 carries no teammate-relative information."""
        model, lineups = self._model()
        positions = self._full_field(lineups, leading=["ANT", "LEC", "HAM"])
        del positions["RUS"]
        before_ant = model.ratings["ANT"]
        before_lec = model.ratings["LEC"]

        model.update_teammate_relative(
            observations=positions,
            session_name="R_Canadian Grand Prix",
            lineups=lineups,
            confidence=0.35,
        )

        assert model.ratings["ANT"] == before_ant
        assert model.ratings["LEC"] != before_lec

    def test_paired_drivers_still_update(self):
        """The normal path is untouched: both teams complete, both teams update."""
        model, lineups = self._model()

        model.update_teammate_relative(
            observations=self._full_field(lineups, leading=["RUS", "ANT", "LEC", "HAM"]),
            session_name="Q_Canadian Grand Prix",
            lineups=lineups,
            confidence=1.0,
        )

        assert model.ratings["RUS"][0] > model.ratings["ANT"][0]
        # Nobody may be pinned near the absolute ends of the 1..22 scale.
        for driver in ("RUS", "ANT", "LEC", "HAM"):
            assert 5.0 < model.ratings[driver][0] < 18.0

    def test_no_pairs_at_all_falls_back_to_absolute(self):
        """The degenerate whole-field case keeps its documented fallback."""
        model, lineups = self._model()

        model.update_teammate_relative(
            observations={"ANT": 1, "LEC": 2},
            session_name="R_Sole Survivors",
            lineups=lineups,
            confidence=1.0,
        )

        assert model.ratings["ANT"][0] > 11.5


class TestPositionFallbackKeepsMargin:
    """The fallback must score the size of the gap, not just the finishing order.

    This is the path that actually runs: every stored 2026 value is an exact multiple
    of 0.1 (`1 - rank/10` on an 11-team grid), so the telemetry path has never
    contributed a race.
    """

    @staticmethod
    def _results(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
        """Build a race-results frame from (driver, team, position, status) rows."""
        return pd.DataFrame(
            [{"Abbreviation": d, "TeamName": t, "Position": p, "Status": s} for d, t, p, s in rows]
        )

    @staticmethod
    def _pace(results) -> dict[str, float]:
        """Score a race with identity team mapping."""
        from src.systems.updater_flow import _build_position_fallback_race_pace

        teams = sorted({str(t) for t in results["TeamName"]})
        return _build_position_fallback_race_pace(
            race_results=results,
            team_names=teams,
            map_team_to_characteristics_fn=lambda raw, known: (
                str(raw) if str(raw) in known else None
            ),
            logger=logging.getLogger(__name__),
        )

    def test_closing_the_gap_raises_score_without_changing_rank(self):
        """The whole point of fix 3: still last, but closer, must score higher."""
        far = self._pace(
            self._results(
                [
                    ("A1", "Fast", 1, "Finished"),
                    ("A2", "Fast", 2, "Finished"),
                    ("B1", "Mid", 3, "Finished"),
                    ("B2", "Mid", 4, "Finished"),
                    ("C1", "Slow", 9, "Finished"),
                    ("C2", "Slow", 10, "Finished"),
                ]
            )
        )
        close = self._pace(
            self._results(
                [
                    ("A1", "Fast", 1, "Finished"),
                    ("A2", "Fast", 2, "Finished"),
                    ("B1", "Mid", 3, "Finished"),
                    ("B2", "Mid", 4, "Finished"),
                    ("C1", "Slow", 5, "Finished"),
                    ("C2", "Slow", 6, "Finished"),
                ]
            )
        )

        assert far["Slow"] < close["Slow"]
        # Rank is unchanged in both races, which is exactly what rank scoring missed.
        assert far["Fast"] > far["Mid"] > far["Slow"]
        assert close["Fast"] > close["Mid"] > close["Slow"]

    def test_retirement_does_not_sink_a_front_team(self):
        """A DNF is a reliability event, not race pace; margin scoring is sensitive."""
        pace = self._pace(
            self._results(
                [
                    ("A1", "Fast", 1, "Finished"),
                    ("A2", "Fast", 10, "Accident"),
                    ("B1", "Mid", 2, "Finished"),
                    ("B2", "Mid", 3, "Finished"),
                    ("C1", "Slow", 4, "Finished"),
                    ("C2", "Slow", 5, "Finished"),
                ]
            )
        )

        assert pace["Fast"] > pace["Mid"] > pace["Slow"]

    def test_scores_stay_in_unit_range(self):
        """Scores feed a 0..1 team-strength baseline and must never leave it."""
        pace = self._pace(
            self._results(
                [
                    ("A1", "Fast", 1, "Finished"),
                    ("A2", "Fast", 2, "Finished"),
                    ("C1", "Slow", 21, "Finished"),
                    ("C2", "Slow", 22, "Finished"),
                ]
            )
        )

        assert all(0.0 <= v <= 1.0 for v in pace.values())

    def test_empty_results_return_no_scores(self):
        """No classified finishers must not raise or invent a score."""
        assert (
            self._pace(
                pd.DataFrame({"Abbreviation": [], "TeamName": [], "Position": [], "Status": []})
            )
            == {}
        )

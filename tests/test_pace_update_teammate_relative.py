"""Tests for teammate-relative quali_pace and race_pace EMA updates.

The core invariant: pace fields should reflect driver ability, not car
performance. A backmarker who consistently beats their teammate should not
accumulate a depressed pace just because their car finishes P18 every Sunday.
"""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd

from src.systems.updater import update_bayesian_driver_ratings

if TYPE_CHECKING:
    from src.models.bayesian import BayesianDriverRanking

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_race_results(
    positions: dict[str, int],
    statuses: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a minimal race results DataFrame from driver_code -> finishing position."""
    rows = []
    for code, pos in positions.items():
        row: dict = {"Abbreviation": code, "Position": pos, "race_name": "Test GP"}
        if statuses:
            row["Status"] = statuses.get(code, "Finished")
        rows.append(row)
    return pd.DataFrame(rows)


def _make_quali_results(positions: dict[str, int]) -> pd.DataFrame:
    """Build a minimal qualifying results DataFrame from driver_code -> grid position."""
    return pd.DataFrame(
        [{"Abbreviation": code, "Position": pos} for code, pos in positions.items()]
    )


def _make_driver_entry(quali_pace: float = 0.50, race_pace: float = 0.50) -> dict:
    """Build a driver characteristics entry with neutral pace values."""
    return {
        "pace": {"quali_pace": quali_pace, "race_pace": race_pace},
        "racecraft": {"skill_score": 0.50},
        "bayesian": {"rating_mu": 11.0, "rating_sigma": 2.5},
    }


def _run_update(
    driver_entries: dict[str, dict],
    race_positions: dict[str, int],
    quali_positions: dict[str, int] | None = None,
    lineups: dict[str, list[str]] | None = None,
    statuses: dict[str, str] | None = None,
    weather: str = "dry",
    qualifying_weather: str | None = None,
    trace_rows: list[dict] | None = None,
) -> dict[str, dict]:
    """Run one update cycle and return the mutated drivers dict.

    Patches all I/O and the Bayesian model so the test stays unit-level.
    The Bayesian mock returns the same mu/sigma from the payload, so skill
    blend doesn't interfere with what we're measuring (pace fields).
    """
    driver_payload: dict = {"drivers": driver_entries}

    mock_bayesian = MagicMock()
    mock_bayesian.ratings = {
        code: (entry["bayesian"]["rating_mu"], entry["bayesian"]["rating_sigma"])
        for code, entry in driver_entries.items()
    }

    race_results = _make_race_results(race_positions, statuses=statuses)
    quali_results = _make_quali_results(quali_positions) if quali_positions else None

    with (
        patch("src.models.priors_factory.PriorsFactory.create_priors", return_value={}),
        patch("src.systems.updater.BayesianDriverRanking", return_value=mock_bayesian),
        patch("src.utils.lineups.load_current_lineups", return_value=lineups),
        patch(
            "src.systems.updater._load_driver_characteristics_payload",
            return_value=driver_payload,
        ),
        patch("src.systems.updater._persist_driver_characteristics_payload"),
    ):
        update_bayesian_driver_ratings(
            race_results,
            qualifying_results=quali_results,
            weather=weather,
            qualifying_weather=qualifying_weather,
            trace_rows=trace_rows,
        )

    return driver_payload["drivers"]


# ---------------------------------------------------------------------------
# FIX 1: teammate-relative race_pace
# ---------------------------------------------------------------------------


class TestRacePaceTeammateRelative:
    """race_pace EMA updates should reflect driver ability, not car performance."""

    def test_backmarker_beating_teammate_not_penalized(self):
        """Finishing P18 while beating a P19 teammate should not depress race_pace below 0.5.

        With absolute updates, P18 maps to a raw pace near 0.10 on a 20-driver
        grid, which would drag any prior downward. With teammate-relative updates,
        the P18 driver is centered near the field mean because they outperformed
        their teammate.
        """
        lineups = {"RedBull": ["VER", "NOR"], "Williams": ["PIA", "COL"]}
        drivers = {
            "VER": _make_driver_entry(race_pace=0.50),
            "NOR": _make_driver_entry(race_pace=0.50),
            "PIA": _make_driver_entry(race_pace=0.50),
            "COL": _make_driver_entry(race_pace=0.50),
        }
        # Top team 1-3, backmarker team 18-19
        result = _run_update(
            drivers,
            race_positions={"VER": 1, "NOR": 3, "PIA": 18, "COL": 19},
            lineups=lineups,
        )

        pia_pace = result["PIA"]["pace"]["race_pace"]
        col_pace = result["COL"]["pace"]["race_pace"]

        # PIA beat their teammate so they should have a higher pace rating
        assert pia_pace > col_pace, (
            f"PIA beat COL but got lower race_pace: {pia_pace:.3f} vs {col_pace:.3f}"
        )

        # PIA's pace should not be substantially below 0.5 despite finishing P18 absolute
        assert pia_pace >= 0.46, (
            f"PIA's race_pace ({pia_pace:.3f}) too low - absolute position leaking in "
            "despite teammate-relative normalization"
        )

    def test_top_car_driver_beaten_by_teammate_rated_lower(self):
        """Finishing P3 while your teammate wins should produce a lower pace than them."""
        lineups = {"RedBull": ["VER", "NOR"], "Williams": ["PIA", "COL"]}
        drivers = {
            "VER": _make_driver_entry(race_pace=0.50),
            "NOR": _make_driver_entry(race_pace=0.50),
            "PIA": _make_driver_entry(race_pace=0.50),
            "COL": _make_driver_entry(race_pace=0.50),
        }
        result = _run_update(
            drivers,
            race_positions={"VER": 1, "NOR": 3, "PIA": 18, "COL": 19},
            lineups=lineups,
        )

        ver_pace = result["VER"]["pace"]["race_pace"]
        nor_pace = result["NOR"]["pace"]["race_pace"]

        assert ver_pace > nor_pace, (
            f"VER won but NOR has equal/higher race_pace: {ver_pace:.3f} vs {nor_pace:.3f}"
        )

    def test_no_lineups_falls_back_to_absolute(self):
        """Without lineup data, the update should still run (absolute mode)."""
        drivers = {"VER": _make_driver_entry(race_pace=0.50)}
        result = _run_update(
            drivers,
            race_positions={"VER": 1},
            lineups=None,
        )
        # P1 in absolute mode on a 1-driver grid maps to 1.0; blended from 0.5
        assert result["VER"]["pace"]["race_pace"] > 0.50

    def test_solo_driver_no_teammate_falls_back_gracefully(self):
        """A driver with no teammate in lineups gets raw absolute pace (no crash)."""
        # VER has a teammate in lineups, but SAI has no team entry
        lineups = {"RedBull": ["VER", "NOR"]}
        drivers = {
            "VER": _make_driver_entry(race_pace=0.50),
            "NOR": _make_driver_entry(race_pace=0.50),
            "SAI": _make_driver_entry(race_pace=0.50),
        }
        result = _run_update(
            drivers,
            race_positions={"VER": 1, "NOR": 3, "SAI": 5},
            lineups=lineups,
        )
        # SAI should still get updated - just without teammate normalization
        assert "race_pace" in result["SAI"]["pace"]
        assert 0.05 <= result["SAI"]["pace"]["race_pace"] <= 0.99

    def test_fully_wet_race_does_not_update_dry_race_pace(self):
        """Fully wet race results should not move dry race pace state."""
        lineups = {"Ferrari": ["LEC", "HAM"]}
        drivers = {
            "LEC": _make_driver_entry(race_pace=0.61),
            "HAM": _make_driver_entry(race_pace=0.47),
        }

        result = _run_update(
            drivers,
            race_positions={"LEC": 1, "HAM": 20},
            lineups=lineups,
            weather="rain",
        )

        assert result["LEC"]["pace"]["race_pace"] == 0.61
        assert result["HAM"]["pace"]["race_pace"] == 0.47
        assert result["LEC"]["bayesian"] == {"rating_mu": 11.0, "rating_sigma": 2.5}
        assert result["HAM"]["bayesian"] == {"rating_mu": 11.0, "rating_sigma": 2.5}

    def test_fully_wet_race_trace_reports_zero_dry_rating_delta(self):
        """The updater trace should expose wet routing evidence per driver."""
        trace_rows: list[dict] = []
        _run_update(
            {
                "LEC": {**_make_driver_entry(), "wet_skill": 0.70},
                "HAM": {**_make_driver_entry(), "wet_skill": 0.70},
            },
            race_positions={"LEC": 1, "HAM": 20},
            lineups={"Ferrari": ["LEC", "HAM"]},
            weather="rain",
            trace_rows=trace_rows,
        )

        race_rows = [row for row in trace_rows if row["session_kind"] == "race"]
        assert {row["driver_code"] for row in race_rows} == {"LEC", "HAM"}
        assert all(row["weather_route"] == "rain" for row in race_rows)
        assert all(row["dry_race_update_applied"] is False for row in race_rows)
        assert all(row["legacy_rating_mu_delta"] == 0.0 for row in race_rows)


# ---------------------------------------------------------------------------
# FIX 1: teammate-relative quali_pace
# ---------------------------------------------------------------------------


class TestQualiPaceTeammateRelative:
    """quali_pace EMA updates should reflect driver ability, not car performance."""

    def test_backmarker_qualifying_ahead_of_teammate_not_penalized(self):
        """A Q17 who beats their Q20 teammate should not get a depressed quali_pace."""
        lineups = {"RedBull": ["VER", "NOR"], "Williams": ["PIA", "COL"]}
        drivers = {
            "VER": _make_driver_entry(quali_pace=0.50),
            "NOR": _make_driver_entry(quali_pace=0.50),
            "PIA": _make_driver_entry(quali_pace=0.50),
            "COL": _make_driver_entry(quali_pace=0.50),
        }
        result = _run_update(
            drivers,
            race_positions={"VER": 1, "NOR": 3, "PIA": 18, "COL": 19},
            quali_positions={"VER": 1, "NOR": 2, "PIA": 17, "COL": 20},
            lineups=lineups,
        )

        pia_pace = result["PIA"]["pace"]["quali_pace"]
        col_pace = result["COL"]["pace"]["quali_pace"]

        assert pia_pace > col_pace, (
            f"PIA outqualified COL but got lower quali_pace: {pia_pace:.3f} vs {col_pace:.3f}"
        )
        assert pia_pace >= 0.46, (
            f"PIA's quali_pace ({pia_pace:.3f}) too low despite beating teammate"
        )

    def test_fully_wet_qualifying_does_not_update_dry_quali_pace(self):
        """Fully wet qualifying results should not move dry qualifying pace state."""
        lineups = {"Ferrari": ["LEC", "HAM"]}
        drivers = {
            "LEC": _make_driver_entry(quali_pace=0.63),
            "HAM": _make_driver_entry(quali_pace=0.45),
        }

        result = _run_update(
            drivers,
            race_positions={"LEC": 2, "HAM": 5},
            quali_positions={"LEC": 1, "HAM": 20},
            lineups=lineups,
            weather="dry",
            qualifying_weather="rain",
        )

        assert result["LEC"]["pace"]["quali_pace"] == 0.63
        assert result["HAM"]["pace"]["quali_pace"] == 0.45


# ---------------------------------------------------------------------------
# FIX 3: qualifying Bayesian update interaction
# ---------------------------------------------------------------------------


class TestQualifyingBayesianInteraction:
    """Two Bayesian updates per weekend should not amplify beyond their individual effects.

    Race and qualifying are sequentially correlated (grid advantage carries into
    finishing position). We verify the combined shift is bounded relative to what
    each update would produce in isolation.
    """

    def _run_with_real_bayesian(
        self,
        race_positions: dict[str, int],
        quali_positions: dict[str, int] | None,
        lineups: dict[str, list[str]] | None = None,
    ) -> dict[str, float]:
        """Run the full update (real Bayesian model) and return mu per driver."""
        from src.models.bayesian import DriverPrior

        priors = {
            code: DriverPrior(
                driver_number=str(i),
                driver_code=code,
                team="RedBull" if code in ("VER", "NOR") else "Williams",
                team_tier="top" if code in ("VER", "NOR") else "backmarker",
                mu=11.0,
                sigma=3.0,
            )
            for i, code in enumerate(race_positions)
        }

        driver_entries = {code: _make_driver_entry() for code in race_positions}
        driver_payload: dict = {"drivers": driver_entries}

        race_results = _make_race_results(race_positions)
        quali_results = _make_quali_results(quali_positions) if quali_positions else None

        with (
            patch("src.models.priors_factory.PriorsFactory.create_priors", return_value=priors),
            patch("src.utils.lineups.load_current_lineups", return_value=lineups),
            patch(
                "src.systems.updater._load_driver_characteristics_payload",
                return_value=driver_payload,
            ),
            patch("src.systems.updater._persist_driver_characteristics_payload"),
        ):
            update_bayesian_driver_ratings(race_results, qualifying_results=quali_results)

        return {
            code: driver_payload["drivers"][code]["bayesian"]["rating_mu"]
            for code in race_positions
        }

    def test_qualifying_update_shifts_rating(self):
        """A strong qualifying result should nudge the rating upward vs race-only."""
        race_only = self._run_with_real_bayesian(
            race_positions={"VER": 1, "NOR": 3},
            quali_positions=None,
        )
        race_and_quali = self._run_with_real_bayesian(
            race_positions={"VER": 1, "NOR": 3},
            quali_positions={"VER": 1, "NOR": 2},
        )

        # Both drivers did well in quali too - their ratings should be at least
        # as high as race-only, since the qualifying update adds positive signal
        assert race_and_quali["VER"] >= race_only["VER"], (
            "VER's rating dropped after adding a confirming qualifying update"
        )

    def test_combined_shift_bounded_vs_race_only(self):
        """The combined race+quali shift should not greatly exceed the race-only shift.

        With qualifying confidence at 0.15 and race at 0.35, qualifying should
        add a modest nudge - not double the effect. We allow up to 2x the race-only
        delta as the ceiling; anything beyond that suggests amplification.
        """
        initial_mu = 11.0
        race_only = self._run_with_real_bayesian(
            race_positions={"VER": 1, "NOR": 5},
            quali_positions=None,
        )
        race_and_quali = self._run_with_real_bayesian(
            race_positions={"VER": 1, "NOR": 5},
            quali_positions={"VER": 1, "NOR": 4},
        )

        race_only_delta = abs(race_only["VER"] - initial_mu)
        combined_delta = abs(race_and_quali["VER"] - initial_mu)

        assert combined_delta <= race_only_delta * 2.0, (
            f"Combined quali+race delta ({combined_delta:.4f}) more than 2x "
            f"the race-only delta ({race_only_delta:.4f}). "
            "Sequential updates may be amplifying rather than adding."
        )

    def test_conflicting_signals_moderated(self):
        """A driver who qualifies poorly but races well should land between the two signals.

        This is the Gasly-era pattern: strong race pace, weaker quali. The
        Bayesian rating after both updates should sit between what race-only
        and quali-only would produce individually.
        """
        race_only = self._run_with_real_bayesian(
            race_positions={"VER": 1, "NOR": 5},
            quali_positions=None,
        )
        # NOR qualifies P8 but races to P5 - modest conflict
        race_and_quali = self._run_with_real_bayesian(
            race_positions={"VER": 1, "NOR": 5},
            quali_positions={"VER": 1, "NOR": 8},
        )

        # NOR's combined rating should be lower than race-only because
        # the poor-ish quali result drags it back slightly
        assert race_and_quali["NOR"] <= race_only["NOR"] + 0.05, (
            "Conflicting qualifying signal should moderate NOR's rating, not inflate it"
        )


class TestBayesianSequentialUpdates:
    """Validate that sequential Bayesian updates across a race weekend don't amplify badly.

    These tests cover the order in which updater.py ingests completed session results
    into the ratings store - not the on-track session calendar order.

    On-track calendar:
      Normal:  FP1 -> FP2 -> FP3 -> Qualifying -> Race
      Sprint:  FP1 -> Sprint Qualifying -> Sprint Race -> Qualifying -> Race

    Ratings store update order (after sessions complete):
      Normal:  race results first, then qualifying appended in the same
               update_bayesian_driver_ratings() call (updater.py:425, 480).
      Sprint:  sprint (Saturday, update_from_sprint_race) -> race -> qualifying
               (Sunday, update_bayesian_driver_ratings - race then quali internally).
    """

    @staticmethod
    def _fresh_ranker() -> "BayesianDriverRanking":
        from src.models.bayesian import BayesianDriverRanking, DriverPrior

        priors = {
            "VER": DriverPrior("1", "VER", "Red Bull", "top", 11.0, 3.0),
            "NOR": DriverPrior("4", "NOR", "McLaren", "top", 11.0, 3.0),
            "PIA": DriverPrior("81", "PIA", "Williams", "midfield", 11.0, 3.0),
        }
        return BayesianDriverRanking(priors, grid_size=22)

    def test_normal_weekend_sequential_shift_bounded(self):
        """Normal weekend: race then qualifying update should not over-amplify.

        Qualifying and race at the same weekend are correlated - grid advantage
        carries over. The combined shift is allowed to be up to 1.5x the sum of
        the individual shifts, which catches runaway amplification while still
        permitting normal accumulation.

        Production order: race update -> qualifying update (see updater.py:425, 480).
        """

        race_obs = {"VER": 1, "NOR": 3, "PIA": 19}
        quali_obs = {"VER": 1, "NOR": 4, "PIA": 18}

        # Race only
        race_only = self._fresh_ranker()
        race_only.update(race_obs, session_name="Race", confidence=0.35)
        race_shift = {d: abs(race_only.ratings[d][0] - 11.0) for d in race_obs}

        # Qualifying only
        quali_only = self._fresh_ranker()
        quali_only.update(quali_obs, session_name="Qualifying", confidence=0.15)
        quali_shift = {d: abs(quali_only.ratings[d][0] - 11.0) for d in race_obs}

        # Combined: race first, then qualifying - production order
        combined = self._fresh_ranker()
        combined.update(race_obs, session_name="Race", confidence=0.35)
        combined.update(quali_obs, session_name="Qualifying", confidence=0.15)
        combined_shift = {d: abs(combined.ratings[d][0] - 11.0) for d in race_obs}

        for driver in race_obs:
            individual_sum = race_shift[driver] + quali_shift[driver]
            assert combined_shift[driver] <= individual_sum * 1.5, (
                f"{driver}: combined shift {combined_shift[driver]:.3f} exceeds "
                f"1.5x individual sum {individual_sum:.3f}"
            )

    def test_sprint_weekend_three_updates_bounded(self):
        """Sprint weekend: sprint -> race -> qualifying should not over-amplify.

        Sprint weekends have three Bayesian update calls across Saturday and Sunday
        (update_from_sprint_race, then update_bayesian_driver_ratings which does
        race then qualifying). All three are correlated - the same car advantage
        shows up in each session. The combined shift must stay within 1.5x the
        sum of the three individual shifts.
        """
        sprint_obs = {"VER": 1, "NOR": 2, "PIA": 18}
        race_obs = {"VER": 1, "NOR": 3, "PIA": 19}
        quali_obs = {"VER": 1, "NOR": 4, "PIA": 18}

        sprint_only = self._fresh_ranker()
        sprint_only.update(sprint_obs, session_name="Sprint", confidence=0.20)
        sprint_shift = {d: abs(sprint_only.ratings[d][0] - 11.0) for d in race_obs}

        race_only = self._fresh_ranker()
        race_only.update(race_obs, session_name="Race", confidence=0.35)
        race_shift = {d: abs(race_only.ratings[d][0] - 11.0) for d in race_obs}

        quali_only = self._fresh_ranker()
        quali_only.update(quali_obs, session_name="Qualifying", confidence=0.15)
        quali_shift = {d: abs(quali_only.ratings[d][0] - 11.0) for d in race_obs}

        # Combined: sprint -> race -> qualifying - production order for sprint weekends
        combined = self._fresh_ranker()
        combined.update(sprint_obs, session_name="Sprint", confidence=0.20)
        combined.update(race_obs, session_name="Race", confidence=0.35)
        combined.update(quali_obs, session_name="Qualifying", confidence=0.15)
        combined_shift = {d: abs(combined.ratings[d][0] - 11.0) for d in race_obs}

        for driver in race_obs:
            individual_sum = sprint_shift[driver] + race_shift[driver] + quali_shift[driver]
            assert combined_shift[driver] <= individual_sum * 1.5, (
                f"{driver}: sprint weekend combined shift {combined_shift[driver]:.3f} "
                f"exceeds 1.5x individual sum {individual_sum:.3f}"
            )

    def test_qualifying_update_smaller_than_race_for_same_positions(self):
        """Qualifying shift should be smaller than race shift for identical finishing order.

        Qualifying confidence (0.15) is lower than race confidence (0.35) by design - one flying lap in controlled conditions is a noisier signal than full race pace.
        """
        positions = {"VER": 1, "NOR": 5, "PIA": 15}

        quali_ranker = self._fresh_ranker()
        quali_ranker.update(positions, session_name="Qualifying", confidence=0.15)

        race_ranker = self._fresh_ranker()
        race_ranker.update(positions, session_name="Race", confidence=0.35)

        for driver in positions:
            quali_shift = abs(quali_ranker.ratings[driver][0] - 11.0)
            race_shift = abs(race_ranker.ratings[driver][0] - 11.0)
            assert quali_shift < race_shift, (
                f"{driver}: qualifying shift ({quali_shift:.3f}) should be smaller "
                f"than race shift ({race_shift:.3f})"
            )

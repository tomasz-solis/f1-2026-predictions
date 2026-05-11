"""Historical red tests for the superseded position-based de-carring formula.

These tests target a specific failure mode of the formula
    adjusted = raw - team_mean + field_mean
when used with finishing position as the input signal.

The bug: in a dominant team where both drivers finish near the front, the
team_mean is high, the field_mean is grid-median, and the adjusted rating
collapses toward field-median. The driver who actually performed near the
top of the grid gets a posterior update that pushes their rating toward
the middle.

Symmetric failure: in a backmarker team where both drivers finish near the
back, the adjusted rating inflates toward field-median and the weak driver
gets credit they did not earn.

The May 9 design replaced this position-scale contract with seconds-native
teammate residuals. Shared team movement belongs to `team_strength`; the
driver rating only carries teammate-relative residual evidence. The old
absolute-position magnitude assertions are kept as Phase 13 rewrite input,
not as active acceptance criteria.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.systems.updater import update_bayesian_driver_ratings

SUPERSEDED_POSITION_DECARRING_REASON = (
    "Superseded by the May 9 seconds-native orthogonality contract. "
    "Dominant/backmarker team movement belongs to team_strength; these "
    "position-scale rating_mu assertions are Phase 13 rewrite input."
)


# ---------------------------------------------------------------------------
# Helpers (modeled on tests/test_pace_update_teammate_relative.py)
# ---------------------------------------------------------------------------


def _make_race_results(
    positions: dict[str, int],
    statuses: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a minimal race results DataFrame from driver_code -> position."""
    rows = []
    for code, pos in positions.items():
        row: dict = {"Abbreviation": code, "Position": pos, "race_name": "Test GP"}
        if statuses:
            row["Status"] = statuses.get(code, "Finished")
        rows.append(row)
    return pd.DataFrame(rows)


def _make_driver_entry(
    quali_pace: float = 0.50,
    race_pace: float = 0.50,
    rating_mu: float = 11.0,
    rating_sigma: float = 2.5,
) -> dict:
    """Build a driver characteristics entry with neutral defaults."""
    return {
        "pace": {"quali_pace": quali_pace, "race_pace": race_pace},
        "racecraft": {"skill_score": 0.50},
        "bayesian": {"rating_mu": rating_mu, "rating_sigma": rating_sigma},
    }


def _run_update(
    driver_entries: dict[str, dict],
    race_positions: dict[str, int],
    lineups: dict[str, list[str]] | None = None,
) -> tuple[dict[str, dict], dict[str, tuple[float, float]]]:
    """Run one update cycle. Returns (mutated drivers, post-update bayesian ratings).

    The Bayesian model is wired up so its posterior gets recorded — unlike the
    earlier pace-only test which mocked it out — because we need to assert
    against the rating mu, not just the pace fields.
    """
    from src.models.bayesian import BayesianDriverRanking, DriverPrior

    driver_payload: dict = {"drivers": driver_entries}

    priors = {
        code: DriverPrior(
            driver_number=code,
            driver_code=code,
            team=next((t for t, drs in (lineups or {}).items() if code in drs), "Unknown"),
            team_tier="midfield",
            mu=entry["bayesian"]["rating_mu"],
            sigma=entry["bayesian"]["rating_sigma"],
        )
        for code, entry in driver_entries.items()
    }
    real_bayesian = BayesianDriverRanking(priors=priors, grid_size=len(driver_entries))

    race_results = _make_race_results(race_positions)

    with (
        patch(
            "src.models.priors_factory.PriorsFactory.create_priors",
            return_value=priors,
        ),
        patch(
            "src.systems.updater.BayesianDriverRanking",
            return_value=real_bayesian,
        ),
        patch("src.utils.lineups.load_current_lineups", return_value=lineups),
        patch(
            "src.systems.updater._load_driver_characteristics_payload",
            return_value=driver_payload,
        ),
        patch("src.systems.updater._persist_driver_characteristics_payload"),
    ):
        update_bayesian_driver_ratings(race_results, qualifying_results=None)

    return driver_payload["drivers"], dict(real_bayesian.ratings)


# ---------------------------------------------------------------------------
# Bug 1: dominant team compresses winner's signal
# ---------------------------------------------------------------------------


class TestDominantTeamMagnitude:
    """A driver who wins from a dominant car must not get a near-neutral update.

    This is the PIA-in-McLaren shape: McLaren goes 1-2, both cars near the top,
    and the de-carring pulls the winner's adjusted signal down to roughly
    field-median. The Bayesian posterior for a winning driver should reflect
    a strong observation, not a median one.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=SUPERSEDED_POSITION_DECARRING_REASON,
    )
    def test_winner_in_dominant_team_gets_meaningful_upward_update(self):
        """P1 in a 1-2 finish should produce a posterior mu noticeably above prior.

        Setup: McLaren goes P1/P2, every other team is field. The winner's
        prior is 11.0 (median). A correct update should move the posterior
        meaningfully toward the top of the rating scale (>~14 on a 22-grid).

        With the current de-carring formula, both McLaren drivers get adjusted
        ratings near the field mean (~11.5), so the posterior barely moves.
        """
        lineups = {
            "McLaren": ["NOR", "PIA"],
            "Ferrari": ["LEC", "SAI"],
            "Red Bull": ["VER", "TSU"],
            "Mercedes": ["RUS", "ANT"],
            "Williams": ["ALB", "COL"],
            "Aston": ["ALO", "STR"],
            "Alpine": ["GAS", "DOO"],
            "Haas": ["BEA", "OCO"],
            "RB": ["LAW", "HAD"],
            "Audi": ["HUL", "BOR"],
            "Cadillac": ["PER", "BOT"],
        }
        all_drivers = [d for team_drivers in lineups.values() for d in team_drivers]
        drivers = {code: _make_driver_entry() for code in all_drivers}

        # McLaren 1-2, others in arbitrary mid/back positions
        race_positions = {
            "NOR": 1,
            "PIA": 2,
            "LEC": 3,
            "SAI": 4,
            "VER": 5,
            "TSU": 6,
            "RUS": 7,
            "ANT": 8,
            "ALB": 9,
            "COL": 10,
            "ALO": 11,
            "STR": 12,
            "GAS": 13,
            "DOO": 14,
            "BEA": 15,
            "OCO": 16,
            "LAW": 17,
            "HAD": 18,
            "HUL": 19,
            "BOR": 20,
            "PER": 21,
            "BOT": 22,
        }

        _, ratings = _run_update(drivers, race_positions, lineups=lineups)

        nor_mu, _ = ratings["NOR"]
        pia_mu, _ = ratings["PIA"]
        prior_mu = 11.0

        # Winner of the race must move materially above prior. We choose a
        # threshold that is conservative for "strong observation": a P1
        # finish on a 22-driver grid maps to a raw rating of 22, and with
        # confidence 0.35 a single update should comfortably move a prior
        # at 11 toward at least 14.
        assert nor_mu >= 14.0, (
            f"NOR won the race in a dominant car but posterior mu is {nor_mu:.3f} "
            f"(prior was {prior_mu}). De-carring formula compresses the signal."
        )
        # P2 in the same dominant team should also move up, just less than the winner.
        assert pia_mu >= 12.5, (
            f"PIA finished P2 in a dominant car but posterior mu is {pia_mu:.3f}. "
            "Driver pulled toward field-median by team-centering."
        )

    @pytest.mark.xfail(
        strict=True,
        reason=SUPERSEDED_POSITION_DECARRING_REASON,
    )
    def test_winner_in_dominant_team_gets_higher_pace_than_field_median(self):
        """The race_pace EMA for the race winner must clear neutral by a margin.

        Mirrors the magnitude check at the pace-EMA layer rather than the
        Bayesian layer. A driver who wins from a dominant car should not
        end up with race_pace barely above 0.5.
        """
        lineups = {
            "McLaren": ["NOR", "PIA"],
            "Ferrari": ["LEC", "SAI"],
            "Red Bull": ["VER", "TSU"],
            "Williams": ["ALB", "COL"],
        }
        all_drivers = [d for team_drivers in lineups.values() for d in team_drivers]
        drivers = {code: _make_driver_entry(race_pace=0.50) for code in all_drivers}
        # Pad to a full 22-driver grid so field_mean is realistic
        for i in range(8):
            code = f"X{i:02d}"
            drivers[code] = _make_driver_entry(race_pace=0.50)
            lineups[f"Pad{i // 2}"] = lineups.get(f"Pad{i // 2}", []) + [code]

        race_positions = {
            "NOR": 1,
            "PIA": 2,
            "LEC": 3,
            "SAI": 4,
            "VER": 5,
            "TSU": 6,
            "ALB": 7,
            "COL": 8,
        }
        for i, code in enumerate(c for c in drivers if c.startswith("X")):
            race_positions[code] = 9 + i

        result, _ = _run_update(drivers, race_positions, lineups=lineups)

        nor_pace = result["NOR"]["pace"]["race_pace"]
        # Winner of the race in a dominant pair should have race_pace clearly
        # above neutral, not stuck at 0.51-ish.
        assert nor_pace >= 0.58, (
            f"NOR (P1 in dominant pair) race_pace = {nor_pace:.3f}; "
            "de-carring is pulling it toward field-median."
        )


# ---------------------------------------------------------------------------
# Bug 2 (symmetric): dominated team inflates weak signal
# ---------------------------------------------------------------------------


class TestDominatedTeamMagnitude:
    """A driver who finishes near the back in a backmarker pair must not be inflated.

    Symmetric to the dominant-team case. If both teammates are at the back,
    the team_mean is low, and `raw - team_mean + field_mean` re-centers them
    near the field median, producing an artificially favorable update.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=SUPERSEDED_POSITION_DECARRING_REASON,
    )
    def test_backmarker_pair_does_not_lift_posteriors_above_prior(self):
        """Drivers finishing P21/P22 should not see posterior mu rise above prior."""
        lineups = {
            "McLaren": ["NOR", "PIA"],
            "Ferrari": ["LEC", "SAI"],
            "Red Bull": ["VER", "TSU"],
            "Mercedes": ["RUS", "ANT"],
            "Williams": ["ALB", "COL"],
            "Aston": ["ALO", "STR"],
            "Alpine": ["GAS", "DOO"],
            "Haas": ["BEA", "OCO"],
            "RB": ["LAW", "HAD"],
            "Audi": ["HUL", "BOR"],
            "Cadillac": ["PER", "BOT"],
        }
        all_drivers = [d for team_drivers in lineups.values() for d in team_drivers]
        drivers = {code: _make_driver_entry() for code in all_drivers}

        race_positions = {
            "NOR": 1,
            "PIA": 2,
            "LEC": 3,
            "SAI": 4,
            "VER": 5,
            "TSU": 6,
            "RUS": 7,
            "ANT": 8,
            "ALB": 9,
            "COL": 10,
            "ALO": 11,
            "STR": 12,
            "GAS": 13,
            "DOO": 14,
            "BEA": 15,
            "OCO": 16,
            "LAW": 17,
            "HAD": 18,
            "HUL": 19,
            "BOR": 20,
            # Cadillac at the back
            "PER": 21,
            "BOT": 22,
        }

        _, ratings = _run_update(drivers, race_positions, lineups=lineups)

        per_mu, _ = ratings["PER"]
        bot_mu, _ = ratings["BOT"]
        prior_mu = 11.0

        # Both drivers finished P21/P22. The posterior should reflect a poor
        # observation, not a median one. Allowing some Bayesian shrinkage,
        # but not letting them stay at or above prior.
        assert per_mu < prior_mu, (
            f"PER (P21 in backmarker pair) posterior mu = {per_mu:.3f}; "
            f"de-carring is inflating a weak observation toward prior."
        )
        assert bot_mu < prior_mu, f"BOT (P22 in backmarker pair) posterior mu = {bot_mu:.3f}."


# ---------------------------------------------------------------------------
# Bug 3 (regression guard): mid-pack noise must not drive teammates apart
# ---------------------------------------------------------------------------


class TestMidPackNoiseStability:
    """When two teammates are equally good and finishing-order is noise, posteriors should not drift.

    This guards the fix from over-correcting. If the new mechanism amplifies
    raw finishing-position differences instead of compressing them via
    teammate-centering, two teammates who are statistically identical will
    drift apart over multiple races. We don't want that either.
    """

    def test_equal_teammates_in_noisy_finishes_stay_close_after_three_races(self):
        """Three races where teammates swap P10-P11 in noise should not separate them."""
        lineups = {"Williams": ["ALB", "COL"]}
        for i in range(10):
            lineups[f"Pad{i}"] = [f"X{i:02d}A", f"X{i:02d}B"]

        all_drivers = [d for drs in lineups.values() for d in drs]
        drivers = {code: _make_driver_entry() for code in all_drivers}

        # Build three races with ALB/COL alternating P10/P11 plus one P9/P12 case
        race_specs = [
            {"ALB": 10, "COL": 11},
            {"ALB": 11, "COL": 10},
            {"ALB": 9, "COL": 12},
        ]
        # Pad finishing positions
        for race_spec in race_specs:
            others = [c for c in all_drivers if c not in race_spec]
            slot = 1
            for code in others:
                while slot in race_spec.values():
                    slot += 1
                race_spec[code] = slot
                slot += 1

        latest_ratings = None
        for race_spec in race_specs:
            _, latest_ratings = _run_update(drivers, race_spec, lineups=lineups)

        assert latest_ratings is not None
        alb_mu, _ = latest_ratings["ALB"]
        col_mu, _ = latest_ratings["COL"]

        # After three races of essentially equal performance, posterior mus
        # should be within 1.0 rating point of each other.
        assert abs(alb_mu - col_mu) <= 1.0, (
            f"ALB and COL diverged to {alb_mu:.3f} vs {col_mu:.3f} on noise alone. "
            "The update mechanism is amplifying finishing-order randomness."
        )


# ---------------------------------------------------------------------------
# Sanity: prior and posterior identity guard
# ---------------------------------------------------------------------------


class TestTeammatePairAtFieldMedianDoesNotDrift:
    """When a teammate pair finishes at the field median, posteriors should not drift.

    This is the only "neutral" case the de-carring formula handles correctly:
    when team_mean equals field_mean for that pair, adjusted == raw, and the
    Bayesian update is driven by raw position vs prior. We use a pair finishing
    P11/P12 in the middle of a 22-driver grid — exactly at field median.

    NOTE: this test deliberately does NOT make the *whole grid* neutral.
    Earlier drafts did, and discovered that the de-carring formula compresses
    teammates near each other independent of where in the grid the pair sits.
    The bug we are catching elsewhere; this test guards a different invariant:
    pairs that finish AT the field median should give a near-neutral update.
    """

    def test_pair_at_field_median_no_meaningful_drift(self):
        """A teammate pair finishing P11/P12 in a 22-driver grid should produce ~no update."""
        lineups = {
            "McLaren": ["NOR", "PIA"],
            "Ferrari": ["LEC", "SAI"],
            "Red Bull": ["VER", "TSU"],
            "Mercedes": ["RUS", "ANT"],
            # Test pair sits dead center
            "Haas": ["BEA", "OCO"],
            "RB": ["LAW", "HAD"],
            "Audi": ["HUL", "BOR"],
            "Cadillac": ["PER", "BOT"],
            "Aston": ["ALO", "STR"],
            "Alpine": ["GAS", "DOO"],
            "Williams": ["ALB", "COL"],
        }
        all_drivers = [d for team_drivers in lineups.values() for d in team_drivers]
        drivers = {code: _make_driver_entry(rating_mu=11.5) for code in all_drivers}

        # Haas pair lands at P11/P12 — exactly the field median for grid_size=22.
        # Other teams are arranged so the field is symmetric.
        race_positions = {
            "NOR": 1,
            "PIA": 2,
            "LEC": 3,
            "SAI": 4,
            "VER": 5,
            "TSU": 6,
            "RUS": 7,
            "ANT": 8,
            "ALO": 9,
            "STR": 10,
            "BEA": 11,
            "OCO": 12,  # the pair under test
            "GAS": 13,
            "DOO": 14,
            "ALB": 15,
            "COL": 16,
            "LAW": 17,
            "HAD": 18,
            "HUL": 19,
            "BOR": 20,
            "PER": 21,
            "BOT": 22,
        }

        _, ratings = _run_update(drivers, race_positions, lineups=lineups)

        bea_mu, _ = ratings["BEA"]
        oco_mu, _ = ratings["OCO"]
        # Both should stay essentially at prior since they delivered exactly
        # field-median performance.
        assert abs(bea_mu - 11.5) <= 0.3, (
            f"BEA: posterior mu = {bea_mu:.3f} drifted from neutral observation"
        )
        assert abs(oco_mu - 11.5) <= 0.3, (
            f"OCO: posterior mu = {oco_mu:.3f} drifted from neutral observation"
        )

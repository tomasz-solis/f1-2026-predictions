"""
Tests for Bayesian driver ranking system.
"""

import pytest

from src.models.bayesian import BayesianDriverRanking, DriverPrior


@pytest.mark.unit
class TestDriverPrior:
    """Test DriverPrior dataclass."""

    def test_driver_prior_creation(self):
        prior = DriverPrior(
            driver_number="1",
            driver_code="VER",
            team="Red Bull Racing",
            team_tier="top",
            mu=18.0,
            sigma=2.0,
        )
        assert prior.driver_number == "1"
        assert prior.mu == 18.0
        assert prior.sigma == 2.0


@pytest.mark.unit
class TestBayesianDriverRanking:
    """Test BayesianDriverRanking core functionality."""

    def test_initialization(self, sample_priors):
        ranker = BayesianDriverRanking(sample_priors)

        assert len(ranker.ratings) == 4
        assert ranker.ratings["1"] == (18.0, 2.0)
        assert ranker.ratings["4"] == (17.0, 2.5)
        assert len(ranker.history) == 0

    def test_update_reduces_uncertainty(self, sample_priors):
        """Bayesian update should reduce sigma (uncertainty)."""
        ranker = BayesianDriverRanking(sample_priors)
        initial_mu, initial_sigma = ranker.ratings["1"]

        # VER wins (position 1)
        ranker.update({"1": 1}, "test_race", confidence=1.0)

        updated_mu, updated_sigma = ranker.ratings["1"]

        # Sigma should decrease after observing data
        assert updated_sigma < initial_sigma

    def test_update_shifts_mean_toward_observation(self, sample_priors):
        """Update should move mu toward observed performance."""
        ranker = BayesianDriverRanking(sample_priors)

        # BOT (backmarker with mu=10) finishes P3 (way better than expected)
        # Expected rating for P3 = 21 - 3 = 18
        ranker.update({"77": 3}, "test_race", confidence=1.0)

        updated_mu, _ = ranker.ratings["77"]

        # Mu should increase (BOT performed better than prior)
        assert updated_mu > 10.0
        # But not jump all the way to 18 in one update
        assert updated_mu < 18.0

    def test_high_confidence_update_stronger_than_low(self, sample_priors):
        """High confidence observations should move beliefs more."""
        ranker_high = BayesianDriverRanking(sample_priors.copy())
        ranker_low = BayesianDriverRanking(sample_priors.copy())

        # Same observation, different confidence
        ranker_high.update({"1": 10}, "race", confidence=1.0)
        ranker_low.update({"1": 10}, "practice", confidence=0.2)

        mu_high, _ = ranker_high.ratings["1"]
        mu_low, _ = ranker_low.ratings["1"]

        # High confidence should shift belief more
        assert abs(mu_high - 18.0) > abs(mu_low - 18.0)

    def test_shock_factor_activates_on_outlier(self, sample_priors):
        """Shock factor should trigger when result is far from prior."""
        ranker = BayesianDriverRanking(sample_priors)

        # VER (top driver, mu=18) finishes dead last (P20)
        # This is ~8 positions worse than expected
        ranker.update({"1": 20}, "crash_race", confidence=1.0)

        # Check that update was recorded
        assert len(ranker.history) == 1
        update = ranker.history[0]

        # Shock factor should be positive (outlier detected)
        assert update.shock_factor > 0

    def test_multiple_updates_accumulate_history(self, sample_priors):
        """Each update should be logged."""
        ranker = BayesianDriverRanking(sample_priors)

        ranker.update({"1": 1, "4": 2}, "race_1", confidence=1.0)
        ranker.update({"1": 3, "4": 1}, "race_2", confidence=1.0)

        assert len(ranker.history) == 4  # 2 drivers x 2 races

    def test_get_current_ratings_returns_dataframe(self, sample_priors):
        """Should return properly formatted DataFrame."""
        ranker = BayesianDriverRanking(sample_priors)
        df = ranker.get_current_ratings()

        assert len(df) == 4
        assert "driver_code" in df.columns
        assert "rating_mu" in df.columns
        assert "expected_position" in df.columns

        # Should be sorted by rating (descending)
        assert df.iloc[0]["driver_code"] == "VER"  # Highest rated

    def test_expected_position_calculation(self, sample_priors):
        """Expected position should be inverse of rating."""
        ranker = BayesianDriverRanking(sample_priors)
        df = ranker.get_current_ratings()

        # VER has mu=18, expected pos should be ~3 (21-18)
        ver_row = df[df["driver_code"] == "VER"].iloc[0]
        assert 1 <= ver_row["expected_position"] <= 5

        # BOT has mu=10, expected pos should be ~11 (21-10)
        bot_row = df[df["driver_code"] == "BOT"].iloc[0]
        assert 9 <= bot_row["expected_position"] <= 13

    def test_confidence_intervals_calculated(self, sample_priors):
        """CI bounds should exist and be reasonable."""
        ranker = BayesianDriverRanking(sample_priors)
        df = ranker.get_current_ratings()

        for _, row in df.iterrows():
            assert row["lower_ci"] >= 1
            assert row["upper_ci"] <= 20
            assert row["lower_ci"] <= row["expected_position"]
            assert row["upper_ci"] >= row["expected_position"]

    def test_get_history_df_returns_updates(self, sample_priors):
        """History export should work."""
        ranker = BayesianDriverRanking(sample_priors)
        ranker.update({"1": 1}, "test", confidence=1.0)

        history_df = ranker.get_history_df()
        assert len(history_df) == 1
        assert "session_name" in history_df.columns
        assert history_df.iloc[0]["session_name"] == "test"

    def test_update_ignores_unknown_drivers(self, sample_priors):
        """Should skip drivers not in priors."""
        ranker = BayesianDriverRanking(sample_priors)

        # Try to update non-existent driver
        ranker.update({"999": 1}, "race", confidence=1.0)

        # Should not crash, just ignore
        assert len(ranker.history) == 0

    def test_process_noise_injection(self, sample_priors):
        """Process noise should widen uncertainty between races."""
        ranker = BayesianDriverRanking(sample_priors)

        # Get initial sigma
        _, initial_sigma = ranker.ratings["1"]

        # Update with exact prior expectation (minimal learning)
        # VER (mu=18) finishes P3 (rating 18)
        ranker.update({"1": 3}, "race", confidence=1.0)

        # Even with perfect prediction, sigma should decrease less
        # due to process noise widening it first
        _, updated_sigma = ranker.ratings["1"]
        assert updated_sigma < initial_sigma  # Still decreases overall
        assert updated_sigma > 0.5  # But not too confident

    def test_volatility_prevents_overconfidence(self, sample_priors):
        """After many consistent results, sigma shouldn't go to zero."""
        ranker = BayesianDriverRanking(sample_priors)

        # Simulate 10 races where VER always wins
        for i in range(10):
            ranker.update({"1": 1}, f"race_{i}", confidence=1.0)

        _, final_sigma = ranker.ratings["1"]

        # Sigma should be small but not zero (volatility prevents collapse)
        assert final_sigma > 0.3
        assert final_sigma < 2.0

    def test_sigma_floor_prevents_total_collapse(self, sample_priors):
        """Repeated high-confidence updates should respect the sigma floor."""
        ranker = BayesianDriverRanking(sample_priors)

        for i in range(200):
            ranker.update({"1": 1}, f"race_{i}", confidence=100.0)

        _, final_sigma = ranker.ratings["1"]
        assert final_sigma >= 0.05

    def test_consistent_loser_rating_drops(self, sample_priors):
        """Driver who consistently underperforms should see rating drop."""
        ranker = BayesianDriverRanking(sample_priors)

        initial_mu, _ = ranker.ratings["1"]  # VER starts at 18

        # VER finishes P10 five times in a row (bad for him)
        for i in range(5):
            ranker.update({"1": 10}, f"race_{i}", confidence=1.0)

        final_mu, _ = ranker.ratings["1"]

        # Rating should drop significantly
        assert final_mu < initial_mu - 3.0

    def test_update_from_session_alias_works(self, sample_priors):
        """Backward compatibility alias should work."""
        ranker = BayesianDriverRanking(sample_priors)

        ranker.update_from_session({"1": 1}, "race", confidence=1.0)

        assert len(ranker.history) == 1

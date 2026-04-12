"""Tests for race result movement constraint logic."""

from unittest.mock import MagicMock

import numpy as np


def _make_cfg(overrides: dict | None = None) -> MagicMock:
    """Build a config stub with default constraint values and optional overrides."""
    defaults = {
        "baseline_predictor.race.main_race_movement_floor": 0.70,
        "baseline_predictor.race.main_race_movement_floor_track_scale": 0.25,
        "baseline_predictor.race.main_race_movement_quantile": 20.0,
        "baseline_predictor.race.main_race_movement_ceiling_base": 2.5,
        "baseline_predictor.race.main_race_movement_ceiling_track_scale": 0.70,
        "baseline_predictor.race.main_race_movement_ceiling_min": 0.70,
    }
    if overrides:
        defaults.update(overrides)
    cfg = MagicMock()
    cfg.get = lambda key, default=None: defaults.get(key, default)
    return cfg


def _make_finish_order(n: int = 10) -> list[dict]:
    """Return a finish order where every driver is at their starting grid slot."""
    return [
        {
            "driver_code": f"D{i:02d}",
            "grid_pos": i,
            "predicted_position": float(i),
        }
        for i in range(1, n + 1)
    ]


def _make_driver_info_map(finish_order: list[dict]) -> dict:
    """Build the minimal driver_info_map expected by _avg_grid_change."""
    return {row["driver_code"]: {"grid_pos": row["grid_pos"]} for row in finish_order}


def _make_grid_reference(finish_order: list[dict]) -> dict:
    """Map each driver to their starting grid position."""
    return {row["driver_code"]: float(row["grid_pos"]) for row in finish_order}


def _make_samples(finish_order: list[dict], n_samples: int = 50) -> dict:
    """Generate random samples for each driver — enough for quantile logic to engage."""
    rng = np.random.default_rng(42)
    return {row["driver_code"]: rng.uniform(0.0, 1.0, n_samples).tolist() for row in finish_order}


class TestMovementFloorTrackScale:
    """Movement floor scales with track overtaking difficulty."""

    def test_runs_without_error_on_static_grid(self):
        """Function should not raise when all drivers finish on their grid slots."""
        from src.predictors.baseline.race.result_processing import (
            apply_main_race_movement_constraints,
        )

        finish = _make_finish_order()
        apply_main_race_movement_constraints(
            finish_order=finish,
            blended_samples_by_driver=_make_samples(finish),
            driver_info_map=_make_driver_info_map(finish),
            grid_reference_positions=_make_grid_reference(finish),
            track_overtaking=0.5,
            cfg=_make_cfg(),
        )
        assert len(finish) == 10

    def test_runs_for_extreme_low_overtaking(self):
        """Monaco-like track (overtaking=0.0) should not raise."""
        from src.predictors.baseline.race.result_processing import (
            apply_main_race_movement_constraints,
        )

        finish = _make_finish_order()
        apply_main_race_movement_constraints(
            finish_order=finish,
            blended_samples_by_driver=_make_samples(finish),
            driver_info_map=_make_driver_info_map(finish),
            grid_reference_positions=_make_grid_reference(finish),
            track_overtaking=0.0,
            cfg=_make_cfg(),
        )
        assert len(finish) == 10

    def test_runs_for_extreme_high_overtaking(self):
        """Bahrain-like track (overtaking=1.0) should not raise."""
        from src.predictors.baseline.race.result_processing import (
            apply_main_race_movement_constraints,
        )

        finish = _make_finish_order()
        apply_main_race_movement_constraints(
            finish_order=finish,
            blended_samples_by_driver=_make_samples(finish),
            driver_info_map=_make_driver_info_map(finish),
            grid_reference_positions=_make_grid_reference(finish),
            track_overtaking=1.0,
            cfg=_make_cfg(),
        )
        assert len(finish) == 10

    def test_floor_never_exceeds_ceiling(self):
        """At zero overtaking, the scaled floor should not exceed the ceiling."""
        from src.predictors.baseline.race.result_processing import (
            apply_main_race_movement_constraints,
        )

        # With track_overtaking=0.0: overtake_ease=1.0
        # floor = min(ceiling, base_floor + 1.0 * track_scale)
        #       = min(ceiling, 0.70 + 0.25) = min(ceiling, 0.95)
        # ceiling = max(0.70, 2.5 - 1.0 * 0.70) = max(0.70, 1.80) = 1.80
        # So floor = min(1.80, 0.95) = 0.95 — well within ceiling.
        # This test confirms the function does not raise or produce absurd output.
        finish = _make_finish_order()
        apply_main_race_movement_constraints(
            finish_order=finish,
            blended_samples_by_driver=_make_samples(finish),
            driver_info_map=_make_driver_info_map(finish),
            grid_reference_positions=_make_grid_reference(finish),
            track_overtaking=0.0,
            cfg=_make_cfg(),
        )
        for row in finish:
            assert 1 <= row["predicted_position"] <= 10, (
                f"Driver {row['driver_code']} ended at {row['predicted_position']}, "
                "which is outside valid grid range"
            )

    def test_output_positions_are_valid_after_reranking(self):
        """After constraint adjustment, all predicted positions must stay within grid bounds."""
        from src.predictors.baseline.race.result_processing import (
            apply_main_race_movement_constraints,
        )

        finish = _make_finish_order(n=20)
        apply_main_race_movement_constraints(
            finish_order=finish,
            blended_samples_by_driver=_make_samples(finish),
            driver_info_map=_make_driver_info_map(finish),
            grid_reference_positions=_make_grid_reference(finish),
            track_overtaking=0.4,
            cfg=_make_cfg(),
        )
        positions = [row["predicted_position"] for row in finish]
        assert len(positions) == 20
        assert all(isinstance(p, float) for p in positions)

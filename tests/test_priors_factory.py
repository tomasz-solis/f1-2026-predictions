"""Tests for Bayesian prior construction from stored season artifacts."""

import json

from src.models.priors_factory import PriorsFactory


def test_priors_factory_resolves_second_year_driver_from_debut_year(tmp_path, patcher):
    """A stale stored rookie label should not give a second-year driver rookie uncertainty."""
    data_dir = tmp_path / "processed"
    driver_dir = data_dir / "driver_characteristics"
    car_dir = data_dir / "car_characteristics"
    driver_dir.mkdir(parents=True)
    car_dir.mkdir(parents=True)

    (driver_dir / "2026_driver_characteristics.json").write_text(
        json.dumps(
            {
                "year": 2026,
                "drivers": {
                    "ANT": {
                        "number": 12,
                        "experience": {
                            "tier": "rookie",
                            "years_of_experience": 0,
                            "debut_year": 2025,
                        },
                        "racecraft": {"skill_score": 0.55},
                        "pace": {"quali_pace": 0.58, "race_pace": 0.52},
                        "dnf_risk": {"dnf_rate": 0.10},
                    }
                },
            }
        )
    )
    (car_dir / "2026_car_characteristics.json").write_text(
        json.dumps(
            {
                "year": 2026,
                "teams": {
                    "Mercedes": {
                        "base_rating": 15.0,
                        "tier": "top",
                        "stability": 0.8,
                    }
                },
            }
        )
    )
    patcher.setattr(
        "src.utils.lineups.load_current_lineups",
        lambda: {"Mercedes": ["ANT"]},
    )

    priors = PriorsFactory(data_dir=data_dir, season_year=2026).create_priors()

    assert priors["ANT"].sigma == 3.0

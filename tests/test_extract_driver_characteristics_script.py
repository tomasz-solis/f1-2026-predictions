"""Tests for the driver-characteristics extraction script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script_module():
    """Load the driver-characteristics script as a module."""
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "extract_driver_characteristics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "extract_driver_characteristics_script", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_seed_initial_bayesian_state_omits_legacy_normalized_skill_score():
    """Seeded Bayesian payloads should avoid legacy derived fields."""
    module = _load_script_module()
    final_ratings = {
        "NOR": {
            "racecraft": {"skill_score": 0.82},
        }
    }

    module._seed_initial_bayesian_state(final_ratings, grid_size=20)

    bayesian = final_ratings["NOR"]["bayesian"]
    assert bayesian["rating_mu"] > 0
    assert bayesian["rating_sigma"] == module._DEFAULT_BAYESIAN_SIGMA
    assert bayesian["seeded_from"] == "extraction_prior"
    assert "normalized_skill_score" not in bayesian


def test_load_lineup_seed_context_returns_optional_season(tmp_path):
    """Lineup seed loading should preserve the target season when present."""
    module = _load_script_module()
    lineup_file = tmp_path / "lineups.json"
    lineup_file.write_text(
        json.dumps(
            {
                "season": 2025,
                "current_lineups": {"Audi": ["HUL", "BOR"]},
            }
        )
    )

    current_lineups, lineup_season = module._load_lineup_seed_context(lineup_file)

    assert lineup_season == 2025
    assert current_lineups == {"Audi": ["HUL", "BOR"]}


def test_build_team_based_prior_entry_uses_requested_rookie_debut_year():
    """Missing lineup drivers should inherit the target season, not a hard-coded year."""
    module = _load_script_module()

    entry = module._build_team_based_prior_entry(
        driver_code="BOR",
        team_name="Audi",
        teammate_ratings=[0.80],
        rookie_debut_year=2025,
    )

    assert entry["prior_source"] == "team_based_prior"
    assert entry["experience"]["debut_year"] == 2025
    assert entry["racecraft"]["skill_score"] == 0.72

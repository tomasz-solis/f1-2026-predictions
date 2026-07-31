from __future__ import annotations

from itertools import combinations

import pytest

from src.models.challenger_variants import (
    CHAMPION_VARIANT,
    VARIANT_COMPONENTS,
    component_enabled,
    known_variant_ids,
    resolve_model_variant,
    variant_id_for_components,
)


class _Config:
    def __init__(self, variant: str | None = None):
        self.variant = variant

    def get(self, key: str, default=None):
        assert key == "baseline_predictor.model_variant"
        return self.variant if self.variant is not None else default


def test_challenger_variant_defaults_to_champion() -> None:
    config = _Config()

    assert resolve_model_variant(config) == CHAMPION_VARIANT
    assert not component_enabled(config, "q0")
    assert not component_enabled(config, "r1")


def test_full_challenger_exposes_each_selected_component() -> None:
    config = _Config("full_challenger")

    assert component_enabled(config, "q0")
    assert component_enabled(config, "q1")
    assert component_enabled(config, "r0")
    assert component_enabled(config, "r1")
    assert component_enabled(config, "r2_source_anchor")


def test_unknown_variant_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown baseline_predictor.model_variant"):
        resolve_model_variant(_Config("typo"))


def test_variant_ids_are_stable_and_champion_first() -> None:
    variants = known_variant_ids()

    assert variants[0] == "champion"
    assert len(variants) == len(set(variants))
    assert VARIANT_COMPONENTS["r1_r2_no_anchor"] == frozenset({"r1", "r2_no_anchor"})
    assert VARIANT_COMPONENTS["r1_r2_source_anchor"] == frozenset({"r1", "r2_source_anchor"})
    assert VARIANT_COMPONENTS["q0_q1_r0_r1"] == frozenset({"q0", "q1", "r0", "r1"})
    assert VARIANT_COMPONENTS["full_no_anchor"] == frozenset(
        {"q0", "q1", "r0", "r1", "r2_no_anchor"}
    )


def test_registry_covers_every_valid_component_subset_exactly_once() -> None:
    base_components = ("q0", "q1", "r0", "r1")
    expected: set[frozenset[str]] = set()
    for component_count in range(len(base_components) + 1):
        for selected in combinations(base_components, component_count):
            base = frozenset(selected)
            for r2_component in (None, "r2_no_anchor", "r2_source_anchor"):
                expected.add(base | ({r2_component} if r2_component is not None else set()))

    registered = list(VARIANT_COMPONENTS.values())
    assert len(VARIANT_COMPONENTS) == 48
    assert len(set(registered)) == len(registered)
    assert set(registered) == expected
    for components in expected:
        assert VARIANT_COMPONENTS[variant_id_for_components(components)] == components


def test_component_resolver_rejects_unknown_or_mutually_exclusive_r2_modes() -> None:
    with pytest.raises(ValueError, match="Unknown challenger components"):
        variant_id_for_components({"q1", "made_up"})
    with pytest.raises(ValueError, match="mutually exclusive"):
        variant_id_for_components({"r2_no_anchor", "r2_source_anchor"})

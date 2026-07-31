"""Explicit, champion-default component registry for model challengers.

The registry contains every valid subset of independently promotable challenger
components.  R2's two post-simulation anchor modes are mutually exclusive.  A
small set of descriptive identifiers is retained for compatibility; remaining
subsets receive deterministic component-based identifiers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any

CHAMPION_VARIANT = "champion"

BASE_COMPONENTS = ("q0", "q1", "r0", "r1")
R2_COMPONENTS = ("r2_no_anchor", "r2_source_anchor")
KNOWN_COMPONENTS = frozenset((*BASE_COMPONENTS, *R2_COMPONENTS))

_DESCRIPTIVE_VARIANTS: tuple[tuple[str, frozenset[str]], ...] = (
    (CHAMPION_VARIANT, frozenset()),
    ("q0_driver_state", frozenset({"q0"})),
    ("q1_qualifying_practice", frozenset({"q1"})),
    ("q0_q1_qualifying", frozenset({"q0", "q1"})),
    ("r0_long_run", frozenset({"r0"})),
    ("r1_joint_grid", frozenset({"r1"})),
    ("q1_r1_joint_grid", frozenset({"q1", "r1"})),
    ("r0_r1_joint_grid", frozenset({"r0", "r1"})),
    ("q0_q1_r1_joint_grid", frozenset({"q0", "q1", "r1"})),
    ("q0_q1_r0_r1", frozenset({"q0", "q1", "r0", "r1"})),
    ("r2_no_anchor", frozenset({"r2_no_anchor"})),
    ("r2_source_anchor", frozenset({"r2_source_anchor"})),
    ("r1_r2_no_anchor", frozenset({"r1", "r2_no_anchor"})),
    ("r1_r2_source_anchor", frozenset({"r1", "r2_source_anchor"})),
    ("full_no_anchor", frozenset({"q0", "q1", "r0", "r1", "r2_no_anchor"})),
    ("full_challenger", frozenset({"q0", "q1", "r0", "r1", "r2_source_anchor"})),
)


def _component_variant_id(components: frozenset[str]) -> str:
    """Return the deterministic fallback identifier for one component set."""

    ordered = [
        component for component in (*BASE_COMPONENTS, *R2_COMPONENTS) if component in components
    ]
    return "_".join(ordered)


def _build_variant_registry() -> dict[str, frozenset[str]]:
    """Build all valid subsets while preserving established identifiers first."""

    variants: dict[str, frozenset[str]] = dict(_DESCRIPTIVE_VARIANTS)
    registered_sets = set(variants.values())
    for component_count in range(len(BASE_COMPONENTS) + 1):
        for selected in combinations(BASE_COMPONENTS, component_count):
            base = frozenset(selected)
            for r2_component in (None, *R2_COMPONENTS):
                components = base | ({r2_component} if r2_component is not None else set())
                if not components or components in registered_sets:
                    continue
                variant_id = _component_variant_id(components)
                if variant_id in variants:
                    raise RuntimeError(f"Generated challenger variant id collision: {variant_id}")
                variants[variant_id] = components
                registered_sets.add(components)
    return variants


VARIANT_COMPONENTS: Mapping[str, frozenset[str]] = _build_variant_registry()


def variant_id_for_components(components: Iterable[str]) -> str:
    """Resolve the unique registered variant for an exact valid component set."""

    normalized = frozenset(str(component).strip().lower() for component in components)
    unknown = sorted(normalized.difference(KNOWN_COMPONENTS))
    if unknown:
        raise ValueError(f"Unknown challenger components: {', '.join(unknown)}")
    if set(R2_COMPONENTS).issubset(normalized):
        raise ValueError("r2_no_anchor and r2_source_anchor are mutually exclusive")
    for variant_id, registered in VARIANT_COMPONENTS.items():
        if registered == normalized:
            return variant_id
    raise ValueError(f"No challenger variant is registered for components: {sorted(normalized)}")


def known_variant_ids() -> tuple[str, ...]:
    """Return stable model-variant identifiers for config and manifests."""

    return tuple(VARIANT_COMPONENTS)


def resolve_model_variant(config: Any) -> str:
    """Resolve and validate the configured model variant."""

    getter = getattr(config, "get", None)
    raw_variant = (
        getter("baseline_predictor.model_variant", CHAMPION_VARIANT)
        if callable(getter)
        else CHAMPION_VARIANT
    )
    variant = str(raw_variant or CHAMPION_VARIANT).strip().lower()
    if variant not in VARIANT_COMPONENTS:
        raise ValueError(
            f"Unknown baseline_predictor.model_variant {variant!r}; "
            f"expected one of {', '.join(known_variant_ids())}"
        )
    return variant


def component_enabled(config: Any, component: str) -> bool:
    """Return whether one challenger component is active for this run."""

    normalized_component = str(component).strip().lower()
    return normalized_component in VARIANT_COMPONENTS[resolve_model_variant(config)]

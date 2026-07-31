from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.analysis.challenger_governance import (
    race_candidate_kind_for_components,
    stable_json_sha256,
    validate_challenger_manifest,
)
from src.analysis.challenger_release import (
    _QUALIFYING_GATE_CHECKS,
    _RACE_GATE_BASE_CHECKS,
    _RACE_GATE_KIND_CHECKS,
    build_gate_result_envelope,
    build_weekend_actuals_attachment,
    classify_shadow_registration,
    evaluate_release_readiness,
    freeze_forecast_pair,
    validate_weekend_actuals_attachment,
)
from src.persistence.research_sidecar import ResearchSidecarStore
from tests.challenger_test_helpers import strict_manifest, strict_replay_provenance


def _grid(order: list[str]) -> list[dict[str, object]]:
    return [
        {"driver": driver, "team": f"Team {driver}", "position": position}
        for position, driver in enumerate(order, start=1)
    ]


def _forecast_reference(
    tmp_path: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    return freeze_forecast_pair(
        store=store,
        manifest=manifest,
        year=2026,
        event_name="Example Grand Prix",
        champion_prediction={"model_variant": "champion", "grid": _grid(["A", "B", "C"])},
        challenger_prediction={
            "model_variant": manifest["variant_id"],
            "grid": _grid(["B", "A", "C"]),
        },
        frozen_at="2026-07-18T11:30:00Z",
    )


def _attachment(
    tmp_path: Path,
    manifest: dict[str, Any],
    reference: dict[str, str] | None = None,
) -> dict[str, Any]:
    return build_weekend_actuals_attachment(
        manifest=manifest,
        frozen_forecast_reference=reference or _forecast_reference(tmp_path, manifest),
        year=2026,
        event_name="Example Grand Prix",
        qualifying_actual=_grid(["A", "B", "C"]),
        race_actual=_grid(["B", "A", "C"]),
        attached_at="2026-07-19T16:00:00Z",
    )


def _gate_envelope(manifest: dict[str, Any], *, passed: bool = True) -> dict[str, Any]:
    validated = validate_challenger_manifest(manifest)
    kind = race_candidate_kind_for_components(validated.components)
    checks = (
        _QUALIFYING_GATE_CHECKS
        if validated.components.issubset({"q0", "q1"})
        else _RACE_GATE_BASE_CHECKS | _RACE_GATE_KIND_CHECKS[kind]
    )
    replay = strict_replay_provenance()
    gate = {
        "target": (
            "main_qualifying" if validated.components.issubset({"q0", "q1"}) else "grand_prix_race"
        ),
        "candidate_kind": kind,
        "passed": passed,
        "checks": {name: passed for name in checks},
        "reasons": [] if passed else ["failed"],
        "thresholds": {},
        "variant_id": validated.variant_id,
        "manifest_sha256": validated.manifest_sha256,
        "replay_provenance": {
            "seeds": list(replay.seeds),
            "simulation_counts": dict(replay.simulation_counts),
            "dry_only": replay.dry_only,
            "checkpoint_event_counts": dict(replay.checkpoint_event_counts),
            "replay_sha256": replay.replay_sha256,
        },
        "event_set_sha256": stable_json_sha256([f"event-{index:02d}" for index in range(30)]),
    }
    return build_gate_result_envelope(manifest=manifest, gate_result=gate)


def _audits() -> dict[str, str]:
    return {
        name: "passed"
        for name in ("evaluation", "candidate", "shadow", "movement", "promotion", "leakage")
    }


def test_shadow_registration_requires_manifest_cutoff_and_frozen_forecasts(
    tmp_path: Path,
) -> None:
    manifest = strict_manifest()
    reference = _forecast_reference(tmp_path, manifest)

    preregistered = classify_shadow_registration(
        manifest,
        qualifying_start_at="2026-07-18T14:00:00Z",
        frozen_forecast_reference=reference,
    )
    no_forecasts = classify_shadow_registration(
        manifest,
        qualifying_start_at="2026-07-18T14:00:00Z",
    )
    late = classify_shadow_registration(
        manifest,
        qualifying_start_at="2026-07-18T11:15:00Z",
        frozen_forecast_reference=reference,
    )

    assert preregistered["classification"] == "preregistered_shadow"
    assert no_forecasts["classification"] == "retrospective_diagnostic"
    assert late["classification"] == "retrospective_diagnostic"


def test_actuals_attachment_recomputes_forecast_event_and_attachment_digests(
    tmp_path: Path,
) -> None:
    manifest = strict_manifest()
    attachment = _attachment(tmp_path, manifest)

    assert set(attachment["forecast_sha256"]) == {"champion", "challenger"}
    assert len(str(attachment["attachment_sha256"])) == 64
    tampered = dict(attachment)
    tampered["forecast_sha256"] = {"champion": "a" * 64, "challenger": "b" * 64}
    tampered["attachment_sha256"] = stable_json_sha256(
        {key: value for key, value in tampered.items() if key != "attachment_sha256"}
    )

    with pytest.raises(ValueError, match="forecast digests"):
        validate_weekend_actuals_attachment(tampered, manifest=manifest)


def test_release_readiness_passes_only_for_exact_manifest_component_gates(
    tmp_path: Path,
) -> None:
    release_manifest = strict_manifest("q0_q1_qualifying")
    components = {
        "q0": _gate_envelope(strict_manifest("q0_driver_state")),
        "q1": _gate_envelope(strict_manifest("q1_qualifying_practice")),
    }
    combination = _gate_envelope(release_manifest)
    actuals = _attachment(tmp_path, release_manifest)

    ready = evaluate_release_readiness(
        release_at="2026-07-20T09:00:00Z",
        component_gate_results=components,
        combination_gate_result=combination,
        post_race_audits=_audits(),
        actuals_attachment=actuals,
        rollback_variant="champion",
        champion_shadow_weekends=3,
        manifest=release_manifest,
    )
    arbitrary = evaluate_release_readiness(
        release_at="2026-07-20T09:00:00Z",
        component_gate_results={"made_up": components["q0"]},
        combination_gate_result=combination,
        post_race_audits=_audits(),
        actuals_attachment=actuals,
        rollback_variant="champion",
        champion_shadow_weekends=3,
        manifest=release_manifest,
    )
    mismatched_components = dict(components)
    mismatched_q1 = dict(components["q1"])
    mismatched_q1["gate"] = {
        **components["q1"]["gate"],
        "event_set_sha256": "f" * 64,
    }
    mismatched_q1["gate_envelope_sha256"] = stable_json_sha256(
        {key: value for key, value in mismatched_q1.items() if key != "gate_envelope_sha256"}
    )
    mismatched_components["q1"] = mismatched_q1
    mismatched_events = evaluate_release_readiness(
        release_at="2026-07-20T09:00:00Z",
        component_gate_results=mismatched_components,
        combination_gate_result=combination,
        post_race_audits=_audits(),
        actuals_attachment=actuals,
        rollback_variant="champion",
        champion_shadow_weekends=3,
        manifest=release_manifest,
    )

    assert ready["release_allowed"] is True
    assert ready["required_components"] == ["q0", "q1"]
    assert arbitrary["release_allowed"] is False
    assert arbitrary["checks"]["independent_component_gates_passed"] is False
    assert mismatched_events["release_allowed"] is False
    assert mismatched_events["checks"]["gate_event_sets_match"] is False


def test_release_readiness_fails_closed_on_tampered_gate_actuals_or_missing_manifest(
    tmp_path: Path,
) -> None:
    manifest = strict_manifest()
    component_gate = _gate_envelope(manifest)
    tampered_gate = dict(component_gate)
    tampered_gate["gate"] = {**component_gate["gate"], "candidate_kind": "invalid_kind"}
    tampered_gate["gate_envelope_sha256"] = stable_json_sha256(
        {key: value for key, value in tampered_gate.items() if key != "gate_envelope_sha256"}
    )
    actuals = _attachment(tmp_path, manifest)
    fabricated_actuals = dict(actuals)
    fabricated_actuals["actuals"] = {
        "qualifying": _grid(["C", "B", "A"]),
        "race": _grid(["B", "A", "C"]),
    }

    result = evaluate_release_readiness(
        release_at="2026-07-20T09:00:00Z",
        component_gate_results={"q1": tampered_gate},
        combination_gate_result=tampered_gate,
        post_race_audits=_audits(),
        actuals_attachment=fabricated_actuals,
        rollback_variant="champion",
        champion_shadow_weekends=3,
        manifest=manifest,
    )
    missing_manifest = evaluate_release_readiness(
        release_at="2026-07-20T09:00:00Z",
        component_gate_results={},
        combination_gate_result=None,
        post_race_audits=_audits(),
        actuals_attachment=actuals,
        rollback_variant="champion",
        champion_shadow_weekends=3,
    )

    assert result["release_allowed"] is False
    assert result["checks"]["independent_component_gates_passed"] is False
    assert result["checks"]["actuals_attached"] is False
    assert missing_manifest["checks"]["manifest_valid"] is False


def test_release_readiness_still_requires_weekday_audits_and_shadow_window(
    tmp_path: Path,
) -> None:
    manifest = strict_manifest()
    gate = _gate_envelope(manifest)
    result = evaluate_release_readiness(
        release_at="2026-07-18T09:00:00Z",
        component_gate_results={"q1": gate},
        combination_gate_result=gate,
        post_race_audits={"evaluation": "passed"},
        actuals_attachment=_attachment(tmp_path, manifest),
        rollback_variant="champion",
        champion_shadow_weekends=2,
        manifest=manifest,
    )

    assert result["release_allowed"] is False
    assert result["checks"]["weekday_release"] is False
    assert result["checks"]["post_race_audits_passed"] is False
    assert result["checks"]["champion_shadow_window_planned"] is False


def test_release_readiness_rejects_a_research_gate_relaxation_manifest(tmp_path: Path) -> None:
    """A manifest built under a research-only threshold relaxation (Q1/R2 walk-forward
    minimum-training-event floors lowered for the short 2026 season) must never be
    promotable, even when every other release check would otherwise pass."""
    release_manifest = strict_manifest("q0_q1_qualifying")
    release_manifest["metadata"] = {
        "research_gate_relaxation": {
            "component": "q1",
            "original_threshold": 30,
            "relaxed_threshold": 4,
            "training_events_used": 4,
        }
    }
    release_manifest["manifest_sha256"] = stable_json_sha256(
        {key: value for key, value in release_manifest.items() if key != "manifest_sha256"}
    )
    components = {
        "q0": _gate_envelope(strict_manifest("q0_driver_state")),
        "q1": _gate_envelope(strict_manifest("q1_qualifying_practice")),
    }
    combination = _gate_envelope(release_manifest)
    actuals = _attachment(tmp_path, release_manifest)

    result = evaluate_release_readiness(
        release_at="2026-07-20T09:00:00Z",
        component_gate_results=components,
        combination_gate_result=combination,
        post_race_audits=_audits(),
        actuals_attachment=actuals,
        rollback_variant="champion",
        champion_shadow_weekends=3,
        manifest=release_manifest,
    )

    assert result["release_allowed"] is False
    assert result["checks"]["no_research_gate_relaxation"] is False
    assert "research_gate_relaxation" in " ".join(result["reasons"])

    # Confirm the check is the only thing standing between this manifest and release:
    # without the marker the identical fixture is release-ready.
    clean_manifest = strict_manifest("q0_q1_qualifying")
    clean_result = evaluate_release_readiness(
        release_at="2026-07-20T09:00:00Z",
        component_gate_results=components,
        combination_gate_result=_gate_envelope(clean_manifest),
        post_race_audits=_audits(),
        actuals_attachment=_attachment(tmp_path, clean_manifest),
        rollback_variant="champion",
        champion_shadow_weekends=3,
        manifest=clean_manifest,
    )
    assert clean_result["checks"]["no_research_gate_relaxation"] is True


def _retrospective_manifest() -> dict[str, Any]:
    manifest = strict_manifest("q1_qualifying_practice")
    manifest["metadata"] = {"retrospective_diagnostic": True}
    manifest["manifest_sha256"] = stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def test_release_readiness_rejects_a_retrospective_diagnostic_manifest() -> None:
    """The Q1 runtime's retrospective_diagnostic marker (a bundle built today resolved
    against an already-past historical cutoff) must never be promotable either,
    independent of -- and even without -- a research_gate_relaxation marker."""
    # Forecast freezing itself already refuses this manifest (see
    # test_freeze_forecast_pair_refuses_a_retrospective_diagnostic_manifest), so
    # there is no real actuals_attachment to build for it -- pass None and confirm
    # release_readiness's own check catches the marker independently either way.
    release_manifest = _retrospective_manifest()
    gate = _gate_envelope(release_manifest)

    result = evaluate_release_readiness(
        release_at="2026-07-20T09:00:00Z",
        component_gate_results={"q1": gate},
        combination_gate_result=gate,
        post_race_audits=_audits(),
        actuals_attachment=None,
        rollback_variant="champion",
        champion_shadow_weekends=3,
        manifest=release_manifest,
    )

    assert result["release_allowed"] is False
    assert result["checks"]["no_retrospective_diagnostic"] is False
    assert "retrospective_diagnostic" in " ".join(result["reasons"])


def test_freeze_forecast_pair_refuses_a_retrospective_diagnostic_manifest(tmp_path: Path) -> None:
    manifest = _retrospective_manifest()
    with pytest.raises(ValueError, match="retrospective_diagnostic"):
        _forecast_reference(tmp_path, manifest)


def test_classify_shadow_registration_refuses_a_retrospective_diagnostic_manifest() -> None:
    manifest = _retrospective_manifest()
    with pytest.raises(ValueError, match="retrospective_diagnostic"):
        classify_shadow_registration(manifest, qualifying_start_at="2026-07-18T14:00:00Z")

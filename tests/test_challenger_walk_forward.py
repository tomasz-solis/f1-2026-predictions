"""Tests for the leakage-safe challenger walk-forward coordinator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts import run_challenger_walk_forward as replay_cli

from src.analysis import challenger_walk_forward as replay
from src.analysis.challenger_governance import ValidatedChallengerManifest
from src.persistence.research_sidecar import ResearchSidecarStore
from tests.challenger_test_helpers import strict_manifest


def _validated_manifest(*components: str) -> ValidatedChallengerManifest:
    return ValidatedChallengerManifest(
        candidate_id="candidate",
        variant_id="q1_qualifying_practice" if "q1" in components else "q0_driver_state",
        components=frozenset(components),
        manifest_sha256="a" * 64,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        cutoff_at=datetime(2026, 1, 1, tzinfo=UTC),
        simulation_counts={"qualifying": 100, "race": 80},
    )


def _manifest() -> dict[str, Any]:
    return {"provenance": {"input_snapshot_ids": ["snapshot"]}}


def _grid(order: tuple[str, ...], *, interval: bool = True) -> list[dict[str, Any]]:
    teams = {"A": "T1", "B": "T1", "C": "T2", "D": "T2"}
    rows: list[dict[str, Any]] = []
    for position, driver in enumerate(order, start=1):
        row: dict[str, Any] = {
            "driver": driver,
            "team": teams[driver],
            "position": position,
        }
        if interval:
            row.update({"p5": max(1, position - 1), "p95": min(4, position + 1)})
        rows.append(row)
    return rows


def _event(index: int, *, dry: bool = True) -> dict[str, Any]:
    event_start = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=index * 7)
    actual_grid = _grid(("A", "B", "C", "D"), interval=False)
    actual_race = [dict(row, dnf=False) for row in actual_grid]
    return {
        "event_id": f"event-{index:02d}",
        "event_start_at": event_start.isoformat(),
        "qualifying_start_at": (event_start + timedelta(days=1)).isoformat(),
        "session_kind": "main",
        "is_dry": dry,
        "checkpoint_payloads": {"PRE": {"available_at": event_start.isoformat()}},
        "actual_qualifying_grid": actual_grid,
        "actual_race_finish_order": actual_race,
        "input_snapshot_ids": ["snapshot"],
    }


def _strict_q0_manifest(*, created_before_events: bool = False) -> dict[str, Any]:
    manifest = strict_manifest("q0_driver_state", candidate_id="q0-walk-forward")
    manifest["provenance"]["input_snapshot_ids"] = ["snapshot"]
    if created_before_events:
        manifest["created_at"] = "2024-12-20T11:00:00Z"
        manifest["cutoff_at"] = "2024-12-20T10:00:00Z"
    manifest["manifest_sha256"] = replay.stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def _freeze_event(
    raw_event: dict[str, Any],
    *,
    manifest: dict[str, Any],
    store: ResearchSidecarStore,
    frozen_at: str = "2026-07-18T12:00:00Z",
) -> dict[str, Any]:
    source_backend = _Backend()
    event_stub = SimpleNamespace(event_id=raw_event["event_id"])
    qualifying_predictions: dict[str, dict[str, Any]] = {
        "champion": {},
        "challenger": {},
    }
    race_predictions: dict[str, dict[str, Any]] = {
        "champion": {},
        "challenger": {},
    }
    for role in ("champion", "challenger"):
        for seed in replay.DEFAULT_REPLAY_SEEDS:
            qualifying_predictions[role][str(seed)] = source_backend.predict_qualifying(
                role=role,
                seed=seed,
                event=event_stub,
                checkpoint="PRE",
                fold_artifacts=None,
            )
            race_predictions[role][str(seed)] = source_backend.predict_race_views(
                role=role,
                seed=seed,
                event=event_stub,
                checkpoint="PRE",
                fold_artifacts=None,
            )
    checkpoint = raw_event["checkpoint_payloads"]["PRE"]
    information_cutoff = checkpoint["available_at"]
    reference = replay.freeze_checkpoint_forecast_bundle(
        store=store,
        manifest=manifest,
        event_id=raw_event["event_id"],
        event_start_at=raw_event["event_start_at"],
        session_kind=raw_event["session_kind"],
        checkpoint="PRE",
        information_cutoff_at=information_cutoff,
        qualifying_start_at=raw_event["qualifying_start_at"],
        frozen_at=frozen_at,
        qualifying_predictions=qualifying_predictions,
        race_view_predictions=race_predictions,
    )
    return {
        **raw_event,
        "checkpoint_payloads": {
            "PRE": {
                "information_cutoff_at": information_cutoff,
                "forecast_reference": reference,
            }
        },
    }


class _Backend:
    def __init__(self, *, future_fold: bool = False) -> None:
        self.future_fold = future_fold
        self.fit_calls: list[dict[str, Any]] = []
        self.qualifying_calls: list[tuple[str, int, str, str]] = []
        self.race_calls: list[tuple[str, int, str, str]] = []

    def fit_fold(self, **kwargs: Any) -> dict[str, Any]:
        training = kwargs["training_events"]
        calibration = kwargs["calibration_events"]
        target = kwargs["target_event"]
        checkpoint = kwargs["checkpoint"]
        self.fit_calls.append(dict(kwargs))
        max_input = max(event.event_start_at for event in (*training, *calibration))
        if self.future_fold:
            max_input = target.event_start_at
        return {
            "variant_id": "q1_qualifying_practice",
            "checkpoint": checkpoint,
            "session_kind": target.session_kind,
            "target_event_id": target.event_id,
            "training_event_ids": [event.event_id for event in training],
            "calibration_event_ids": [event.event_id for event in calibration],
            "cutoff_at": target.event_start_at.isoformat(),
            "max_input_timestamp": max_input.isoformat(),
        }

    def predict_qualifying(self, **kwargs: Any) -> dict[str, Any]:
        role = kwargs["role"]
        seed = kwargs["seed"]
        event = kwargs["event"]
        checkpoint = kwargs["checkpoint"]
        self.qualifying_calls.append((role, seed, event.event_id, checkpoint))
        variant = "champion" if role == "champion" else "q0_driver_state"
        if kwargs["fold_artifacts"] is not None:
            variant = "q1_qualifying_practice"
        order = ("B", "A", "D", "C") if role == "champion" else ("A", "B", "C", "D")
        probability = 0.55 if role == "champion" else 0.75
        return {
            "model_variant": variant,
            "grid": _grid(order),
            "teammate_head_to_head": [
                {
                    "team": "T1",
                    "driver_a": "A",
                    "driver_b": "B",
                    "p_driver_a_ahead": probability,
                    "p_driver_b_ahead": 1.0 - probability,
                },
                {
                    "team": "T2",
                    "driver_a": "C",
                    "driver_b": "D",
                    "p_driver_a_ahead": probability,
                    "p_driver_b_ahead": 1.0 - probability,
                },
            ],
        }

    def predict_race_views(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        role = kwargs["role"]
        seed = kwargs["seed"]
        event = kwargs["event"]
        checkpoint = kwargs["checkpoint"]
        self.race_calls.append((role, seed, event.event_id, checkpoint))
        variant = "champion" if role == "champion" else "q0_driver_state"
        if kwargs["fold_artifacts"] is not None:
            variant = "q1_qualifying_practice"
        order = ("B", "A", "C", "D") if role == "champion" else ("A", "B", "C", "D")

        def prediction(detail: str) -> dict[str, Any]:
            return {
                "model_variant": variant,
                "grid_source_detail": detail,
                "finish_order": [
                    dict(row, dnf_probability=0.1) for row in _grid(order, interval=False)
                ],
            }

        return {
            "conditional_actual_grid": prediction("actual_starting_grid"),
            "end_to_end_predicted_grid": prediction("predicted_marginal_fallback"),
        }


def test_walk_forward_filters_wet_events_and_uses_common_seeds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        replay,
        "validate_challenger_manifest",
        lambda _manifest: _validated_manifest("q0"),
    )
    backend = _Backend()
    report = replay.run_challenger_walk_forward(
        events=[_event(2), _event(0), _event(1, dry=False)],
        manifest=_manifest(),
        backend=backend,
    )

    assert [row["event_id"] for row in report["scored_events"]] == ["event-00", "event-02"]
    assert report["skipped_events"] == [{"event_id": "event-01", "reason": "dry_only_candidate"}]
    assert report["checkpoint_event_counts"] == {"PRE": 2, "FP1": 0, "FP2": 0, "FP3": 0}
    assert {call[1] for call in backend.qualifying_calls} == {17, 42, 91}
    assert len(backend.qualifying_calls) == 12
    assert len(backend.race_calls) == 12
    assert report["leakage_audit"]["passed"] is True
    assert len(report["replay_sha256"]) == 64


def test_walk_forward_records_per_checkpoint_refusal_and_keeps_scoring_others(
    monkeypatch: pytest.MonkeyPatch,
):
    """A backend that fails closed on one event-checkpoint's own inputs (e.g. a
    real FastF1 session too thin to extract) must not void the whole variant run:
    every other eligible event-checkpoint still gets scored, and the refusal is
    recorded loudly (event_id, checkpoint, reason, error_type) instead of being
    silently dropped or raised as an unhandled ValueError."""
    monkeypatch.setattr(
        replay,
        "validate_challenger_manifest",
        lambda _manifest: _validated_manifest("q0"),
    )

    class _PoisonedBackend(_Backend):
        def predict_qualifying(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["event"].event_id == "event-01":
                raise replay.CheckpointInputUnavailable(
                    "event-01 PRE: required session 'Practice 1' could not be "
                    "extracted: teams=1 mapped=0 selected_laps=0"
                )
            return super().predict_qualifying(**kwargs)

    backend = _PoisonedBackend()
    report = replay.run_challenger_walk_forward(
        events=[_event(0), _event(1)],
        manifest=_manifest(),
        backend=backend,
    )

    assert [row["event_id"] for row in report["scored_events"]] == ["event-00"]
    assert report["checkpoint_refusals"] == [
        {
            "event_id": "event-01",
            "checkpoint": "PRE",
            "reason": (
                "event-01 PRE: required session 'Practice 1' could not be "
                "extracted: teams=1 mapped=0 selected_laps=0"
            ),
            "error_type": "CheckpointInputUnavailable",
        }
    ]
    assert report["checkpoint_event_counts"] == {"PRE": 1, "FP1": 0, "FP2": 0, "FP3": 0}
    assert report["leakage_audit"]["passed"] is True


def test_research_gate_relaxation_requires_matching_manifest_disclosure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        replay,
        "validate_challenger_manifest",
        lambda _manifest: _validated_manifest("q1"),
    )
    with pytest.raises(ValueError, match="research_gate_relaxation"):
        replay.run_challenger_walk_forward(
            events=[_event(index) for index in range(6)],
            manifest=_manifest(),  # no metadata.research_gate_relaxation disclosure
            backend=_Backend(),
            research_gate_relaxation={"q1": 4},
        )


def test_research_gate_relaxation_makes_q1_scoreable_from_round_five(
    monkeypatch: pytest.MonkeyPatch,
):
    """Floor-clamped research relaxation (4, not the production 30) makes Q1
    scoreable once 4 disjoint prior dry events exist, with the exact relaxation
    detail recorded on every checkpoint it applies to."""
    monkeypatch.setattr(
        replay,
        "validate_challenger_manifest",
        lambda _manifest: _validated_manifest("q1"),
    )
    manifest = {
        **_manifest(),
        "metadata": {"research_gate_relaxation": {"q1": 4}},
    }
    backend = _Backend()
    report = replay.run_challenger_walk_forward(
        events=[_event(index) for index in range(6)],
        manifest=manifest,
        backend=backend,
        research_gate_relaxation={"q1": 4},
    )

    # events 0-3 are the 4 required training events; event-04 is held out as the
    # calibration event (_split_prior_events reserves ~1/5); event-05 is scoreable.
    assert [row["event_id"] for row in report["scored_events"]] == ["event-05"]
    relaxation = report["scored_events"][0]["checkpoints"]["PRE"]["research_gate_relaxation"]
    assert relaxation == {
        "component": "q1",
        "original_threshold": 30,
        "relaxed_threshold": 4,
        "training_events_used": 4,
        "shrinkage_applied": pytest.approx(4 / 30),
    }

    # A floor below the safety minimum (4) must still clamp up, not go lower.
    report_floor = replay.run_challenger_walk_forward(
        events=[_event(index) for index in range(6)],
        manifest={**_manifest(), "metadata": {"research_gate_relaxation": {"q1": 1}}},
        backend=_Backend(),
        research_gate_relaxation={"q1": 1},
    )
    floor_relaxation = report_floor["scored_events"][0]["checkpoints"]["PRE"][
        "research_gate_relaxation"
    ]
    assert floor_relaxation["relaxed_threshold"] == 4


def test_q1_walk_forward_uses_only_earlier_disjoint_events(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        replay,
        "validate_challenger_manifest",
        lambda _manifest: _validated_manifest("q1"),
    )
    backend = _Backend()
    report = replay.run_challenger_walk_forward(
        events=[_event(index) for index in range(32)],
        manifest=_manifest(),
        backend=backend,
    )

    assert len(report["scored_events"]) == 1
    assert report["scored_events"][0]["event_id"] == "event-31"
    assert len(report["skipped_events"]) == 31
    assert len(backend.fit_calls) == 1
    fit = backend.fit_calls[0]
    assert [event.event_id for event in fit["training_events"]] == [
        f"event-{index:02d}" for index in range(30)
    ]
    assert [event.event_id for event in fit["calibration_events"]] == ["event-30"]
    assert fit["target_event"].event_id == "event-31"
    assert report["leakage_audit"]["passed"] is True


def test_q1_walk_forward_rejects_fold_inputs_at_target_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        replay,
        "validate_challenger_manifest",
        lambda _manifest: _validated_manifest("q1"),
    )
    with pytest.raises(ValueError, match="information boundary"):
        replay.run_challenger_walk_forward(
            events=[_event(index) for index in range(32)],
            manifest=_manifest(),
            backend=_Backend(future_fold=True),
        )


def test_walk_forward_rejects_nonstandard_seeds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        replay,
        "validate_challenger_manifest",
        lambda _manifest: _validated_manifest("q0"),
    )
    with pytest.raises(ValueError, match="exactly"):
        replay.run_challenger_walk_forward(
            events=[_event(0)],
            manifest=_manifest(),
            backend=_Backend(),
            seeds=(17, 42),
        )


def test_walk_forward_builds_gate_envelopes_from_same_event_set(
    tmp_path: Path,
):
    manifest = _strict_q0_manifest()
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    report = replay.run_challenger_walk_forward(
        events=[
            _freeze_event(_event(0), manifest=manifest, store=store),
            _freeze_event(_event(1), manifest=manifest, store=store),
        ],
        manifest=manifest,
        backend=replay.FrozenPredictionBundleBackend(),
    )

    qualifying = replay.build_qualifying_gate_metrics_from_walk_forward(
        report,
        manifest=manifest,
        target="main_qualifying",
    )
    race = replay.build_race_gate_metrics_from_walk_forward(
        report,
        manifest=manifest,
        target="grand_prix_race",
    )

    assert qualifying.grid_mae.event_ids == ("event-00", "event-01")
    assert qualifying.race_views.conditional_actual_grid.event_ids == (
        "event-00",
        "event-01",
    )
    assert race.race_views.end_to_end_predicted_grid.event_ids == (
        "event-00",
        "event-01",
    )
    assert qualifying.replay_provenance is not None
    assert qualifying.replay_provenance.seeds == (17, 42, 91)


def test_frozen_bundle_backend_selects_exact_role_seed_and_views(tmp_path: Path):
    manifest = _strict_q0_manifest()
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    raw_event = _freeze_event(_event(0), manifest=manifest, store=store)
    event = replay._normalise_event(raw_event)
    backend = replay.FrozenPredictionBundleBackend()
    checkpoint_payload = event.checkpoint_payloads["PRE"]

    qualifying = backend.predict_qualifying(
        event=event,
        checkpoint="PRE",
        checkpoint_payload=checkpoint_payload,
        manifest=manifest,
        role="champion",
        seed=17,
    )
    race_views = backend.predict_race_views(
        event=event,
        checkpoint="PRE",
        checkpoint_payload=checkpoint_payload,
        manifest=manifest,
        role="challenger",
        seed=42,
    )

    assert qualifying["model_variant"] == "champion"
    assert set(race_views) == {
        "conditional_actual_grid",
        "end_to_end_predicted_grid",
    }


def test_checkpoint_catalog_rejects_embedded_forecasts() -> None:
    event = _event(0)
    event["checkpoint_payloads"]["PRE"]["qualifying_predictions"] = {}

    with pytest.raises(ValueError, match="use forecast_reference"):
        replay._normalise_event(event)


def test_late_frozen_historical_bundle_is_retrospective_not_rejected(tmp_path: Path) -> None:
    manifest = _strict_q0_manifest()
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    event = _freeze_event(_event(0), manifest=manifest, store=store)

    report = replay.run_challenger_walk_forward(
        events=[event],
        manifest=manifest,
        backend=replay.FrozenPredictionBundleBackend(),
    )

    registration = report["scored_events"][0]["checkpoints"]["PRE"]["forecast_registration"]
    assert registration["classification"] == "retrospective_diagnostic"
    assert report["forecast_registration_counts"]["retrospective_diagnostic"] == 1
    metrics = replay.build_qualifying_gate_metrics_from_walk_forward(
        report,
        manifest=manifest,
        target="main_qualifying",
    )
    assert metrics.replay_provenance is not None
    assert metrics.replay_provenance.replay_sha256 == report["replay_sha256"]


def test_pre_qualifying_frozen_bundle_is_preregistered(tmp_path: Path) -> None:
    manifest = _strict_q0_manifest(created_before_events=True)
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    event = _event(0)
    frozen_at = (datetime.fromisoformat(event["event_start_at"]) + timedelta(hours=1)).isoformat()
    frozen_event = _freeze_event(
        event,
        manifest=manifest,
        store=store,
        frozen_at=frozen_at,
    )

    report = replay.run_challenger_walk_forward(
        events=[frozen_event],
        manifest=manifest,
        backend=replay.FrozenPredictionBundleBackend(),
    )

    registration = report["scored_events"][0]["checkpoints"]["PRE"]["forecast_registration"]
    assert registration["classification"] == "preregistered_shadow"


def test_frozen_checkpoint_reference_rejects_tamper_and_cross_manifest(
    tmp_path: Path,
) -> None:
    manifest = _strict_q0_manifest()
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    event = _freeze_event(_event(0), manifest=manifest, store=store)
    checkpoint = event["checkpoint_payloads"]["PRE"]
    reference = checkpoint["forecast_reference"]
    other_manifest = _strict_q0_manifest()
    other_manifest["candidate_id"] = "another-q0-candidate"
    other_manifest["manifest_sha256"] = replay.stable_json_sha256(
        {key: value for key, value in other_manifest.items() if key != "manifest_sha256"}
    )

    with pytest.raises(ValueError, match="another manifest"):
        replay.validate_frozen_checkpoint_forecast_reference(
            reference,
            manifest=other_manifest,
            event_id=event["event_id"],
            event_start_at=event["event_start_at"],
            session_kind=event["session_kind"],
            checkpoint="PRE",
            information_cutoff_at=checkpoint["information_cutoff_at"],
            qualifying_start_at=event["qualifying_start_at"],
        )

    path = Path(reference["path"])
    envelope = replay.json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["frozen_at"] = "1999-01-01T00:00:00Z"
    path.write_text(replay.json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact digest"):
        replay.validate_frozen_checkpoint_forecast_reference(
            reference,
            manifest=manifest,
            event_id=event["event_id"],
            event_start_at=event["event_start_at"],
            session_kind=event["session_kind"],
            checkpoint="PRE",
            information_cutoff_at=checkpoint["information_cutoff_at"],
            qualifying_start_at=event["qualifying_start_at"],
        )


def test_gate_conversion_rejects_replay_tamper_and_cross_manifest(tmp_path: Path) -> None:
    manifest = _strict_q0_manifest()
    store = ResearchSidecarStore(tmp_path / "research", repo_root=tmp_path)
    report = replay.run_challenger_walk_forward(
        events=[_freeze_event(_event(0), manifest=manifest, store=store)],
        manifest=manifest,
        backend=replay.FrozenPredictionBundleBackend(),
    )
    tampered = replay.json.loads(replay.json.dumps(report))
    tampered["scored_events"][0]["checkpoints"]["PRE"]["champion"]["grid_mae"] = 0.0
    with pytest.raises(ValueError, match="replay_sha256"):
        replay.build_qualifying_gate_metrics_from_walk_forward(
            tampered,
            manifest=manifest,
            target="main_qualifying",
        )

    count_tamper = replay.json.loads(replay.json.dumps(report))
    count_tamper["checkpoint_event_counts"]["PRE"] = 99
    count_tamper["replay_sha256"] = replay.stable_json_sha256(
        {key: value for key, value in count_tamper.items() if key != "replay_sha256"}
    )
    with pytest.raises(ValueError, match="checkpoint counts"):
        replay.build_qualifying_gate_metrics_from_walk_forward(
            count_tamper,
            manifest=manifest,
            target="main_qualifying",
        )

    other_manifest = _strict_q0_manifest()
    other_manifest["candidate_id"] = "cross-manifest"
    other_manifest["manifest_sha256"] = replay.stable_json_sha256(
        {key: value for key, value in other_manifest.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="different challenger manifest"):
        replay.build_qualifying_gate_metrics_from_walk_forward(
            report,
            manifest=other_manifest,
            target="main_qualifying",
        )


def test_race_metrics_use_canonical_finisher_and_status_dnf_semantics() -> None:
    prediction = {
        "finish_order": [
            {"driver": "A", "team": "T1", "position": 1, "dnf_probability": 0.1},
            {"driver": "C", "team": "T3", "position": 2, "dnf_probability": 0.1},
            {"driver": "B", "team": "T2", "position": 3, "dnf_probability": 0.1},
        ]
    }
    actual = [
        {"driver": "A", "team": "T1", "position": 1, "status": "Finished"},
        {"driver": "B", "team": "T2", "position": 2, "status": "Finished"},
        {"driver": "C", "team": "T3", "position": 3, "status": "Retired"},
    ]

    metrics = replay._race_metrics(prediction, actual)

    assert metrics["finisher_mae"] == pytest.approx(0.5)
    assert metrics["dnf_brier"] == pytest.approx(0.2766666667)


def test_qualifying_validation_rejects_selective_intervals_and_h2h() -> None:
    actual = _grid(("A", "B", "C", "D"), interval=False)
    backend = _Backend()
    event_stub = SimpleNamespace(event_id="selective")
    complete = backend.predict_qualifying(
        role="challenger",
        seed=17,
        event=event_stub,
        checkpoint="PRE",
        fold_artifacts=None,
    )
    missing_interval = replay.json.loads(replay.json.dumps(complete))
    del missing_interval["grid"][0]["p5"]
    with pytest.raises(ValueError, match="p5/p95 for every driver"):
        replay._validate_qualifying_prediction(
            missing_interval,
            role="challenger",
            variant_id="q0_driver_state",
            actual_grid=actual,
        )

    missing_pair = replay.json.loads(replay.json.dumps(complete))
    missing_pair["teammate_head_to_head"] = missing_pair["teammate_head_to_head"][:1]
    with pytest.raises(ValueError, match="exactly every actual teammate pair"):
        replay._validate_qualifying_prediction(
            missing_pair,
            role="challenger",
            variant_id="q0_driver_state",
            actual_grid=actual,
        )


def test_walk_forward_cli_consumes_reference_only_catalog(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _strict_q0_manifest()
    input_store = ResearchSidecarStore(tmp_path / "frozen-inputs", repo_root=tmp_path)
    event = _freeze_event(_event(0), manifest=manifest, store=input_store)
    manifest_path = tmp_path / "manifest.json"
    catalog_path = tmp_path / "events.json"
    manifest_path.write_text(replay.json.dumps(manifest), encoding="utf-8")
    catalog_path.write_text(replay.json.dumps({"events": [event]}), encoding="utf-8")

    exit_code = replay_cli.main(
        [
            "--manifest",
            str(manifest_path),
            "--event-catalog",
            str(catalog_path),
            "--sidecar-root",
            str(tmp_path / "outputs"),
            "--repo-root",
            str(tmp_path),
        ]
    )

    output_path = Path(capsys.readouterr().out.strip())
    envelope = replay.json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert envelope["payload"]["replay"]["forecast_registration_counts"] == {
        "preregistered_shadow": 0,
        "research_backend_generated": 0,
        "retrospective_diagnostic": 1,
    }

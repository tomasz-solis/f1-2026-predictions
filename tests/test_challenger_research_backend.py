"""Focused tests for the real-data walk-forward research backend.

These tests avoid invoking the (slow) Monte Carlo predictor. Instead they prove the
backend's leakage discipline and caching contract at the level that matters: which
real sessions get replayed into which state directory, in which order, and whether
the disk-backed prediction cache is correctly keyed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

import src.analysis.challenger_research_backend as backend_module
from src.analysis.challenger_research_backend import ProductionReplayBackend
from src.analysis.challenger_walk_forward import _normalise_catalog
from src.models.challenger_variants import VARIANT_COMPONENTS
from src.models.qualifying_practice_challenger import DEFAULT_FEATURE_COLUMNS


def _grid(n: int = 4) -> list[dict[str, Any]]:
    return [{"driver": f"D{i}", "team": f"T{i % 2}", "position": i} for i in range(1, n + 1)]


def _raw_event(index: int, *, checkpoints: dict[str, list[str]]) -> dict[str, Any]:
    event_start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index * 14)
    qualifying_start = event_start + timedelta(days=1, hours=5)
    checkpoint_payloads: dict[str, Any] = {
        "PRE": {"information_cutoff_at": event_start.isoformat(), "sessions_available": []}
    }
    cursor = event_start + timedelta(hours=1)
    for checkpoint, sessions in checkpoints.items():
        checkpoint_payloads[checkpoint] = {
            "information_cutoff_at": cursor.isoformat(),
            "sessions_available": sessions,
        }
        cursor += timedelta(hours=3)
    return {
        "event_id": f"event-{index:02d}",
        "race_name": f"Test Grand Prix {index}",
        "year": 2026,
        "event_start_at": event_start.isoformat(),
        "qualifying_start_at": qualifying_start.isoformat(),
        "session_kind": "main",
        "is_dry": True,
        "checkpoint_payloads": checkpoint_payloads,
        "actual_qualifying_grid": _grid(),
        "actual_race_finish_order": [dict(row, dnf=False) for row in _grid()],
        "actual_starting_grid": [dict(row, start_type="grid") for row in _grid()],
        "input_snapshot_ids": [f"snapshot-{index}"],
        "fastf1_cache_dir": "data/raw/.fastf1_cache",
    }


def _catalog(n: int = 3) -> list[dict[str, Any]]:
    return [
        _raw_event(
            i,
            checkpoints={
                "FP1": ["Practice 1"],
                "FP2": ["Practice 1", "Practice 2"],
                "FP3": ["Practice 1", "Practice 2", "Practice 3"],
            },
        )
        for i in range(n)
    ]


def _backend(tmp_path, raw_events: list[dict[str, Any]]) -> ProductionReplayBackend:
    return ProductionReplayBackend(
        events=raw_events,
        source_processed_dir="data/processed",
        state_root=tmp_path / "state",
        prediction_cache_root=tmp_path / "cache",
    )


def test_checkpoint_state_only_replays_sessions_listed_for_that_checkpoint(
    tmp_path, monkeypatch
) -> None:
    """A fold's checkpoint state must never touch a session past its own cutoff."""
    raw_events = _catalog(1)
    backend = _backend(tmp_path, raw_events)

    recorded: list[str] = []

    def fake_apply_session_update(
        *,
        year,
        event_name,
        session_name,
        cache_dirs,
        processed_dir,
        new_weight=None,
        min_field_coverage=None,
    ):
        recorded.append(session_name)

    def fake_reset(processed_dir, *, year):
        processed_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "src.analysis.challenger_research_backend._apply_session_update",
        fake_apply_session_update,
    )
    monkeypatch.setattr(
        "src.analysis.challenger_research_backend._reset_replay_artifacts", fake_reset
    )
    monkeypatch.setattr(
        "src.analysis.challenger_research_backend.shutil.copytree",
        lambda src, dst: dst.mkdir(parents=True, exist_ok=True),
    )

    backend._checkpoint_state_for(event_id="event-00", checkpoint="FP2")

    # FP2's sessions_available is exactly ["Practice 1", "Practice 2"] -> FP1, FP2.
    # FP3/Q/R must never be replayed into a FP2 checkpoint's state: leaking either
    # would let the fold see practice/qualifying/race data from after its own cutoff.
    assert recorded == ["FP1", "FP2"]


def test_prefix_state_never_commits_the_target_event_itself(tmp_path, monkeypatch) -> None:
    """Fitting event N's season state must only ever commit events strictly before N."""
    raw_events = _catalog(3)
    backend = _backend(tmp_path, raw_events)

    committed: list[str] = []

    def fake_commit(*, processed_dir, event_id):
        committed.append(event_id)

    def fake_reset(processed_dir, *, year):
        processed_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(backend, "_commit_event", fake_commit)
    monkeypatch.setattr(
        "src.analysis.challenger_research_backend._reset_replay_artifacts", fake_reset
    )
    monkeypatch.setattr(
        "src.analysis.challenger_research_backend.shutil.copytree",
        lambda src, dst: dst.mkdir(parents=True, exist_ok=True),
    )

    # The prefix state for event-02 (index 2) must only ever have committed
    # event-00 and event-01, in chronological order, and never event-02 itself.
    backend._prefix_state_for("event-02")

    assert committed == ["event-00", "event-01"]
    assert "event-02" not in committed


def test_fit_fold_rejects_a_training_event_that_is_not_strictly_earlier(tmp_path) -> None:
    """fit_fold must fail closed if handed a training event at/after the target."""
    raw_events = _catalog(2)
    catalog = _normalise_catalog(raw_events)
    backend = _backend(tmp_path, raw_events)
    target = catalog[0]
    # A non-chronological training event (the target itself) must be rejected before
    # any state is built -- this is the leakage-rejection contract fit_fold owns.
    try:
        backend.fit_fold(
            training_events=[target],
            calibration_events=[],
            target_event=target,
            checkpoint="PRE",
            manifest={"variant_id": "r2_source_anchor"},
        )
    except ValueError as exc:
        assert "non-chronological" in str(exc)
    else:
        raise AssertionError("fit_fold accepted a training event at/after its target")


def test_prediction_cache_round_trips_and_is_isolated_by_key(tmp_path) -> None:
    backend = _backend(tmp_path, _catalog(1))
    key_a = {"kind": "qualifying", "event_id": "event-00", "seed": 17, "variant_id": "champion"}
    key_b = {"kind": "qualifying", "event_id": "event-00", "seed": 42, "variant_id": "champion"}

    assert backend._cached(key_a) is None
    backend._store(key_a, {"grid": [1, 2, 3]})
    assert backend._cached(key_a) == {"grid": [1, 2, 3]}
    # A different seed must be a cache miss: reusing seed 17's cached prediction for
    # seed 42 would silently collapse the three-seed replay contract to one seed.
    assert backend._cached(key_b) is None


def test_state_digest_is_independent_of_simulation_counts(tmp_path) -> None:
    """Raising sim counts must reuse fitted state dirs, not force a full rebuild.

    Season/checkpoint state (team/driver fitting) never depends on how many Monte
    Carlo samples a later prediction call uses. Only the prediction cache path should
    change when sim counts change.
    """
    raw_events = _catalog(1)
    low = ProductionReplayBackend(
        events=raw_events,
        state_root=tmp_path / "state",
        prediction_cache_root=tmp_path / "cache",
        qualifying_simulations=20,
        race_simulations=20,
    )
    high = ProductionReplayBackend(
        events=raw_events,
        state_root=tmp_path / "state",
        prediction_cache_root=tmp_path / "cache",
        qualifying_simulations=500,
        race_simulations=500,
    )
    assert low._prefix_dir_root() == high._prefix_dir_root()
    assert low._source_digest != high._source_digest


def test_research_cumulative_pull_cap_defaults_to_byte_identical_state_digest(tmp_path) -> None:
    """Round-9 fix-A backtest hatch: omitting the flag, or passing it explicitly as
    None, must produce the EXACT same _state_digest as before the flag existed --
    proof the existing (hours of real replay work) season-state cache stays valid
    for every ordinary run. Same no-op-by-default proof pattern as
    retrospective_diagnostic."""
    raw_events = _catalog(1)
    omitted = ProductionReplayBackend(
        events=raw_events, state_root=tmp_path / "state", prediction_cache_root=tmp_path / "cache"
    )
    explicit_none = ProductionReplayBackend(
        events=raw_events,
        state_root=tmp_path / "state",
        prediction_cache_root=tmp_path / "cache",
        research_cumulative_pull_cap=None,
    )
    capped = ProductionReplayBackend(
        events=raw_events,
        state_root=tmp_path / "state",
        prediction_cache_root=tmp_path / "cache",
        research_cumulative_pull_cap=0.25,
    )
    assert omitted._state_digest == explicit_none._state_digest
    # A real cap must land in a genuinely separate state/cache directory tree --
    # the distinct-cache-dimension requirement -- never silently reuse or collide
    # with the uncapped cache.
    assert capped._state_digest != omitted._state_digest
    assert capped._prefix_dir_root() != omitted._prefix_dir_root()


def test_research_min_field_coverage_defaults_to_byte_identical_state_digest(tmp_path) -> None:
    """Round-9 fix-B backtest hatch: same no-op-by-default proof as fix A --
    omitted/None must reproduce today's digest exactly; a real threshold must
    land in its own separate state/cache tree, and independently of any pull
    cap that might also be set (both are separate digest keys)."""
    raw_events = _catalog(1)
    omitted = ProductionReplayBackend(
        events=raw_events, state_root=tmp_path / "state", prediction_cache_root=tmp_path / "cache"
    )
    explicit_none = ProductionReplayBackend(
        events=raw_events,
        state_root=tmp_path / "state",
        prediction_cache_root=tmp_path / "cache",
        research_min_field_coverage=None,
    )
    gated = ProductionReplayBackend(
        events=raw_events,
        state_root=tmp_path / "state",
        prediction_cache_root=tmp_path / "cache",
        research_min_field_coverage=0.5,
    )
    assert omitted._state_digest == explicit_none._state_digest
    assert gated._state_digest != omitted._state_digest
    assert gated._prefix_dir_root() != omitted._prefix_dir_root()


def test_capped_session_weights_freezes_after_hitting_the_ceiling() -> None:
    """ceiling == base_weight: the first session alone already hits the ceiling,
    so every later session in the same checkpoint contributes nothing further."""
    weights = backend_module._capped_session_weights(3, 0.25, base_weight=0.25)
    assert weights == [0.25, 0.0, 0.0]
    # Cumulative pull after each session must never exceed the ceiling.
    pull = 0.0
    for w in weights:
        pull += (1.0 - pull) * w
    assert pull == 0.25


def test_capped_session_weights_tops_up_to_a_higher_ceiling_then_freezes() -> None:
    weights = backend_module._capped_session_weights(3, 0.35, base_weight=0.25)
    assert weights[0] == 0.25
    assert 0.0 < weights[1] < 0.25
    assert weights[2] == 0.0
    pull = 0.0
    for w in weights:
        pull += (1.0 - pull) * w
    assert abs(pull - 0.35) < 1e-9


def test_capped_session_weights_never_exceeds_base_weight_or_ceiling() -> None:
    """A generous ceiling (>= what n sessions could ever reach) must reproduce
    today's flat 0.25-per-session behavior exactly -- proof the cap only ever
    REDUCES trust, never inflates it beyond the existing default."""
    weights = backend_module._capped_session_weights(3, 0.99, base_weight=0.25)
    assert weights == [0.25, 0.25, 0.25]


def test_q0_variant_has_no_race_side_component_by_construction() -> None:
    """q0 only ever gates qualifying_mixin; race prediction_mixin never checks it.

    This is what makes conditional_actual_grid race output identical between
    champion and q0 at a matched seed: q0's component set cannot reach any of the
    flags (r0/r1/r2_no_anchor/r2_source_anchor) that predict_race branches on.
    """
    q0_components = VARIANT_COMPONENTS["q0_driver_state"]
    race_side_components = {"r0", "r1", "r2_no_anchor", "r2_source_anchor"}
    assert q0_components.isdisjoint(race_side_components)


def _relaxed_manifest(*, component: str, floor: int) -> dict[str, Any]:
    return {
        "variant_id": ("q1_qualifying_practice" if component == "q1" else "r2_source_anchor"),
        "metadata": {"research_gate_relaxation": {component: floor}},
    }


def _mock_state_building(monkeypatch) -> None:
    """fit_fold always force-builds season/checkpoint state first; these tests care
    about what happens *after* that, so replace the (real session-replay, real
    FastF1 cache) machinery with no-ops, same as the state-building tests above."""
    monkeypatch.setattr(
        "src.analysis.challenger_research_backend._apply_session_update",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "src.analysis.challenger_research_backend._reset_replay_artifacts",
        lambda processed_dir, *, year: processed_dir.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(
        "src.analysis.challenger_research_backend.update_from_race",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.analysis.challenger_research_backend.shutil.copytree",
        lambda src, dst: dst.mkdir(parents=True, exist_ok=True),
    )


def test_q1_research_fit_refuses_without_a_curated_track_class_binding(
    tmp_path, monkeypatch
) -> None:
    """A real attempt, a real (never fabricated/pooled) refusal reason.

    No 2026 research event carries a curated track_class_by_event binding, so every
    Q1 research fold must refuse rather than pool unlike tracks under a fabricated
    class -- this is the honest finding the fitter's own contract demands.
    """
    _mock_state_building(monkeypatch)
    raw_events = _catalog(6)
    catalog = _normalise_catalog(raw_events)
    backend = _backend(tmp_path, raw_events)
    target = catalog[5]
    training = list(catalog[:4])

    try:
        backend.fit_fold(
            training_events=training,
            calibration_events=[catalog[4]],
            target_event=target,
            checkpoint="PRE",
            manifest=_relaxed_manifest(component="q1", floor=4),
        )
    except Exception as exc:  # noqa: BLE001 - assert the exact refusal type/reason
        from src.analysis.challenger_walk_forward import CheckpointInputUnavailable

        assert isinstance(exc, CheckpointInputUnavailable)
        assert "track_class" in str(exc)
    else:
        raise AssertionError("q1 research fit should refuse without a track_class binding")


def test_q1_research_fit_rejects_a_training_event_at_or_after_the_target(
    tmp_path, monkeypatch
) -> None:
    """Leakage guard: fold N's Q1 research fit must reject a non-chronological
    training event before it ever reaches the track-class check."""
    _mock_state_building(monkeypatch)
    raw_events = _catalog(2)
    catalog = _normalise_catalog(raw_events)
    backend = _backend(tmp_path, raw_events)
    target = catalog[0]

    try:
        backend.fit_fold(
            training_events=[target],  # the target itself: not strictly earlier
            calibration_events=[],
            target_event=target,
            checkpoint="PRE",
            manifest=_relaxed_manifest(component="q1", floor=4),
        )
    except ValueError as exc:
        assert "non-chronological" in str(exc)
    else:
        raise AssertionError("fit_fold accepted a training event at/after its target")


def test_r2_source_anchor_shrinkage_scales_with_training_event_count(tmp_path, monkeypatch) -> None:
    """The calibrated weight is real (from fit_source_specific_grid_anchors over
    real-shaped rows); the shrinkage-to-champion weight is n_training_events/8."""
    raw_events = _catalog(4)
    catalog = _normalise_catalog(raw_events)
    backend = _backend(tmp_path, raw_events)

    def fake_rows(*, training_event):  # noqa: ANN001 - test double
        # Rows that make the calibration converge to weight 1.0 (grid_position
        # perfectly predicts actual_position; simulated_position does not).
        return [
            {
                "event_id": training_event.event_id,
                "event_at": training_event.event_start_at.isoformat(),
                "grid_source_detail": "actual_starting_grid",
                "driver_id": f"D{i}",
                "simulated_position": float(5 - i),
                "grid_position": float(i),
                "actual_position": float(i),
            }
            for i in range(1, 5)
        ]

    monkeypatch.setattr(backend, "_r2_no_anchor_calibration_rows", lambda **kw: fake_rows(**kw))

    result = backend._fit_r2_source_anchor(
        training_events=list(catalog[:3]),
        target_event=catalog[3],
        relaxed_floor=3,
    )
    assert result["status"] == "fitted"
    assert result["calibrated_weight"] == 1.0
    assert result["shrinkage_weight"] == 3 / 8
    assert result["n_training_events"] == 3

    # Below the relaxed floor: fit_source_specific_grid_anchors legitimately
    # refuses (insufficient_events), and that refusal must be surfaced, not hidden.
    sparse_result = backend._fit_r2_source_anchor(
        training_events=list(catalog[:1]),
        target_event=catalog[3],
        relaxed_floor=3,
    )
    assert sparse_result["status"] == "insufficient_events"


def test_identity_guard_flags_champion_identical_challenger_predictions(tmp_path) -> None:
    """A challenger view claiming a race-side component that produced a
    champion-identical finish order must be flagged, never silently scored."""
    raw_events = _catalog(1)
    backend = _backend(tmp_path, raw_events)
    cache_key = {
        "kind": "race_views",
        "source_digest": backend._source_digest,
        "event_id": "event-00",
        "checkpoint": "PRE",
        "variant_id": "r2_source_anchor",
        "seed": 17,
    }
    identical_finish_order = [{"driver": "D1", "position": 1}, {"driver": "D2", "position": 2}]
    different_finish_order = [{"driver": "D2", "position": 1}, {"driver": "D1", "position": 2}]
    backend._store(
        {**cache_key, "variant_id": "champion"},
        {
            "conditional_actual_grid": {"finish_order": identical_finish_order},
            "end_to_end_predicted_grid": {"finish_order": identical_finish_order},
        },
    )

    views = {
        "conditional_actual_grid": {"finish_order": identical_finish_order},
        "end_to_end_predicted_grid": {"finish_order": different_finish_order},
    }
    backend._apply_identity_guard(
        views=views,
        components=VARIANT_COMPONENTS["r2_source_anchor"],
        cache_key=cache_key,
    )

    assert views["conditional_actual_grid"]["ineffective_for_fold"] is True
    assert "r2_source_anchor" in views["conditional_actual_grid"]["ineffective_reason"]
    assert "ineffective_for_fold" not in views["end_to_end_predicted_grid"]


def _fake_q1_feature_rows(event_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, driver in enumerate(("D1", "D2", "D3", "D4"), start=1):
        row = {"event_id": event_id, "driver": driver, "actual_position": float(index)}
        for column in DEFAULT_FEATURE_COLUMNS:
            row[column] = float(index) + (0.1 if column == "best_adjusted_lap_s" else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_q1_research_fit_rejects_a_prior_event_at_or_after_the_fold_cutoff(tmp_path) -> None:
    """Leakage guard: a "prior" event whose own data starts at/after the fold's
    cutoff must be rejected, even though _q1_track_class_eligibility should never
    hand fit_fold one -- this is the defense-in-depth check inside the fit itself."""
    raw_events = _catalog(2)  # event-00 (target), event-01 (chronologically LATER)
    catalog = _normalise_catalog(raw_events)
    backend = _backend(tmp_path, raw_events)
    target = catalog[0]  # earliest event; event-01 starts 14 days after it

    def fake_rows(self, *, event_id, session_labels, track_class):  # noqa: ANN001
        return _fake_q1_feature_rows(event_id)

    original = backend_module.ProductionReplayBackend._q1_prior_event_feature_rows
    backend_module.ProductionReplayBackend._q1_prior_event_feature_rows = fake_rows
    try:
        launch_path, diagnostics = backend._fit_q1_research_launch(
            target_event=target,
            checkpoint="FP1",
            matching_event_ids=["event-01"],  # later than target-00's own cutoff
            relaxed_floor=4,
            track_class="permanent",
        )
    finally:
        backend_module.ProductionReplayBackend._q1_prior_event_feature_rows = original

    assert launch_path is None
    assert diagnostics["status"] == "leakage_guard_rejected"


def test_q1_research_fit_dry_run_verifies_real_runtime_resolution(tmp_path) -> None:
    """The fold-fit's success claim must come from an actual
    resolve_qualifying_practice_launch_envelope() call against the real written
    artifacts (real digests, real chronology), not merely from writing files --
    prove it by checking the diagnostics carry the real resolution outcome (with
    its exact reason/cutoffs on failure, or an explicit verified flag on success),
    never a bare "fitted" claim with no runtime evidence behind it."""
    raw_events = _catalog(2)
    catalog = _normalise_catalog(raw_events)
    backend = _backend(tmp_path, raw_events)
    target = catalog[1]

    def fake_rows(self, *, event_id, session_labels, track_class):  # noqa: ANN001
        return _fake_q1_feature_rows(event_id)

    original = backend_module.ProductionReplayBackend._q1_prior_event_feature_rows
    backend_module.ProductionReplayBackend._q1_prior_event_feature_rows = fake_rows
    try:
        launch_path, diagnostics = backend._fit_q1_research_launch(
            target_event=target,
            checkpoint="FP1",
            matching_event_ids=["event-00"],
            relaxed_floor=4,
            track_class="permanent",
        )
    finally:
        backend_module.ProductionReplayBackend._q1_prior_event_feature_rows = original

    # Real 2026 events are always historical (built "today" targets a past fold),
    # so the dry-run resolution genuinely executes and genuinely fails closed --
    # this is the expected, disclosed outcome (see diagnostics["note"]), proving
    # the check is real rather than a rubber stamp. If the underlying artifact
    # chain is ever used for a future/live fold where resolution can succeed,
    # runtime_resolution_verified is set explicitly (never implied by file
    # existence alone).
    assert diagnostics["status"] in {"runtime_resolution_failed", "fitted"}
    if diagnostics["status"] == "fitted":
        assert diagnostics["runtime_resolution_verified"] is True
        assert launch_path is not None and launch_path.is_file()
    else:
        assert launch_path is None
        assert "resolution_cutoff" in diagnostics
        assert "envelope_created_at" in diagnostics

#!/usr/bin/env python3
"""Evaluate immutable challenger forecasts in chronological walk-forward order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.challenger_governance import (  # noqa: E402
    evaluate_qualifying_gate,
    evaluate_race_gate,
)
from src.analysis.challenger_release import build_gate_result_envelope  # noqa: E402
from src.analysis.challenger_walk_forward import (  # noqa: E402
    FrozenPredictionBundleBackend,
    build_qualifying_gate_metrics_from_walk_forward,
    build_race_gate_metrics_from_walk_forward,
    run_challenger_walk_forward,
)
from src.persistence.research_sidecar import (  # noqa: E402
    DEFAULT_RESEARCH_SIDECAR_ROOT,
    ResearchSidecarStore,
)


def _read_json(path: Path, *, field_name: str) -> Any:
    try:
        return json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {field_name}: {path}") from exc


def _event_catalog(payload: Any) -> list[dict[str, Any]]:
    raw_events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("event catalog must be a non-empty list or {'events': [...]} object")
    if not all(isinstance(event, dict) for event in raw_events):
        raise ValueError("event catalog rows must be JSON objects")
    return [dict(event) for event in raw_events]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--event-catalog",
        type=Path,
        required=True,
        help=(
            "Catalog whose checkpoints contain information_cutoff_at plus immutable "
            "forecast_reference sidecars; embedded predictions are rejected."
        ),
    )
    parser.add_argument("--movement-reviews", type=Path)
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=DEFAULT_RESEARCH_SIDECAR_ROOT,
        help="Immutable challenger-only output root.",
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--qualifying-target",
        choices=("main_qualifying", "sprint_qualifying"),
    )
    parser.add_argument(
        "--race-target",
        choices=("grand_prix_race", "sprint_race"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _read_json(args.manifest, field_name="manifest")
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    events = _event_catalog(_read_json(args.event_catalog, field_name="event catalog"))
    reviews: dict[str, Any] | None = None
    if args.movement_reviews is not None:
        raw_reviews = _read_json(args.movement_reviews, field_name="movement reviews")
        if not isinstance(raw_reviews, dict):
            raise ValueError("movement reviews must be a JSON object")
        reviews = raw_reviews

    replay = run_challenger_walk_forward(
        events=events,
        manifest=manifest,
        backend=FrozenPredictionBundleBackend(),
        movement_reviews=reviews,
    )
    payload: dict[str, Any] = {"replay": replay, "promotion_gates": {}}
    if args.qualifying_target:
        qualifying_metrics = build_qualifying_gate_metrics_from_walk_forward(
            replay,
            manifest=manifest,
            target=args.qualifying_target,
        )
        gate_result = evaluate_qualifying_gate(qualifying_metrics).to_dict()
        payload["promotion_gates"][args.qualifying_target] = build_gate_result_envelope(
            manifest=manifest,
            gate_result=gate_result,
        )
    if args.race_target:
        race_metrics = build_race_gate_metrics_from_walk_forward(
            replay,
            manifest=manifest,
            target=args.race_target,
        )
        gate_result = evaluate_race_gate(race_metrics).to_dict()
        payload["promotion_gates"][args.race_target] = build_gate_result_envelope(
            manifest=manifest,
            gate_result=gate_result,
        )

    store = ResearchSidecarStore(root=args.sidecar_root, repo_root=args.repo_root)
    store.write_manifest(manifest)
    output_path = store.write_artifact(
        manifest=manifest,
        artifact_kind="walk_forward_replay",
        payload=payload,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

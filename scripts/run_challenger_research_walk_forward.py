#!/usr/bin/env python3
"""Run the 2026 walk-forward comparison with the real production-replay backend.

Each requested variant runs once through ``run_challenger_walk_forward``; the runner
always scores champion alongside the requested challenger variant (see
``challenger_walk_forward.py``), so N variant runs give N champion-vs-variant
comparisons that all share one cached champion prediction per (event, checkpoint,
seed) via ``ProductionReplayBackend``'s disk cache.

Output: one raw replay JSON per variant under
``data/historical_replay/2026/walk_forward_runs/<variant_id>.json``, plus a run
manifest recording exactly what ran, what was skipped, and why.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.challenger_governance import (  # noqa: E402
    DEFAULT_CONFIG_PATHS,
    DEFAULT_REPLAY_SEEDS,
    build_challenger_manifest,
)
from src.analysis.challenger_research_backend import ProductionReplayBackend  # noqa: E402
from src.analysis.challenger_walk_forward import run_challenger_walk_forward  # noqa: E402

CATALOG_PATH = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "event_catalog.json"
RUNS_DIR = PROJECT_ROOT / "data" / "historical_replay" / "2026" / "walk_forward_runs"

# Priority order per the phase-2 mission: champion + r0/r1/r2 first, then the
# combinations, then q1 (expected to fail closed on the 30-main-event minimum).
DEFAULT_VARIANTS = [
    "q0_driver_state",
    "r0_long_run",
    "r1_joint_grid",
    "r2_no_anchor",
    "r2_source_anchor",
    "r1_r2_no_anchor",
    "r1_r2_source_anchor",
    "q1_qualifying_practice",
]


def _load_catalog(
    *,
    checkpoints: list[str] | None,
    main_checkpoints: list[str] | None,
    sprint_checkpoints: list[str] | None,
) -> list[dict[str, Any]]:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    events = [dict(row) for row in payload["events"]]
    for row in events:
        row.setdefault("year", 2026)
        if main_checkpoints is not None or sprint_checkpoints is not None:
            allowed = main_checkpoints if row["session_kind"] == "main" else sprint_checkpoints
            allowed = allowed if allowed is not None else []
        elif checkpoints is not None:
            allowed = checkpoints
        else:
            allowed = None
        if allowed is not None:
            row["checkpoint_payloads"] = {
                checkpoint: cp
                for checkpoint, cp in row["checkpoint_payloads"].items()
                if checkpoint in allowed
            }
    return events


def _pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check for a PID recorded by an earlier run.

    A stale/false-positive lock is worse than a rare missed race, so this only
    returns True when a process table lookup positively finds the PID.
    """
    if os.name == "nt":
        try:
            output = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return False
        return str(pid) in output
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


@contextlib.contextmanager
def _single_runner_lock(run_tag: str) -> Iterator[None]:
    """Refuse a second concurrent runner for the same run_tag.

    Two detached campaigns launched for the same run_tag would race on the same
    cache/state directories and on the same output files; this is a plain
    PID-in-a-lockfile guard, not a distributed lock -- it only needs to catch the
    "I launched this twice" mistake, not survive machine failure.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = RUNS_DIR / f".lock-{run_tag or 'default'}"
    if lock_path.is_file():
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing_pid = -1
        if existing_pid > 0 and _pid_is_alive(existing_pid):
            raise SystemExit(
                f"Refusing to start: another runner (PID {existing_pid}) already holds "
                f"the '{run_tag or 'default'}' run_tag lock at {lock_path}. Wait for it to "
                "exit, or pass a different --run-tag."
            )
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--qualifying-simulations", type=int, default=30)
    parser.add_argument("--race-simulations", type=int, default=30)
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Restrict scored checkpoints uniformly (e.g. PRE) to bound runtime.",
    )
    parser.add_argument(
        "--main-checkpoints",
        nargs="*",
        default=None,
        help="Restrict checkpoints for main-format events only (e.g. FP2 FP3).",
    )
    parser.add_argument(
        "--sprint-checkpoints",
        nargs="*",
        default=None,
        help="Restrict checkpoints for sprint-format events only (e.g. FP1).",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Suffix appended to output filenames so a follow-up run never "
        "overwrites an earlier one (e.g. 'fp_hisim').",
    )
    parser.add_argument(
        "--research-gate-relaxation",
        nargs="*",
        default=None,
        metavar="COMPONENT=MIN_EVENTS",
        help="Research-only override of the Q1/R2-source-anchor minimum-training-"
        "event gate (e.g. 'q1=4 r2_source_anchor=3'). Floor-clamped; never changes "
        "production defaults; every manifest built with it carries a "
        "research_gate_relaxation marker that the release gate always rejects.",
    )
    parser.add_argument(
        "--research-cumulative-pull-cap",
        type=float,
        default=None,
        metavar="CEILING",
        help="Research-only backtest override (round 9): caps the cumulative "
        "practice-characteristics EWMA pull toward this weekend's own sessions at "
        "CEILING (e.g. 0.25 or 0.35), instead of letting it compound uncapped "
        "through FP1/FP2/FP3. Default OFF (None); never touches "
        "config/production_config.json, config/default.yaml, or the live-serving "
        "path -- confined to ProductionReplayBackend's own checkpoint-state build. "
        "Uses a distinct cache dimension from every other run.",
    )
    parser.add_argument(
        "--research-min-field-coverage",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Research-only backtest override (round 9, Fix B): a practice "
        "session only moves car characteristics when at least THRESHOLD fraction "
        "of the field has robust evidence (mirrors r0's MIN_R0_TEAM_COVERAGE "
        "gate, e.g. 0.50 or 0.70); below threshold the session contributes "
        "nothing. Default OFF (None); same production/live-path guarantees and "
        "distinct-cache-dimension behavior as --research-cumulative-pull-cap; "
        "may be combined with it for an A+B backtest.",
    )
    args = parser.parse_args()
    relaxation: dict[str, int] | None = None
    if args.research_gate_relaxation:
        relaxation = {}
        for entry in args.research_gate_relaxation:
            component, _, raw_value = entry.partition("=")
            relaxation[component.strip()] = int(raw_value)

    with _single_runner_lock(args.run_tag):
        events = _load_catalog(
            checkpoints=args.checkpoints,
            main_checkpoints=args.main_checkpoints,
            sprint_checkpoints=args.sprint_checkpoints,
        )
        snapshot_ids = sorted({sid for row in events for sid in row["input_snapshot_ids"]})
        backend = ProductionReplayBackend(
            events=events,
            qualifying_simulations=args.qualifying_simulations,
            race_simulations=args.race_simulations,
            research_cumulative_pull_cap=args.research_cumulative_pull_cap,
            research_min_field_coverage=args.research_min_field_coverage,
        )

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_manifest: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "event_catalog_path": str(CATALOG_PATH),
            "event_count": len(events),
            "event_ids": [row["event_id"] for row in events],
            "seeds": list(DEFAULT_REPLAY_SEEDS),
            "qualifying_simulations": args.qualifying_simulations,
            "race_simulations": args.race_simulations,
            "checkpoints_filter": args.checkpoints,
            "main_checkpoints_filter": args.main_checkpoints,
            "sprint_checkpoints_filter": args.sprint_checkpoints,
            "run_tag": args.run_tag,
            "research_gate_relaxation": relaxation,
            "research_cumulative_pull_cap": args.research_cumulative_pull_cap,
            "research_min_field_coverage": args.research_min_field_coverage,
            "results": {},
        }
        tag_suffix = f"__{args.run_tag}" if args.run_tag else ""

        for variant_id in args.variants:
            print(f"=== {variant_id} ===", flush=True)
            candidate_id = f"research-2026-walk-forward-{variant_id.replace('_', '-')}"
            try:
                manifest = build_challenger_manifest(
                    repo_root=PROJECT_ROOT,
                    candidate_id=candidate_id,
                    variant_id=variant_id,
                    feature_schema=f"2026-walk-forward-{variant_id}-v1",
                    input_snapshot_ids=snapshot_ids,
                    cutoff_at=datetime.now(UTC),
                    simulation_counts={
                        "qualifying": args.qualifying_simulations,
                        "race": args.race_simulations,
                    },
                    seeds=DEFAULT_REPLAY_SEEDS,
                    config_paths=DEFAULT_CONFIG_PATHS,
                    metadata=(
                        {"research_gate_relaxation": relaxation} if relaxation is not None else None
                    ),
                )
                replay = run_challenger_walk_forward(
                    events=events,
                    manifest=manifest,
                    backend=backend,
                    research_gate_relaxation=relaxation,
                )
            except Exception as exc:  # noqa: BLE001 - record every refusal, never crash the matrix
                run_manifest["results"][variant_id] = {
                    "status": "refused",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"  refused: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                continue

            output_path = RUNS_DIR / f"{variant_id}{tag_suffix}.json"
            output_path.write_text(
                json.dumps(
                    {"manifest": manifest, "replay": replay}, indent=2, sort_keys=True, default=str
                )
                + "\n",
                encoding="utf-8",
            )
            run_manifest["results"][variant_id] = {
                "status": "scored",
                "output_path": str(output_path),
                "scored_event_count": len(replay["scored_events"]),
                "skipped_event_count": len(replay["skipped_events"]),
                "checkpoint_refusal_count": len(replay["checkpoint_refusals"]),
                "checkpoint_refusals": replay["checkpoint_refusals"],
                "manifest_sha256": manifest["manifest_sha256"],
            }
            print(
                f"  scored {len(replay['scored_events'])} events, "
                f"skipped {len(replay['skipped_events'])}, "
                f"checkpoint refusals {len(replay['checkpoint_refusals'])} -> {output_path}",
                flush=True,
            )
            for refusal in replay["checkpoint_refusals"]:
                print(
                    f"    REFUSED {refusal['event_id']} {refusal['checkpoint']}: {refusal['reason']}",
                    flush=True,
                )

        manifest_path = RUNS_DIR / f"run_manifest{tag_suffix}.json"
        manifest_path.write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        print(f"run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build checkpoint-aware predictors from stored car-characteristics snapshots."""

from __future__ import annotations

import logging
from typing import Any, cast

from src.persistence.artifact_store import ArtifactStore
from src.predictors.baseline_2026 import Baseline2026Predictor
from src.utils.checkpoint_reconstruction import (
    SnapshotOverlayArtifactStore,
    build_snapshot_overlay_car_characteristics,
    load_checkpoint_snapshot_payload,
)

logger = logging.getLogger(__name__)


def _resolve_base_artifact_store(store: Any) -> ArtifactStore | None:
    """Return the underlying artifact store for one predictor instance."""
    if isinstance(store, SnapshotOverlayArtifactStore):
        return store.base_store
    if hasattr(store, "load_artifact") and hasattr(store, "list_artifacts"):
        return cast(ArtifactStore, store)
    return None


def build_checkpoint_overlay_predictor(
    *,
    base_predictor: Any,
    year: int,
    race_name: str,
    checkpoint_session: str,
    is_sprint: bool,
) -> Any:
    """Return a predictor overlaid with the best stored snapshot for one checkpoint.

    The overlay rules intentionally mirror checkpoint reconstruction:
    exact session snapshot when present, and for ``PRE`` the latest snapshot
    captured before the target weekend begins. If the stored snapshot path is
    unavailable, the original predictor is returned unchanged.
    """
    checkpoint_session_upper = str(checkpoint_session or "").strip().upper()
    if not checkpoint_session_upper:
        return base_predictor

    artifact_store = _resolve_base_artifact_store(getattr(base_predictor, "artifact_store", None))
    if artifact_store is None:
        return base_predictor

    base_car_payload = artifact_store.load_artifact(
        "car_characteristics",
        f"{int(year)}::car_characteristics",
    )
    if not isinstance(base_car_payload, dict):
        logger.debug(
            "Checkpoint overlay skipped: missing base car characteristics for %s %s",
            race_name,
            year,
        )
        return base_predictor

    try:
        snapshot_payload = load_checkpoint_snapshot_payload(
            store=artifact_store,
            year=int(year),
            race_name=race_name,
            checkpoint_session=checkpoint_session_upper,
            is_sprint=bool(is_sprint),
        )
    except FileNotFoundError:
        logger.debug(
            "Checkpoint overlay skipped: no stored snapshot for %s %s %s",
            race_name,
            year,
            checkpoint_session_upper,
        )
        return base_predictor

    try:
        overlay_payload = build_snapshot_overlay_car_characteristics(
            base_car_payload=base_car_payload,
            snapshot_payload=snapshot_payload,
        )
    except ValueError as exc:
        logger.warning(
            "Checkpoint overlay skipped: invalid stored snapshot for %s %s %s: %s",
            race_name,
            year,
            checkpoint_session_upper,
            exc,
        )
        return base_predictor
    overlay_store = SnapshotOverlayArtifactStore(
        base_store=artifact_store,
        season_year=int(year),
        car_characteristics_payload=overlay_payload,
    )
    return Baseline2026Predictor(
        data_dir=str(getattr(base_predictor, "data_dir", "data/processed")),
        seed=int(getattr(base_predictor, "seed", 42)),
        season_year=int(year),
        config=getattr(base_predictor, "config", None),
        artifact_store=cast(ArtifactStore, overlay_store),
    )

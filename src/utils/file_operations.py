"""
Safe File Operations with Atomic Writes and Backups

Prevents data corruption by:
1. Writing to temporary file first
2. Creating backup of original
3. Atomic rename (move) operation
4. Rollback capability if write fails
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def atomic_json_write(file_path: Path, data: dict[str, Any], create_backup: bool = True) -> None:
    """Write JSON data to file atomically with optional backup, preserving original on failure."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic move
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=f".{file_path.name}.", dir=file_path.parent
    )

    try:
        # Write to temp file
        with open(temp_fd, "w") as f:
            json.dump(data, f, indent=2)

        # Create backup of original
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + ".backup")
            shutil.copy2(file_path, backup_path)
            logger.debug("Created backup: %s", backup_path)

        # Atomic move (on same filesystem, this is atomic)
        shutil.move(temp_path, file_path)
        logger.debug("Atomically wrote: %s", file_path)

    except (OSError, RuntimeError, TypeError, ValueError) as e:
        # Clean up temp file if it still exists
        try:
            Path(temp_path).unlink()
        except BaseException as exc:
            logger.debug("Could not remove temp file %s: %s", temp_path, exc)
        raise OSError(f"Failed to write {file_path}: {e}") from e

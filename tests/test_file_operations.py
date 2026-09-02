from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.utils.file_operations as file_ops


def test_atomic_json_write_updates_file_and_creates_backup(tmp_path):
    target = tmp_path / "state.json"
    target.write_text(json.dumps({"old": True}))

    file_ops.atomic_json_write(target, {"new": True}, create_backup=True)

    backup = Path(str(target) + ".backup")
    assert json.loads(target.read_text()) == {"new": True}
    assert json.loads(backup.read_text()) == {"old": True}


def test_atomic_json_write_skips_backup_when_disabled(tmp_path):
    target = tmp_path / "state.json"
    target.write_text(json.dumps({"old": True}))

    file_ops.atomic_json_write(target, {"new": True}, create_backup=False)

    backup = Path(str(target) + ".backup")
    assert json.loads(target.read_text()) == {"new": True}
    assert backup.exists() is False


def test_atomic_json_write_raises_and_preserves_original_on_move_failure(tmp_path, patcher):
    target = tmp_path / "state.json"
    target.write_text(json.dumps({"old": True}))

    patcher.setattr(
        file_ops.shutil, "move", lambda src, dst: (_ for _ in ()).throw(RuntimeError("disk full"))
    )

    with pytest.raises(OSError, match="Failed to write"):
        file_ops.atomic_json_write(target, {"new": True}, create_backup=True)

    assert json.loads(target.read_text()) == {"old": True}
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []

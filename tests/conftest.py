"""
Shared test fixtures and configuration.
"""

import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from src.models.bayesian import DriverPrior

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_DRIVER_CHARACTERISTICS_2026 = (
    _REPO_ROOT
    / "data"
    / "processed"
    / "driver_characteristics"
    / "2026_driver_characteristics.json"
)


def pytest_addoption(parser):
    """Register repo-local pytest options used by regression tests."""
    parser.addoption(
        "--update-golden",
        "--update-golden-files",
        action="store_true",
        default=False,
        help="Refresh golden regression fixtures instead of asserting current output.",
    )


class TestPatcher:
    """Small patch helper backed by unittest.mock for test-only replacements."""

    _UNSET = object()

    def __init__(self) -> None:
        self._stack = ExitStack()

    def setattr(self, target, name=_UNSET, value=_UNSET) -> None:
        """
        Patch attributes in two supported forms:
        - setattr(obj, "attr_name", replacement)
        - setattr("dotted.path.to.symbol", replacement)
        """
        if value is self._UNSET:
            if name is self._UNSET:
                raise TypeError("setattr expected replacement value")
            self._stack.enter_context(patch(target, new=name))
            return
        if name is self._UNSET:
            raise TypeError("setattr expected attribute name for object target")
        self._stack.enter_context(patch.object(target, name, new=value))

    def chdir(self, path: Path) -> None:
        previous_cwd = Path.cwd()
        os.chdir(path)
        self._stack.callback(os.chdir, previous_cwd)

    def close(self) -> None:
        self._stack.close()


@pytest.fixture
def patcher():
    helper = TestPatcher()
    try:
        yield helper
    finally:
        helper.close()


@pytest.fixture(autouse=True)
def default_file_only_storage(monkeypatch):
    """Run tests in file-only mode unless a test explicitly overrides persistence behavior."""
    monkeypatch.setenv("USE_DB_STORAGE", "file_only")


@pytest.fixture(scope="session")
def repo_driver_characteristics_2026_baseline() -> bytes | None:
    """Cache the tracked 2026 driver characteristics bytes for test isolation.

    Storing the raw bytes preserves exact file formatting, including whether
    the tracked JSON ends with a trailing newline.
    """
    if not _REPO_DRIVER_CHARACTERISTICS_2026.exists():
        return None
    return _REPO_DRIVER_CHARACTERISTICS_2026.read_bytes()


@pytest.fixture(autouse=True)
def restore_repo_driver_characteristics_2026(repo_driver_characteristics_2026_baseline):
    """Restore the tracked 2026 driver file after tests that write fallback payloads.

    Some updater tests exercise code paths that persist a season-scoped fallback
    file relative to the repo root. Restoring the tracked file after each test
    prevents later tests from inheriting mutated driver form data based only on
    execution order.
    """
    yield

    if repo_driver_characteristics_2026_baseline is None:
        return
    if not _REPO_DRIVER_CHARACTERISTICS_2026.exists():
        return

    current_bytes = _REPO_DRIVER_CHARACTERISTICS_2026.read_bytes()
    if current_bytes != repo_driver_characteristics_2026_baseline:
        _REPO_DRIVER_CHARACTERISTICS_2026.write_bytes(
            repo_driver_characteristics_2026_baseline,
        )


@pytest.fixture
def sample_priors():
    """Standard set of driver priors for testing."""
    return {
        "1": DriverPrior(
            driver_number="1",
            driver_code="VER",
            team="Red Bull Racing",
            team_tier="top",
            mu=18.0,
            sigma=2.0,
        ),
        "4": DriverPrior(
            driver_number="4",
            driver_code="NOR",
            team="McLaren",
            team_tier="top",
            mu=17.0,
            sigma=2.5,
        ),
        "44": DriverPrior(
            driver_number="44",
            driver_code="HAM",
            team="Ferrari",
            team_tier="top",
            mu=17.5,
            sigma=2.2,
        ),
        "77": DriverPrior(
            driver_number="77",
            driver_code="BOT",
            team="Cadillac",
            team_tier="backmarker",
            mu=10.0,
            sigma=3.0,
        ),
    }


@pytest.fixture
def mock_driver_chars():
    """Mock driver characteristics data."""
    return {
        "VER": {
            "racecraft": {"skill_score": 0.95},
            "consistency": {"score": 0.90, "error_rate_wet": 0.05},
            "tire_management": {"degradation_factor": 0.3},
        },
        "NOR": {
            "racecraft": {"skill_score": 0.85},
            "consistency": {"score": 0.80, "error_rate_wet": 0.15},
            "tire_management": {"degradation_factor": 0.5},
        },
        "HAM": {
            "racecraft": {"skill_score": 0.90},
            "consistency": {"score": 0.85, "error_rate_wet": 0.10},
            "tire_management": {"degradation_factor": 0.4},
        },
        "BOT": {
            "racecraft": {"skill_score": 0.70},
            "consistency": {"score": 0.75, "error_rate_wet": 0.25},
            "tire_management": {"degradation_factor": 0.6},
        },
    }


@pytest.fixture
def mock_qualifying_grid():
    """Mock qualifying grid."""
    return [
        {"driver": "VER", "team": "Red Bull Racing", "position": 1},
        {"driver": "NOR", "team": "McLaren", "position": 2},
        {"driver": "HAM", "team": "Ferrari", "position": 3},
        {"driver": "BOT", "team": "Cadillac", "position": 10},
    ]


@pytest.fixture
def mock_track_data():
    """Mock track characteristics."""
    return {
        "Bahrain Grand Prix": {
            "pit_stop_loss": 22.0,
            "safety_car_prob": 0.3,
            "overtaking_difficulty": 0.4,
            "type": "permanent",
        },
        "Monaco Grand Prix": {
            "pit_stop_loss": 25.0,
            "safety_car_prob": 0.7,
            "overtaking_difficulty": 0.9,
            "type": "street",
        },
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True)

    # Create minimal required files
    (processed / "driver_characteristics.json").write_text('{"drivers": {}}')
    (processed / "track_characteristics.json").write_text('{"tracks": {}}')

    return data_dir


@pytest.fixture
def update_golden_files(request):
    """Return whether the current pytest run should rewrite golden fixtures."""
    return bool(request.config.getoption("--update-golden"))

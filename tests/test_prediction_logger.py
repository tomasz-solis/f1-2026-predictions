"""
Tests for Prediction Logger
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.persistence.artifact_store import ArtifactStore
from src.utils.model_version import get_model_version
from src.utils.prediction_logger import PredictionLogger


@pytest.fixture
def temp_predictions_dir():
    """
    Create temporary directory for predictions.

    Returns path that looks like: /tmp/test_xyz/predictions
    ArtifactStore will write to /tmp/test_xyz/predictions/,
    isolated per test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Return predictions subdir - ArtifactStore will create it
        yield str(Path(tmpdir) / "predictions")


@pytest.fixture(autouse=True)
def mock_supabase():
    """Mock Supabase client to prevent DB access during tests."""
    with patch("src.persistence.db.get_supabase_client") as mock_client:
        # Return a mock that raises an exception if called
        mock_client.side_effect = RuntimeError(
            "Supabase should not be accessed in file-only mode tests"
        )
        yield mock_client


@pytest.fixture
def sample_quali_prediction():
    """Sample qualifying prediction."""
    return [
        {
            "driver": "Verstappen",
            "team": "Red Bull",
            "expected_time": 78.5,
            "confidence": 0.8,
        },
        {
            "driver": "Norris",
            "team": "McLaren",
            "expected_time": 78.7,
            "confidence": 0.75,
        },
        {
            "driver": "Leclerc",
            "team": "Ferrari",
            "expected_time": 78.8,
            "confidence": 0.72,
        },
    ]


@pytest.fixture
def sample_race_prediction():
    """Sample race prediction."""
    return [
        {
            "driver": "Verstappen",
            "team": "Red Bull",
            "confidence": 0.8,
            "dnf_risk": 0.05,
        },
        {"driver": "Norris", "team": "McLaren", "confidence": 0.75, "dnf_risk": 0.07},
        {"driver": "Leclerc", "team": "Ferrari", "confidence": 0.72, "dnf_risk": 0.08},
    ]


def test_save_prediction(temp_predictions_dir, sample_quali_prediction, sample_race_prediction):
    """Test saving a prediction."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    # Verify by loading the prediction back
    prediction = logger.load_prediction(2026, "Bahrain Grand Prix", "FP1")

    assert prediction is not None
    assert prediction["metadata"]["year"] == 2026
    assert prediction["metadata"]["race_name"] == "Bahrain Grand Prix"
    assert prediction["metadata"]["session_name"] == "FP1"
    assert prediction["metadata"]["weather"] == "dry"
    assert prediction["metadata"]["model_version"] == get_model_version()
    assert len(prediction["qualifying"]["predicted_grid"]) == 3
    assert len(prediction["race"]["predicted_results"]) == 3
    assert prediction["actuals"]["qualifying"] is None
    assert prediction["actuals"]["race"] is None


def test_save_prediction_always_writes_file_copy(
    temp_predictions_dir,
    sample_quali_prediction,
    sample_race_prediction,
):
    """Prediction saves should keep a filesystem copy even when ArtifactStore succeeds."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)
    save_calls: list[tuple[str, str]] = []
    logger.artifact_store.save_artifact = (
        lambda artifact_type, artifact_key, data, version, run_id: (
            save_calls.append((artifact_type, artifact_key)) or {"artifact_key": artifact_key}
        )
    )

    saved_path = logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    assert saved_path.exists()
    assert save_calls == [("prediction", "2026::Bahrain Grand Prix::FP1")]


def test_load_prediction(temp_predictions_dir, sample_quali_prediction, sample_race_prediction):
    """Test loading a saved prediction."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # Save first
    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP2",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    # Load
    prediction = logger.load_prediction(2026, "Bahrain Grand Prix", "FP2")

    assert prediction is not None
    assert prediction["metadata"]["race_name"] == "Bahrain Grand Prix"
    assert prediction["metadata"]["session_name"] == "FP2"


def test_prediction_identity_is_normalized_for_storage_and_lookup(
    temp_predictions_dir,
    sample_quali_prediction,
    sample_race_prediction,
):
    """Saving one checkpoint should not depend on caller case or whitespace drift."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    logger.save_prediction(
        year=2026,
        race_name="  Chinese   Grand Prix  ",
        session_name=" fp1 ",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    prediction = logger.load_prediction(2026, "Chinese Grand Prix", "FP1")

    assert prediction is not None
    assert prediction["metadata"]["race_name"] == "Chinese Grand Prix"
    assert prediction["metadata"]["session_name"] == "FP1"
    assert logger.has_prediction_for_session(2026, "Chinese Grand Prix", " fp1 ") is True


def test_prediction_logger_and_artifact_store_use_same_prediction_path(temp_predictions_dir):
    """Canonical prediction paths should not drift between persistence layers."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)
    store = ArtifactStore(data_root=Path(temp_predictions_dir).parent)

    logger_path = logger._prediction_file_path(2026, "Bahrain Grand Prix", "FP1")
    store_path = store._get_file_path("prediction", "2026::Bahrain Grand Prix::FP1")

    assert logger_path == store_path


@pytest.mark.parametrize(
    "race_name,session_name",
    [
        ("..", "FP1"),
        ("Bahrain/Grand Prix", "FP1"),
        ("Bahrain\\Grand Prix", "FP1"),
        ("Bahrain\nGrand Prix", "FP1"),
        ("C:\\temp", "FP1"),
        ("Bahrain Grand Prix", ".."),
        ("Bahrain Grand Prix", "FP/1"),
        ("Bahrain Grand Prix", "FP\t1"),
        ("Bahrain Grand Prix", "C:\\temp"),
        ("Bahrain Grand Prix", ""),
    ],
)
def test_prediction_logger_rejects_unsafe_storage_identity(
    temp_predictions_dir,
    sample_quali_prediction,
    sample_race_prediction,
    race_name,
    session_name,
):
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    with pytest.raises(ValueError):
        logger.save_prediction(
            year=2026,
            race_name=race_name,
            session_name=session_name,
            qualifying_prediction=sample_quali_prediction,
            race_prediction=sample_race_prediction,
            weather="dry",
        )


def test_load_nonexistent_prediction(temp_predictions_dir):
    """Test loading a prediction that doesn't exist."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    prediction = logger.load_prediction(2026, "Monaco Grand Prix", "FP1")

    assert prediction is None


def test_update_actuals(temp_predictions_dir, sample_quali_prediction, sample_race_prediction):
    """Test updating a prediction with actual results."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # Save prediction
    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP3",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    # Add actuals
    actual_quali = [
        {"driver": "Verstappen", "team": "Red Bull"},
        {"driver": "Leclerc", "team": "Ferrari"},
        {"driver": "Norris", "team": "McLaren"},
    ]

    actual_race = [
        {"driver": "Verstappen", "team": "Red Bull"},
        {"driver": "Norris", "team": "McLaren"},
        {"driver": "Leclerc", "team": "Ferrari"},
    ]

    success = logger.update_actuals(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP3",
        qualifying_results=actual_quali,
        race_results=actual_race,
    )

    assert success is True

    # Verify actuals were saved
    prediction = logger.load_prediction(2026, "Bahrain Grand Prix", "FP3")
    assert prediction["actuals"]["qualifying"] is not None
    assert prediction["actuals"]["race"] is not None
    assert len(prediction["actuals"]["qualifying"]) == 3
    assert len(prediction["actuals"]["race"]) == 3


def test_save_prediction_and_update_actuals_with_targets(
    temp_predictions_dir,
    sample_quali_prediction,
    sample_race_prediction,
):
    """Target payloads and target actuals should persist alongside legacy fields."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    logger.save_prediction(
        year=2026,
        race_name="Chinese Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
        metadata={"weekend_format": "sprint"},
        target_predictions={
            "sprint_qualifying": {
                "target_session": "SQ",
                "predicted_order": [
                    {"position": 1, "driver": "VER", "team": "Red Bull", "confidence": 0.8}
                ],
                "eligible_at_save": True,
            },
            "grand_prix_race": {
                "target_session": "R",
                "predicted_order": [
                    {"position": 1, "driver": "VER", "team": "Red Bull", "confidence": 0.8}
                ],
                "eligible_at_save": True,
            },
        },
    )

    logger.update_actuals(
        year=2026,
        race_name="Chinese Grand Prix",
        session_name="FP1",
        target_actual_results={
            "sprint_qualifying": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
            "grand_prix_race": [{"position": 1, "driver": "VER", "team": "Red Bull"}],
        },
    )

    prediction = logger.load_prediction(2026, "Chinese Grand Prix", "FP1")

    assert prediction is not None
    assert set(prediction["targets"]) == {"sprint_qualifying", "grand_prix_race"}
    assert prediction["actuals"]["targets"]["sprint_qualifying"][0]["driver"] == "VER"
    assert prediction["actuals"]["targets"]["grand_prix_race"][0]["driver"] == "VER"


def test_save_prediction_attaches_background_shadow_challenger(
    temp_predictions_dir,
    sample_quali_prediction,
    sample_race_prediction,
):
    """Future saves should carry target-specific challenger rows for offline scoring."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)
    target_payload = {
        "main_qualifying": {
            "target_session": "Q",
            "predicted_order": [
                {"position": 1, "driver": "BBB", "team": "B"},
                {"position": 2, "driver": "AAA", "team": "A"},
            ],
            "eligible_at_save": True,
        }
    }

    logger.save_prediction(
        year=2026,
        race_name="Australian Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
        target_predictions=target_payload,
    )
    logger.update_actuals(
        year=2026,
        race_name="Australian Grand Prix",
        session_name="FP1",
        target_actual_results={
            "main_qualifying": [
                {"position": 1, "driver": "AAA", "team": "A"},
                {"position": 2, "driver": "BBB", "team": "B"},
            ]
        },
    )
    logger.save_prediction(
        year=2026,
        race_name="Chinese Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
        target_predictions=target_payload,
    )

    prediction = logger.load_prediction(2026, "Chinese Grand Prix", "FP1")
    challenger = prediction["shadow_challengers"]["main_qualifying"]

    assert challenger["status"] == "active"
    assert challenger["challenger_name"] == "main_qualifying_form_blend_v1"
    assert [row["driver"] for row in challenger["predicted_order"]] == ["AAA", "BBB"]


def test_update_actuals_triggers_systematic_learning(
    temp_predictions_dir, sample_quali_prediction, sample_race_prediction
):
    """Updating actuals should persist adaptive learning signals."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP3",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    actual_quali = [
        {"driver": "Norris", "team": "McLaren"},
        {"driver": "Verstappen", "team": "Red Bull"},
        {"driver": "Leclerc", "team": "Ferrari"},
    ]
    actual_race = [
        {"driver": "Leclerc", "team": "Ferrari"},
        {"driver": "Norris", "team": "McLaren"},
        {"driver": "Verstappen", "team": "Red Bull"},
    ]

    assert (
        logger.update_actuals(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_name="FP3",
            qualifying_results=actual_quali,
            race_results=actual_race,
        )
        is True
    )

    learning_file = Path(temp_predictions_dir).parent / "learning_state.json"
    assert learning_file.exists()

    payload = json.loads(learning_file.read_text())
    adaptive = payload.get("adaptive_calibration", {})
    driver_errors = adaptive.get("driver_position_error", {})

    assert "qualifying" in driver_errors
    assert "race" in driver_errors
    assert driver_errors["qualifying"]  # Non-empty after learning update
    assert driver_errors["race"]  # Non-empty after learning update


def test_has_prediction_for_session(
    temp_predictions_dir, sample_quali_prediction, sample_race_prediction
):
    """Test checking if prediction exists for session."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # Save prediction for FP1
    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    # FP1 prediction should exist
    assert logger.has_prediction_for_session(2026, "Bahrain Grand Prix", "FP1") is True
    # But not FP2
    assert logger.has_prediction_for_session(2026, "Bahrain Grand Prix", "FP2") is False


def test_get_all_predictions(temp_predictions_dir, sample_quali_prediction, sample_race_prediction):
    """Test getting all predictions for a year."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # Save multiple predictions
    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP2",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    logger.save_prediction(
        year=2026,
        race_name="Saudi Arabian Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    # Verify each can be loaded back
    pred1 = logger.load_prediction(2026, "Bahrain Grand Prix", "FP1")
    pred2 = logger.load_prediction(2026, "Bahrain Grand Prix", "FP2")
    pred3 = logger.load_prediction(2026, "Saudi Arabian Grand Prix", "FP1")

    assert pred1 is not None
    assert pred2 is not None
    assert pred3 is not None

    assert pred1["metadata"]["race_name"] == "Bahrain Grand Prix"
    assert pred1["metadata"]["session_name"] == "FP1"
    assert pred2["metadata"]["session_name"] == "FP2"
    assert pred3["metadata"]["race_name"] == "Saudi Arabian Grand Prix"


def test_get_all_predictions_merges_artifact_and_file_backends(
    temp_predictions_dir,
    sample_quali_prediction,
    sample_race_prediction,
):
    """Mixed backend history should surface every unique saved checkpoint."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)
    list_calls: list[dict[str, object]] = []

    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    def _list_artifacts(artifact_type: str, key_prefix=None, limit: int = 100):
        list_calls.append(
            {
                "artifact_type": artifact_type,
                "key_prefix": key_prefix,
                "limit": limit,
            }
        )
        return [
            {
                "data": {
                    "metadata": {
                        "year": 2026,
                        "race_name": "Australian Grand Prix",
                        "session_name": "FP2",
                        "predicted_at": "2026-03-16T12:00:00+00:00",
                        "weather": "dry",
                    },
                    "qualifying": {
                        "predicted_grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]
                    },
                    "race": {
                        "predicted_results": [{"position": 1, "driver": "NOR", "team": "McLaren"}]
                    },
                    "targets": {},
                    "actuals": {"qualifying": None, "race": None, "targets": {}},
                }
            }
        ]

    logger.artifact_store.list_artifacts = _list_artifacts

    predictions = logger.get_all_predictions(2026)

    assert list_calls == [
        {
            "artifact_type": "prediction",
            "key_prefix": "2026::",
            "limit": 4096,
        }
    ]
    assert len(predictions) == 2
    assert {
        (
            prediction["metadata"]["race_name"],
            prediction["metadata"]["session_name"],
        )
        for prediction in predictions
    } == {
        ("Australian Grand Prix", "FP2"),
        ("Bahrain Grand Prix", "FP1"),
    }


def test_get_predictions_for_race_limits_artifact_listing(temp_predictions_dir):
    """Race-scoped history should use a DB prefix and filtered file fallback."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)
    list_calls: list[dict[str, object]] = []
    file_calls: list[tuple[int, str | None]] = []
    miami_payload = {
        "metadata": {
            "year": 2026,
            "race_name": "Miami Grand Prix",
            "session_name": "SQ",
            "predicted_at": "2026-05-02T12:00:00+00:00",
            "weather": "dry",
        },
        "qualifying": {"predicted_grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "race": {"predicted_results": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "targets": {},
        "actuals": {"qualifying": None, "race": None, "targets": {}},
    }

    def _list_artifacts(artifact_type: str, key_prefix=None, limit: int = 100):
        list_calls.append(
            {
                "artifact_type": artifact_type,
                "key_prefix": key_prefix,
                "limit": limit,
            }
        )
        return [{"data": miami_payload}]

    def _load_files(year: int, *, race_name: str | None = None) -> list[dict]:
        file_calls.append((year, race_name))
        return []

    logger.artifact_store.list_artifacts = _list_artifacts
    logger._load_predictions_from_files = _load_files  # type: ignore[method-assign]

    predictions = logger.get_predictions_for_race(2026, "Miami Grand Prix")

    assert predictions == [miami_payload]
    assert list_calls == [
        {
            "artifact_type": "prediction",
            "key_prefix": "2026::Miami Grand Prix::",
            "limit": 256,
        }
    ]
    assert file_calls == [(2026, "Miami Grand Prix")]


def test_get_predictions_for_race_filters_file_fallback(
    temp_predictions_dir,
    sample_quali_prediction,
    sample_race_prediction,
):
    """Race-scoped history should still work when only local files are available."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)
    logger.save_prediction(
        year=2026,
        race_name="Miami Grand Prix",
        session_name="SQ",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )
    logger.save_prediction(
        year=2026,
        race_name="Monaco Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )
    logger.artifact_store.list_artifacts = lambda *_args, **_kwargs: []

    predictions = logger.get_predictions_for_race(2026, "Miami Grand Prix")

    assert len(predictions) == 1
    assert predictions[0]["metadata"]["race_name"] == "Miami Grand Prix"
    assert predictions[0]["metadata"]["session_name"] == "SQ"


def test_get_all_predictions_does_not_reload_files_when_listing_is_file_backed(
    temp_predictions_dir,
    sample_quali_prediction,
    sample_race_prediction,
):
    """File-backed history listing should not walk the same prediction files twice."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    logger.save_prediction(
        year=2026,
        race_name="Bahrain Grand Prix",
        session_name="FP1",
        qualifying_prediction=sample_quali_prediction,
        race_prediction=sample_race_prediction,
        weather="dry",
    )

    def _unexpected_file_reload(year: int) -> list[dict]:
        del year
        pytest.fail("file-backed artifact listings should not trigger a second file scan")

    logger._load_predictions_from_files = _unexpected_file_reload  # type: ignore[method-assign]

    predictions = logger.get_all_predictions(2026)

    assert len(predictions) == 1
    assert predictions[0]["metadata"]["race_name"] == "Bahrain Grand Prix"
    assert predictions[0]["metadata"]["session_name"] == "FP1"


def test_get_all_predictions_deduplicates_checkpoint_identity(temp_predictions_dir):
    """Duplicate stored rows for one checkpoint should collapse to the newest payload."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)
    older_payload = {
        "metadata": {
            "year": 2026,
            "race_name": "Chinese Grand Prix",
            "session_name": "FP1",
            "predicted_at": "2026-03-20T12:00:00+00:00",
            "weather": "dry",
        },
        "qualifying": {"predicted_grid": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "race": {"predicted_results": [{"position": 1, "driver": "VER", "team": "Red Bull"}]},
        "targets": {},
        "actuals": {"qualifying": None, "race": None, "targets": {}},
    }
    newer_payload = {
        "metadata": {
            "year": 2026,
            "race_name": " Chinese  Grand Prix ",
            "session_name": "fp1",
            "predicted_at": "2026-03-20T12:05:00+00:00",
            "weather": "dry",
        },
        "qualifying": {"predicted_grid": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "race": {"predicted_results": [{"position": 1, "driver": "NOR", "team": "McLaren"}]},
        "targets": {},
        "actuals": {"qualifying": None, "race": None, "targets": {}},
    }
    logger.artifact_store.list_artifacts = lambda *_args, **_kwargs: [
        {"data": older_payload},
        {"data": newer_payload},
    ]

    predictions = logger.get_all_predictions(2026)

    assert len(predictions) == 1
    assert predictions[0]["metadata"]["session_name"] == "fp1"
    assert predictions[0]["race"]["predicted_results"][0]["driver"] == "NOR"


def test_save_prediction_empty_validation(temp_predictions_dir):
    """Test that empty predictions raise ValueError."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    with pytest.raises(ValueError, match="cannot be empty"):
        logger.save_prediction(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_name="FP1",
            qualifying_prediction=[],
            race_prediction=[{"driver": "VER", "team": "Red Bull"}],
            weather="dry",
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        logger.save_prediction(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_name="FP1",
            qualifying_prediction=[{"driver": "VER", "team": "Red Bull"}],
            race_prediction=[],
            weather="dry",
        )


def test_save_prediction_missing_fields(temp_predictions_dir):
    """Test that predictions with missing fields raise ValueError."""
    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    with pytest.raises(ValueError, match="missing 'driver' or 'team'"):
        logger.save_prediction(
            year=2026,
            race_name="Bahrain Grand Prix",
            session_name="FP1",
            qualifying_prediction=[{"driver": "VER"}],  # Missing team
            race_prediction=[{"driver": "VER", "team": "Red Bull"}],
            weather="dry",
        )


def test_load_prediction_invalid_schema(temp_predictions_dir):
    """Test loading a prediction with invalid schema."""

    logger = PredictionLogger(predictions_dir=temp_predictions_dir)

    # Create invalid prediction file in the location where ArtifactStore writes
    # ArtifactStore writes to: temp_predictions_dir.parent / "predictions" / ...
    parent_dir = Path(temp_predictions_dir).parent
    predictions_root = parent_dir / "predictions"
    year_dir = predictions_root / "2026"
    race_dir = year_dir / "bahrain_grand_prix"
    race_dir.mkdir(parents=True, exist_ok=True)

    invalid_data = {"metadata": {}, "wrong_key": []}  # Missing required keys

    with open(race_dir / "bahrain_grand_prix_fp1.json", "w") as f:
        json.dump(invalid_data, f)

    # Load should return None for invalid schema
    prediction = logger.load_prediction(2026, "Bahrain Grand Prix", "FP1")
    assert prediction is None

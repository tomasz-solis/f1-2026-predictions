"""Tests for dashboard cache/bootstrap helpers."""

from src.dashboard import cache


def test_enable_fastf1_cache_creates_dir_and_enables_cache(patcher, tmp_path):
    cache_dir = tmp_path / "fastf1_cache"
    patcher.setattr(cache, "_FASTF1_CACHE_DIR", cache_dir)

    seen_paths: list[str] = []
    patcher.setattr(cache.fastf1.Cache, "enable_cache", lambda path: seen_paths.append(path))

    cache.enable_fastf1_cache()

    assert cache_dir.exists()
    assert seen_paths == [str(cache_dir)]


def test_enable_fastf1_cache_swallows_cache_errors(patcher, tmp_path):
    patcher.setattr(cache, "_FASTF1_CACHE_DIR", tmp_path / "fastf1_cache")
    patcher.setattr(
        cache.fastf1.Cache,
        "enable_cache",
        lambda _path: (_ for _ in ()).throw(RuntimeError("cache unavailable")),
    )

    # Should not raise.
    cache.enable_fastf1_cache()


def test_get_file_timestamps_reports_existing_and_missing_files(patcher):
    class _FakePath:
        def __init__(self, raw_path: str, exists: bool):
            self._raw_path = raw_path
            self._exists = exists

        def exists(self) -> bool:
            return self._exists

        def __fspath__(self) -> str:
            return self._raw_path

    existing_files = {"data/2025_pirelli_info.json", "config/default.yaml"}
    patcher.setattr(
        cache,
        "Path",
        lambda file_path: _FakePath(file_path, file_path in existing_files),
    )
    patcher.setattr(cache.os.path, "getmtime", lambda _path: 123.456)

    timestamps = cache._get_file_timestamps()

    assert timestamps["data/2025_pirelli_info.json"] == (123, "123.456")
    assert timestamps["config/default.yaml"] == (123, "123.456")
    assert timestamps["data/2026_pirelli_info.json"] == (0, "")
    assert timestamps["src/predictors/baseline_2026.py"] == (0, "")


def test_get_artifact_versions_combines_store_and_file_timestamps(patcher):
    class _Store:
        def __init__(self, data_root: str):
            assert data_root == "data"

        def load_artifact(self, artifact_type: str, artifact_key: str):
            if artifact_type == "car_characteristics":
                return {"version": 7, "last_updated": "2026-02-01T00:00:00"}
            if artifact_type == "driver_characteristics":
                return {"version": 3, "updated_at": "2026-02-02T00:00:00"}
            raise RuntimeError("db error")

    patcher.setattr(cache, "ArtifactStore", _Store)
    patcher.setattr(
        cache,
        "_get_file_timestamps",
        lambda year=2026: {"config/default.yaml": (9, "9.1")},
    )

    versions = cache.get_artifact_versions()

    assert versions["car_characteristics::2026::car_characteristics"] == (
        7,
        "2026-02-01T00:00:00",
    )
    assert versions["driver_characteristics::2026::driver_characteristics"] == (
        3,
        "2026-02-02T00:00:00",
    )
    assert versions["track_characteristics::2026::track_characteristics"] == (0, "")
    assert versions["config/default.yaml"] == (9, "9.1")


def test_get_artifact_versions_supports_non_default_year(patcher):
    class _Store:
        def __init__(self, data_root: str):
            assert data_root == "data"

        def load_artifact(self, artifact_type: str, artifact_key: str):
            if artifact_key == "2027::car_characteristics":
                return {"version": 11, "last_updated": "2027-02-01T00:00:00"}
            if artifact_key == "2027::driver_characteristics":
                return {"version": 5, "updated_at": "2027-02-02T00:00:00"}
            if artifact_key == "2027::track_characteristics":
                return {"version": 2, "updated_at": "2027-02-03T00:00:00"}
            return None

    patcher.setattr(cache, "ArtifactStore", _Store)
    patcher.setattr(cache, "_get_file_timestamps", lambda year=2027: {})

    versions = cache.get_artifact_versions(year=2027)

    assert versions["car_characteristics::2027::car_characteristics"] == (
        11,
        "2027-02-01T00:00:00",
    )
    assert versions["driver_characteristics::2027::driver_characteristics"] == (
        5,
        "2027-02-02T00:00:00",
    )
    assert versions["track_characteristics::2027::track_characteristics"] == (
        2,
        "2027-02-03T00:00:00",
    )


def test_get_predictor_passes_year_to_predictor_bootstrap(patcher):
    import src.predictors.baseline_2026 as baseline_module
    from src.utils.config_loader import Config

    calls: list[int] = []

    class _Predictor:
        def __init__(self, season_year: int = 2026):
            calls.append(season_year)

    patcher.setattr(baseline_module, "Baseline2026Predictor", _Predictor)
    patcher.setattr(Config, "reload", lambda self: None)

    cache.get_predictor.clear()
    cache.get_predictor({"artifact": (1, "ts")}, year=2027)
    cache.get_predictor.clear()

    assert calls == [2027]

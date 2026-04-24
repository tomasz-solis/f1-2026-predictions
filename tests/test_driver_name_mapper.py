"""
Tests for Driver Name Mapper
"""

from src.utils.driver_name_mapper import DriverNameMapper


def test_normalize_abbreviation():
    """Test normalizing already-abbreviated names."""
    assert DriverNameMapper.normalize_driver_name("VER") == "VER"
    assert DriverNameMapper.normalize_driver_name("ver") == "VER"
    assert DriverNameMapper.normalize_driver_name("NOR") == "NOR"
    assert DriverNameMapper.normalize_driver_name("MSC") == "MSC"
    assert DriverNameMapper.normalize_driver_name("MAG") == "MAG"
    assert DriverNameMapper.normalize_driver_name("LAT") == "LAT"
    assert DriverNameMapper.normalize_driver_name("ZHO") == "ZHO"
    assert DriverNameMapper.normalize_driver_name("RIC") == "RIC"


def test_normalize_full_name():
    """Test normalizing full names."""
    assert DriverNameMapper.normalize_driver_name("Verstappen") == "VER"
    assert DriverNameMapper.normalize_driver_name("Norris") == "NOR"
    assert DriverNameMapper.normalize_driver_name("Hamilton") == "HAM"


def test_normalize_full_name_with_first():
    """Test normalizing full names with first name."""
    assert DriverNameMapper.normalize_driver_name("max verstappen") == "VER"
    assert DriverNameMapper.normalize_driver_name("lando norris") == "NOR"
    assert DriverNameMapper.normalize_driver_name("lewis hamilton") == "HAM"
    assert DriverNameMapper.normalize_driver_name("mick schumacher") == "MSC"
    assert DriverNameMapper.normalize_driver_name("kevin magnussen") == "MAG"
    assert DriverNameMapper.normalize_driver_name("nicholas latifi") == "LAT"
    assert DriverNameMapper.normalize_driver_name("zhou guanyu") == "ZHO"
    assert DriverNameMapper.normalize_driver_name("daniel ricciardo") == "RIC"


def test_normalize_case_insensitive():
    """Test case-insensitive normalization."""
    assert DriverNameMapper.normalize_driver_name("VERSTAPPEN") == "VER"
    assert DriverNameMapper.normalize_driver_name("verstappen") == "VER"
    assert DriverNameMapper.normalize_driver_name("VeRsTaPpEn") == "VER"


def test_get_full_name():
    """Test getting full name from abbreviation."""
    assert DriverNameMapper.get_full_name("VER") == "Verstappen"
    assert DriverNameMapper.get_full_name("NOR") == "Norris"
    assert DriverNameMapper.get_full_name("HAM") == "Hamilton"


def test_normalize_result_list():
    """Test normalizing a list of results."""
    results = [
        {"position": 1, "driver": "Verstappen", "team": "Red Bull"},
        {"position": 2, "driver": "NOR", "team": "McLaren"},
        {"position": 3, "driver": "lewis hamilton", "team": "Ferrari"},
    ]

    normalized = DriverNameMapper.normalize_result_list(results)

    assert normalized[0]["driver"] == "VER"
    assert normalized[1]["driver"] == "NOR"
    assert normalized[2]["driver"] == "HAM"


def test_normalize_unknown_driver():
    """Test normalizing an unknown driver name."""
    unknown = DriverNameMapper.normalize_driver_name("Unknown Driver")
    assert unknown == "Unknown Driver"  # Returns original if not found


def test_unknown_uppercase_driver_code_does_not_warn(caplog):
    """Valid-looking historical or future driver codes should pass through quietly."""
    assert DriverNameMapper.normalize_driver_name("XYZ") == "XYZ"
    assert "Could not normalize driver name" not in caplog.text


def test_add_driver():
    """Test adding a new driver mapping."""
    DriverNameMapper.add_driver("LAW", "Lawson")

    assert DriverNameMapper.normalize_driver_name("LAW") == "LAW"
    assert DriverNameMapper.normalize_driver_name("Lawson") == "LAW"
    assert DriverNameMapper.get_full_name("LAW") == "Lawson"


def test_2026_grid_coverage():
    """Test that all 2026 drivers are covered."""
    drivers_2026 = [
        "VER",
        "PER",
        "NOR",
        "PIA",
        "LEC",
        "HAM",
        "RUS",
        "ANT",
        "ALO",
        "STR",
        "GAS",
        "DOO",
        "ALB",
        "SAI",
        "BEA",
        "OCO",
        "HUL",
        "BOR",
        "TSU",
        "HAD",
    ]

    for abbr in drivers_2026:
        assert abbr in DriverNameMapper.DRIVER_MAP
        full_name = DriverNameMapper.get_full_name(abbr)
        assert DriverNameMapper.normalize_driver_name(full_name) == abbr

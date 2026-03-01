"""
Constants for F1 2026 Prediction System

All magic numbers extracted to one place.
"""

# DNF Risk Constants
DNF_EXPERIENCE_MODIFIERS = {
    "rookie": 0.05,  # +5% crash risk for rookies
    "developing": 0.02,  # +2% for developing drivers
    "established": 0.00,  # Baseline
    "veteran": -0.01,  # -1% for experienced drivers
}

DNF_RATE_HISTORICAL_CAP = 0.20  # Max 20% base DNF rate from historical data
DNF_RATE_FINAL_CAP = 0.35  # Absolute maximum DNF probability

# Qualifying Constants
QUALI_NOISE_STD_SPRINT = 0.025  # Sprint weekends have less practice
QUALI_NOISE_STD_NORMAL = 0.02  # Normal weekends
QUALI_TEAM_WEIGHT = 0.7  # 70% team strength
QUALI_SKILL_WEIGHT = 0.3  # 30% driver skill
QUALI_CONFIDENCE_BASE = 60  # Base confidence percentage
QUALI_CONFIDENCE_MIN = 40  # Minimum confidence

# Race Constants
RACE_BASE_CHAOS_DRY = 0.35  # Dry race unpredictability
RACE_BASE_CHAOS_WET = 0.45  # Wet race unpredictability

# Safety Car Probabilities
SC_BASE_PROB_DRY = 0.45  # 45% baseline for dry races
SC_BASE_PROB_WET = 0.70  # 70% baseline for wet races
SC_TRACK_MODIFIER = 0.25  # Track-specific SC multiplier

# Grid Position Effects
GRID_WEIGHT_MIN = 0.15  # Minimum grid position weight
GRID_WEIGHT_MAX = 0.35  # Maximum grid position weight

# Default Fallbacks
DEFAULT_TEAM_PERFORMANCE = 0.50  # Neutral team if missing
DEFAULT_DNF_RATE = 0.10  # Default 10% DNF if missing
DEFAULT_TEAM_UNCERTAINTY = 0.30  # Default uncertainty

# Telemetry Feature Extraction
CORNER_SPEED_THRESHOLD = 250  # km/h - separates corners from straights
CORNER_SPEED_THRESHOLDS = {
    "slow": (0, 100),  # Monaco hairpin
    "medium": (100, 200),  # Typical corners
    "high": (200, 250),  # Spa Pouhon, Copse
}

# F1 Physical Constants
FUEL_LOAD_MAX_KG = 110  # Maximum fuel load
FUEL_EFFECT_PER_LAP = 0.035  # ~0.035s/lap per 10kg fuel (approximate)
TYPICAL_PIT_STOP_LOSS = 22.0  # Seconds lost in pits (track average)

# Experience Tier Thresholds (years in F1)
EXPERIENCE_VETERAN_YEARS = 10
EXPERIENCE_ESTABLISHED_YEARS = 5
EXPERIENCE_DEVELOPING_YEARS = 2

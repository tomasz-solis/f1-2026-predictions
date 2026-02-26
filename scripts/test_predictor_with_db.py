#!/usr/bin/env python3
"""
Test script to verify predictor works with ArtifactStore.

Usage:
    # Test with DB storage
    export SUPABASE_URL=https://xxxxx.supabase.co
    export SUPABASE_KEY=eyJhbGc...
    export USE_DB_STORAGE=fallback
    python scripts/test_predictor_with_db.py

    # Test with file storage (old way)
    export USE_DB_STORAGE=file_only
    python scripts/test_predictor_with_db.py
"""

import sys
from pathlib import Path

# Add project root to path (so 'src' imports work)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.persistence.config import USE_DB_STORAGE
from src.predictors.baseline_2026 import Baseline2026Predictor


def main():
    print("=" * 70)
    print("Predictor Test with ArtifactStore")
    print("=" * 70)
    print(f"\nStorage mode: {USE_DB_STORAGE}")

    # Initialize predictor
    print("\n1. Initializing predictor...")
    try:
        predictor = Baseline2026Predictor(data_dir=Path("data/processed"))
        print("   [OK] Predictor initialized successfully")
    except Exception as e:
        print(f"   [ERROR] Failed to initialize predictor: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Check loaded data
    print("\n2. Checking loaded data...")
    print(f"   Teams loaded: {len(predictor.teams)}")
    print(f"   Drivers loaded: {len(predictor.drivers)}")
    print(f"   Tracks loaded: {len(predictor.tracks)}")
    print(f"   Races completed: {predictor.races_completed}")
    print(f"   Year: {predictor.year}")

    if len(predictor.teams) == 0:
        print("   [ERROR] No teams loaded!")
        return 1

    if len(predictor.drivers) == 0:
        print("   [ERROR] No drivers loaded!")
        return 1

    print("\n   [OK] All data loaded correctly")

    # Test a simple prediction (just grid anchoring, no actual race sim)
    print("\n3. Testing basic functionality...")
    try:
        # Get a team name
        team_name = list(predictor.teams.keys())[0]
        team_strength = predictor.get_blended_team_strength(team_name, "Bahrain Grand Prix", 1)
        print(f"   Team: {team_name}")
        print(f"   Blended strength: {team_strength:.4f}")
        print("   [OK] Basic functionality works")
    except Exception as e:
        print(f"   [ERROR] Failed basic test: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("[OK] All tests passed! Predictor works with ArtifactStore.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

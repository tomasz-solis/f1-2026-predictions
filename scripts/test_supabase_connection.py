#!/usr/bin/env python3
"""
Test script to verify Supabase connection and artifact store.

Usage:
    export SUPABASE_URL=https://xxxxx.supabase.co
    export SUPABASE_KEY=eyJhbGc...
    export USE_DB_STORAGE=db_only
    python scripts/test_supabase_connection.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persistence.artifact_store import ArtifactStore
from persistence.db import check_connection


def main():
    print("=" * 60)
    print("Supabase Connection Test")
    print("=" * 60)

    # Test 1: Health check
    print("\n1. Testing Supabase connection...")
    try:
        message = check_connection()
        print(f"   [OK] {message}")
    except Exception as e:
        print(f"   [ERROR] {e}")
        return 1

    # Test 2: Initialize artifact store
    print("\n2. Initializing ArtifactStore...")
    try:
        store = ArtifactStore()
        print(f"   [OK] Artifact store initialized (mode: {store.storage_mode})")
    except Exception as e:
        print(f"   [ERROR] Failed to initialize: {e}")
        return 1

    # Test 3: Save test artifact
    print("\n3. Testing artifact save...")
    try:
        test_data = {
            "test": True,
            "message": "Hello from F1 predictions!",
            "timestamp": "2026-02-16T12:00:00Z",
        }
        result = store.save_artifact(
            artifact_type="test",
            artifact_key="connection_test",
            data=test_data,
            version=1,
        )
        print(f"   [OK] Test artifact saved: {result.get('id', 'N/A')}")
    except Exception as e:
        print(f"   [ERROR] Failed to save: {e}")
        return 1

    # Test 4: Load test artifact
    print("\n4. Testing artifact load...")
    try:
        loaded = store.load_artifact(artifact_type="test", artifact_key="connection_test")
        if loaded and loaded.get("test"):
            print(f"   [OK] Test artifact loaded: {loaded.get('message')}")
        else:
            print("   [ERROR] Failed to load artifact")
            return 1
    except Exception as e:
        print(f"   [ERROR] Failed to load: {e}")
        return 1

    # Test 5: List artifacts
    print("\n5. Testing artifact listing...")
    try:
        artifacts = store.list_artifacts(artifact_type="test", limit=5)
        print(f"   [OK] Found {len(artifacts)} test artifact(s)")
    except Exception as e:
        print(f"   [ERROR] Failed to list: {e}")
        return 1

    print("\n" + "=" * 60)
    print("[OK] All tests passed! Supabase is ready to use.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Session selector for F1 predictions

Handles session selection based on:
- Prediction timing (after which session are we predicting?)
- Weekend type (normal vs sprint)
- Track characteristics (overtaking difficulty)
- Available data

Progressive refinement: Predictions improve after each session.
"""

import numpy as np


def calculate_overtaking_difficulty(track_chars: dict) -> float:
    """
    Calculate overtaking difficulty (0 = very easy, 1 = nearly impossible).
    """
    # Street circuits are hardest
    street_factor = track_chars.get("is_street_circuit_z", 0) * 0.4

    # High corner density = less overtaking
    corner_density_factor = track_chars.get("corner_density_z", 0) * 0.3

    # Long straights = more overtaking (negative contribution)
    straight_factor = -track_chars.get("full_throttle_pct_z", 0) * 0.3

    difficulty = street_factor + corner_density_factor + straight_factor

    return float(np.clip(difficulty, 0, 1))


def _normalize_weekend_type(weekend_type: str) -> str:
    """Normalize weekend type labels for compatibility."""
    normalized = (weekend_type or "").strip().lower()
    if normalized in ("normal", "conventional"):
        return "normal"
    if normalized == "sprint":
        return "sprint"
    raise ValueError(f"Unsupported weekend_type '{weekend_type}'. Use 'conventional' or 'sprint'.")


def get_prediction_context(current_session: str, weekend_type: str = "conventional") -> dict:
    """
    Get available data and next prediction target for current session.
    """
    contexts = {
        "normal": {
            "pre_fp1": {
                "available": [],
                "next_prediction": "qualifying",
                "confidence_base": 0.3,
                "note": "Baseline only (testing + similar tracks + historical)",
            },
            "post_fp1": {
                "available": ["fp1"],
                "next_prediction": "qualifying",
                "confidence_base": 0.5,
                "note": "Initial data (race and quali setups mixed)",
            },
            "post_fp2": {
                "available": ["fp1", "fp2"],
                "next_prediction": "qualifying",
                "confidence_base": 0.7,
                "note": "Race sim data available (FP2 = long runs)",
            },
            "post_fp3": {
                "available": ["fp1", "fp2", "fp3"],
                "next_prediction": "qualifying",
                "confidence_base": 0.85,
                "note": "Quali sim data (FP3 = low fuel, soft tires)",
            },
            "post_quali": {
                "available": ["fp1", "fp2", "fp3", "quali"],
                "next_prediction": "race",
                "confidence_base": 0.8,
                "note": "Race prediction (use FP2 race sim + quali result)",
            },
        },
        "sprint": {
            "pre_fp1": {
                "available": [],
                "next_prediction": "sprint_qualifying",
                "confidence_base": 0.3,
                "note": "Baseline only",
            },
            "post_fp1": {
                "available": ["fp1"],
                "next_prediction": "sprint_qualifying",
                "confidence_base": 0.4,
                "note": "Only 60 minutes practice!",
            },
            "post_sprint_quali": {
                "available": ["fp1", "sprint_qualifying"],
                "next_prediction": "sprint",
                "confidence_base": 0.6,
                "note": "Sprint Quali shows quali pace",
            },
            "post_sprint": {
                "available": ["fp1", "sprint_qualifying", "sprint"],
                "next_prediction": "main_qualifying",
                "confidence_base": 0.7,
                "note": "Sprint race provides real race data!",
            },
            "post_quali": {
                "available": ["fp1", "sprint_qualifying", "sprint", "main_quali"],
                "next_prediction": "main_race",
                "confidence_base": 0.8,
                "note": "Full dataset available",
            },
        },
    }

    weekend_key = _normalize_weekend_type(weekend_type)
    return contexts[weekend_key].get(current_session, contexts[weekend_key]["pre_fp1"])


def map_session_name_to_key(session_name: str, team_sessions: dict) -> str | None:
    """
    Map generic session name to actual key in team_sessions dict.
    """
    # Try exact match first
    if session_name in team_sessions:
        return session_name

    # Try fuzzy match (ends with the session name)
    session_suffix = f"_{session_name}"
    for key in team_sessions.keys():
        if key.endswith(session_suffix):
            return key

    return None


def select_best_session(
    team_sessions: dict,
    track_chars: dict,
    current_session: str,
    weekend_type: str = "conventional",
    prediction_target: str = "qualifying",
) -> tuple[dict | None, float, str]:
    """
    Select best session data for prediction. Returns (session_data, confidence, reasoning).
    """
    # Get prediction context
    context = get_prediction_context(current_session, weekend_type)
    available = context["available"]
    base_confidence = context["confidence_base"]

    # Calculate overtaking difficulty
    overtaking_diff = calculate_overtaking_difficulty(track_chars)

    # Session priority based on prediction target
    if prediction_target == "qualifying":
        # Quali: Want low fuel, soft tire data
        session_priority = {
            "fp3": 1.0,  # Quali sim
            "sprint_qualifying": 0.9,  # Short but relevant
            "fp2": 0.7,  # Some quali sims
            "sprint": 0.5,  # Race pace, not quali
            "fp1": 0.4,  # Mixed programs
        }
    else:  # race
        if overtaking_diff > 0.7:
            # Monaco-like: Grid position is everything
            session_priority = {
                "main_quali": 1.0,  # Grid position decides race
                "fp3": 0.9,  # Quali sim
                "sprint_qualifying": 0.8,
                "sprint": 0.6,
                "fp2": 0.5,
                "fp1": 0.4,
            }
        else:
            # Monza-like: Race pace matters
            session_priority = {
                "sprint": 1.0,  # Real race data!
                "fp2": 0.9,  # Race sim (high fuel)
                "main_quali": 0.7,  # Setup might differ
                "fp1": 0.5,
                "sprint_qualifying": 0.4,
                "fp3": 0.3,  # Low fuel, not representative
            }

    # Find best available session
    best_session = None
    best_score = 0.0
    best_session_name = None

    for session_name in available:
        # Map to actual key in team_sessions
        actual_key = map_session_name_to_key(session_name, team_sessions)

        if actual_key and actual_key in team_sessions:
            score = session_priority.get(session_name, 0)
            if score > best_score:
                best_score = score
                best_session = team_sessions[actual_key]
                best_session_name = session_name

    # Adjust confidence based on session quality
    confidence = base_confidence * best_score if best_session else base_confidence * 0.5

    # Generate reasoning
    if best_session:
        reasoning = (
            f"Using {best_session_name} (score: {best_score:.2f}, base conf: {base_confidence:.2f})"
        )
    else:
        reasoning = f"No session data available (using baseline, conf: {confidence:.2f})"

    return best_session, confidence, reasoning


def get_prediction_workflow(weekend_type: str = "conventional") -> list[dict]:
    """
    Get complete prediction workflow for a weekend with context at each stage.
    """
    weekend_key = _normalize_weekend_type(weekend_type)
    if weekend_key == "normal":
        return [
            {
                "timing": "Pre-FP1 (Thursday)",
                "data_available": "Baseline only",
                "prediction": "Initial qualifying prediction",
                "confidence": 0.3,
                "note": "Testing + similar tracks + historical",
            },
            {
                "timing": "Post-FP1 (Friday morning)",
                "data_available": "FP1",
                "prediction": "Updated qualifying prediction",
                "confidence": 0.5,
                "note": "First real data, mixed programs",
            },
            {
                "timing": "Post-FP2 (Friday afternoon)",
                "data_available": "FP1 + FP2",
                "prediction": "Updated qualifying prediction",
                "confidence": 0.7,
                "note": "Race sim data from FP2",
            },
            {
                "timing": "Post-FP3 (Saturday morning)",
                "data_available": "FP1 + FP2 + FP3",
                "prediction": "Final qualifying prediction",
                "confidence": 0.85,
                "note": "Quali sim data from FP3",
            },
            {
                "timing": "Post-Qualifying (Saturday afternoon)",
                "data_available": "FP1 + FP2 + FP3 + Quali",
                "prediction": "Race prediction",
                "confidence": 0.8,
                "note": "Grid known, use FP2 race sim",
            },
        ]
    else:  # sprint
        return [
            {
                "timing": "Pre-FP1 (Friday morning)",
                "data_available": "Baseline only",
                "prediction": "Sprint Quali prediction",
                "confidence": 0.3,
                "note": "Testing + similar tracks + historical",
            },
            {
                "timing": "Post-FP1 (Friday afternoon)",
                "data_available": "FP1 (60 min only!)",
                "prediction": "Updated Sprint Quali prediction",
                "confidence": 0.4,
                "note": "Limited data, single session",
            },
            {
                "timing": "Post-Sprint Quali (Friday evening)",
                "data_available": "FP1 + Sprint Quali",
                "prediction": "Sprint race prediction",
                "confidence": 0.6,
                "note": "Sprint Quali shows quali pace",
            },
            {
                "timing": "Post-Sprint Race (Saturday afternoon)",
                "data_available": "FP1 + Sprint Quali + Sprint",
                "prediction": "Main Qualifying prediction",
                "confidence": 0.7,
                "note": "Sprint provides real race data!",
            },
            {
                "timing": "Post-Main Quali (Saturday evening)",
                "data_available": "FP1 + Sprint Quali + Sprint + Main Quali",
                "prediction": "Main Race prediction",
                "confidence": 0.8,
                "note": "Complete dataset available",
            },
        ]

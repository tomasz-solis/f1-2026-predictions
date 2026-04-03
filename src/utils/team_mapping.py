"""Map team names from different sources to a shared set of identifiers."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TEAM_NAME_MAP = {
    # Sauber lineage (Sauber → Alfa Romeo → Kick Sauber → Audi)
    # Audi (2026+ - continuation of Sauber)
    "Audi": "AUDI",
    "Audi F1 Team": "AUDI",
    "Audi F1": "AUDI",
    "Sauber": "AUDI",
    "Alfa Romeo": "AUDI",
    "Kick Sauber": "AUDI",
    "Stake F1 Team Kick Sauber": "AUDI",
    "Kick Sauber F1 Team": "AUDI",
    # AlphaTauri lineage (Toro Rosso → AlphaTauri → RB)
    "AlphaTauri": "RB",
    "Scuderia AlphaTauri": "RB",
    "Visa Cash App RB": "RB",
    "VCARB": "RB",
    "Racing Bulls": "RB",
    "RB F1 Team": "RB",
    "RB": "RB",
    # Red Bull Racing (stable identity)
    "Red Bull Racing": "RED BULL",
    "Oracle Red Bull Racing": "RED BULL",
    "Red Bull": "RED BULL",
    # Mercedes (stable identity)
    "Mercedes": "MERCEDES",
    "Mercedes-AMG Petronas F1 Team": "MERCEDES",
    # Ferrari (stable identity)
    "Ferrari": "FERRARI",
    "Scuderia Ferrari": "FERRARI",
    # Aston Martin lineage (Racing Point → Aston Martin)
    "Racing Point": "ASTON MARTIN",
    "SportPesa Racing Point": "ASTON MARTIN",
    "BWT Racing Point": "ASTON MARTIN",
    "Aston Martin": "ASTON MARTIN",
    "Aston Martin Aramco": "ASTON MARTIN",
    # Alpine lineage (Renault → Alpine)
    "Renault": "ALPINE",
    "Renault F1 Team": "ALPINE",
    "Alpine": "ALPINE",
    "BWT Alpine F1 Team": "ALPINE",
    # McLaren (stable identity)
    "McLaren": "MCLAREN",
    "McLaren F1 Team": "MCLAREN",
    # Haas (stable identity)
    "Haas": "HAAS",
    "Haas F1 Team": "HAAS",
    "MoneyGram Haas F1 Team": "HAAS",
    # Williams (stable identity)
    "Williams": "WILLIAMS",
    "Williams Racing": "WILLIAMS",
    # Cadillac (new team 2026)
    "Cadillac": "CADILLAC",
    "Cadillac F1": "CADILLAC",
    "Cadillac Racing": "CADILLAC",
}

# Team names used in car-characteristics payloads.
CHARACTERISTICS_TEAM_MAP = {
    "RED BULL": "Red Bull Racing",
    "MCLAREN": "McLaren",
    "FERRARI": "Ferrari",
    "MERCEDES": "Mercedes",
    "ASTON MARTIN": "Aston Martin",
    "ALPINE": "Alpine",
    "HAAS": "Haas F1 Team",
    "RB": "RB",
    "WILLIAMS": "Williams",
    "AUDI": "Audi",
    "CADILLAC": "Cadillac F1",
}


def canonicalize_team(name: str) -> str:
    """Map raw team name to canonical identifier, handling variations and sponsor changes."""
    if name is None:
        return None

    return TEAM_NAME_MAP.get(name, str(name).upper())


def normalize_team_column(df: pd.DataFrame | pd.Series, col: str = "team") -> pd.DataFrame:
    """Normalize team names in DataFrame using canonical mapping, handling missing or aliased names."""
    # Handle Series input (convert to DataFrame)
    if isinstance(df, pd.Series):
        logger.warning(
            "normalize_team_column() called on Series instead of DataFrame. "
            "Auto-converting to DataFrame."
        )
        df = df.to_frame(name=col)

    # Handle missing column (return unchanged)
    if col not in df.columns:
        return df

    # Apply canonicalization
    raw_names = df[col].astype(str)
    canonical_names = raw_names.apply(canonicalize_team)

    # Detect and log unknown teams (those using fallback uppercase)
    unknown_mask = ~raw_names.isin(TEAM_NAME_MAP.keys()) & (
        canonical_names == raw_names.str.upper()
    )

    if unknown_mask.any():
        unknown_teams = sorted(raw_names[unknown_mask].unique())
        logger.warning(
            "Unknown team names encountered (using uppercase fallback): %s",
            unknown_teams,
        )

    df[col] = canonical_names

    return df


def _normalize_name(value: str) -> str:
    """Normalize names for case-insensitive matching."""
    return "".join(char for char in str(value).lower() if char.isalnum())


def _resolve_known_team_by_canonical_id(canonical_id: str, known_teams: set[str]) -> str | None:
    """Resolve a canonical team ID to an existing known-team label."""
    if not canonical_id or not known_teams:
        return None

    candidates = [
        team_name for team_name in known_teams if canonicalize_team(team_name) == canonical_id
    ]
    if not candidates:
        return None

    preferred = CHARACTERISTICS_TEAM_MAP.get(canonical_id)
    if preferred in candidates:
        return preferred

    return sorted(candidates, key=lambda value: (len(value), value))[0]


def map_team_to_characteristics(name: str, known_teams: set[str] | None = None) -> str | None:
    """
    Map a team label from FastF1/other sources to characteristics team naming.

    If `known_teams` is provided, returns only values present in that set; otherwise returns
    the best-effort mapped name (or the input text when no mapping is known).
    """
    if name is None:
        return None

    team_text = str(name).strip()
    if not team_text:
        return None

    if known_teams and team_text in known_teams:
        return team_text

    # Already in characteristics naming.
    if team_text in CHARACTERISTICS_TEAM_MAP.values():
        if known_teams is None or team_text in known_teams:
            return team_text

    canonical_id = canonicalize_team(team_text)
    mapped = CHARACTERISTICS_TEAM_MAP.get(canonical_id)
    if mapped and (known_teams is None or mapped in known_teams):
        return mapped

    # Fuzzy fallback when an explicit known-team set is provided.
    if known_teams:
        # Prefer a caller's known-team labels when they still use an older name.
        known_canonical_match = _resolve_known_team_by_canonical_id(canonical_id, known_teams)
        if known_canonical_match:
            return known_canonical_match

        normalized_input = _normalize_name(team_text)
        normalized_known = {_normalize_name(team): team for team in known_teams}

        if normalized_input in normalized_known:
            return normalized_known[normalized_input]

        normalized_canonical = _normalize_name(canonical_id)
        if normalized_canonical in normalized_known:
            return normalized_known[normalized_canonical]

        if mapped:
            normalized_mapped = _normalize_name(mapped)
            if normalized_mapped in normalized_known:
                return normalized_known[normalized_mapped]

        return None

    return mapped if mapped else team_text

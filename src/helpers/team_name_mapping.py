"""

Originally from: https://github.com/tomasz-solis/formula1
Adapted for 2026 Bayesian prediction system.

==============================================================================

Team Name Canonicalization for F1 Data Pipeline

Handles F1 team name variations across different data sources and years.
Maps all variations to canonical team identifiers for consistent analysis.

Key Features:
- Handles team rebranding (AlphaTauri → RB, Alfa Romeo → Sauber)
- Maps sponsor name changes to core team identity
- Provides safe fallback for unknown teams
- Logs unknown team names for manual review

Example:
    >>> from helpers.team_name_mapping import canonicalize_team, normalize_team_column
    >>> canonicalize_team('Visa Cash App RB')
    'RB'
    >>> df = normalize_team_column(df, col='team')

Author: Tomasz Solis
Date: November 2025
"""

import pandas as pd
import logging
from typing import Union

logger = logging.getLogger(__name__)


# =============================================================================
# TEAM NAME MAPPING
# =============================================================================

TEAM_NAME_MAP = {
    # Sauber lineage (Sauber → Alfa Romeo → Kick Sauber → Audi)
    # Audi (2026+ - continuation of Sauber)
    'Audi': 'AUDI',
    'Audi F1 Team': 'AUDI',
    'Audi F1': 'AUDI',
    'Sauber': 'AUDI',
    'Alfa Romeo': 'AUDI',
    'Kick Sauber': 'AUDI',
    'Stake F1 Team Kick Sauber': 'AUDI',
    'Kick Sauber F1 Team': 'AUDI',
    
    # AlphaTauri lineage (Toro Rosso → AlphaTauri → RB)
    'AlphaTauri': 'RB',
    'Scuderia AlphaTauri': 'RB',
    'Visa Cash App RB': 'RB',
    'VCARB': 'RB',
    'Racing Bulls': 'RB',
    'RB F1 Team': 'RB',
    'RB': 'RB',

    # Red Bull Racing (stable identity)
    'Red Bull Racing': 'RED BULL',
    'Oracle Red Bull Racing': 'RED BULL',
    'Red Bull': 'RED BULL',

    # Mercedes (stable identity)
    'Mercedes': 'MERCEDES',
    'Mercedes-AMG Petronas F1 Team': 'MERCEDES',

    # Ferrari (stable identity)
    'Ferrari': 'FERRARI',
    'Scuderia Ferrari': 'FERRARI',

    # Aston Martin lineage (Racing Point → Aston Martin)
    'Racing Point': 'ASTON MARTIN',
    'SportPesa Racing Point': 'ASTON MARTIN',
    'BWT Racing Point': 'ASTON MARTIN',
    'Aston Martin': 'ASTON MARTIN',
    'Aston Martin Aramco': 'ASTON MARTIN',

    # Alpine lineage (Renault → Alpine)
    'Renault': 'ALPINE',
    'Renault F1 Team': 'ALPINE',
    'Alpine': 'ALPINE',
    'BWT Alpine F1 Team': 'ALPINE',

    # McLaren (stable identity)
    'McLaren': 'MCLAREN',
    'McLaren F1 Team': 'MCLAREN',

    # Haas (stable identity)
    'Haas': 'HAAS',
    'Haas F1 Team': 'HAAS',
    'MoneyGram Haas F1 Team': 'HAAS',

    # Williams (stable identity)
    'Williams': 'WILLIAMS',
    'Williams Racing': 'WILLIAMS',

    # Cadillac (new team 2026)
    'Cadillac': 'CADILLAC',
    'Cadillac F1': 'CADILLAC',
    'Cadillac Racing': 'CADILLAC',
}


# =============================================================================
# CANONICALIZATION FUNCTIONS
# =============================================================================

def canonicalize_team(name: str) -> str:
    """
    Map raw team name to canonical identifier.
    
    Handles team name variations from FastF1 API, historical data,
    and sponsor changes. Falls back to uppercase original if unknown.
    
    Args:
        name: Raw team name string (e.g., "Visa Cash App RB")
        
    Returns:
        Canonical team identifier (e.g., "RB")
        
    Example:
        >>> canonicalize_team('Oracle Red Bull Racing')
        'RED BULL'
        >>> canonicalize_team('Unknown Team')
        'UNKNOWN TEAM'
    """
    if name is None:
        return None
    
    return TEAM_NAME_MAP.get(name, str(name).upper())


def normalize_team_column(
    df: Union[pd.DataFrame, pd.Series],
    col: str = "team"
) -> pd.DataFrame:
    """
    Normalize team names in DataFrame using canonical mapping.
    
    Applies canonicalize_team() to specified column, logging any
    unknown team names for manual review. Handles edge cases safely.
    
    Safety Features:
    - Auto-converts Series to DataFrame if needed
    - Returns unchanged if column doesn't exist
    - Logs unknown teams for monitoring
    
    Args:
        df: DataFrame or Series containing team names
        col: Name of team column (default: "team")
        
    Returns:
        DataFrame with normalized team names
        
    Raises:
        None - handles all errors gracefully
        
    Example:
        >>> df = normalize_team_column(df, col='team')
        >>> df['team'].unique()
        array(['RED BULL', 'FERRARI', 'MERCEDES', ...])
    """
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
    unknown_mask = (
        ~raw_names.isin(TEAM_NAME_MAP.keys()) & 
        (canonical_names == raw_names.str.upper())
    )
    
    if unknown_mask.any():
        unknown_teams = sorted(raw_names[unknown_mask].unique())
        logger.warning(
            "Unknown team names encountered (using uppercase fallback): %s",
            unknown_teams
        )
    
    df[col] = canonical_names
    
    return df
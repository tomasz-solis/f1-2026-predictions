"""
Performance Extraction - Clean Implementation

Extracts and normalizes team performance RELATIVE to each other.
"""

import numpy as np
from typing import Dict, Optional


def extract_all_teams_performance(
    all_team_data: Dict[str, Dict],
    session_name: str = 'fp1'
) -> Dict[str, Dict]:
    """
    Extract performance for all teams, normalized relative to each other.
    
    This is the ONLY function you should call. It handles everything.
    
    Args:
        all_team_data: {team: {session_key: session_data, ...}}
        session_name: Which session to use ('fp1', 'fp2', 'fp3')
    
    Returns:
        {team: {performance_metric: 0-1 score}}
    """
    # Step 1: Extract raw metrics from all teams
    raw_metrics = {}
    
    for team, sessions in all_team_data.items():
        # Find matching session
        session_data = _find_session(sessions, session_name)
        if not session_data:
            continue
        
        metrics = _extract_raw_metrics(session_data)
        if metrics:
            raw_metrics[team] = metrics
    
    if not raw_metrics:
        return {}
    
    # Step 2: Normalize across all teams (z-scores)
    return _normalize_relative(raw_metrics)



def _find_session(sessions: Dict, session_name: str) -> Optional[Dict]:
    """
    Find session that matches name with flexible matching.
    
    Handles variations like:
    - 'fp1' matches 'bahrain_grand_prix_fp1'
    - 'sprint_quali' matches 'miami_grand_prix_sprint_qualifying'
    - 'fp2' matches 'bahrain_fp2'
    """
    # Direct match
    if session_name in sessions:
        return sessions[session_name]
    
    # Normalize session name for flexible matching
    session_normalized = session_name.lower().replace('_', '').replace(' ', '')
    
    # Try exact suffix match first
    suffix = f'_{session_name}'
    for key, data in sessions.items():
        if key.endswith(suffix):
            return data
    
    # Flexible match: check if normalized session name is in key
    # This handles 'sprint_quali' matching 'sprint_qualifying'
    for key, data in sessions.items():
        key_normalized = key.lower().replace('_', '').replace(' ', '')
        if session_normalized in key_normalized:
            # Make sure it's actually the session type, not just coincidence
            # Check if it ends with the session pattern
            if key_normalized.endswith(session_normalized):
                return data
    
    return None


def _extract_raw_metrics(session_data: Dict) -> Dict:
    """Extract raw metrics (no normalization)."""
    metrics = {}
    
    if 'sector_times' in session_data:
        st = session_data['sector_times']
        if st.get('s1'): metrics['s1'] = st['s1']
        if st.get('s2'): metrics['s2'] = st['s2']
        if st.get('s3'): metrics['s3'] = st['s3']
    
    if 'speed_profile' in session_data:
        sp = session_data['speed_profile']
        if sp.get('top_speed'): metrics['top_speed'] = sp['top_speed']
    
    if 'consistency' in session_data:
        cons = session_data['consistency']
        if cons.get('std_lap_time'): metrics['std'] = cons['std_lap_time']
    
    return metrics


def _normalize_relative(raw_metrics: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Normalize all teams relative to each other using z-scores.
    
    Returns 0-1 scores where teams are compared against each other.
    """
    # Collect all values per metric
    all_values = {}
    for team, metrics in raw_metrics.items():
        for metric, value in metrics.items():
            if metric not in all_values:
                all_values[metric] = []
            all_values[metric].append(value)
    
    # Calculate stats
    stats = {}
    for metric, values in all_values.items():
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values) if len(values) > 1 and np.std(values) > 0 else 1.0
        }
    
    # Normalize each team
    normalized = {}
    
    for team, metrics in raw_metrics.items():
        perf = {}
        
        # S1 (lower is better, so negate z-score)
        if 's1' in metrics:
            z = -(metrics['s1'] - stats['s1']['mean']) / stats['s1']['std']
            perf['slow_corner_performance'] = _sigmoid(z)
            perf['braking_performance'] = perf['slow_corner_performance']
        
        # S2 (lower is better)
        if 's2' in metrics:
            z = -(metrics['s2'] - stats['s2']['mean']) / stats['s2']['std']
            perf['medium_corner_performance'] = _sigmoid(z)
        
        # S3 (lower is better)
        if 's3' in metrics:
            z = -(metrics['s3'] - stats['s3']['mean']) / stats['s3']['std']
            perf['fast_corner_performance'] = _sigmoid(z)
        
        # Top speed (higher is better, don't negate)
        if 'top_speed' in metrics:
            z = (metrics['top_speed'] - stats['top_speed']['mean']) / stats['top_speed']['std']
            perf['top_speed'] = _sigmoid(z)
        
        # Consistency (lower std is better, so negate)
        if 'std' in metrics:
            z = -(metrics['std'] - stats['std']['mean']) / stats['std']['std']
            perf['consistency'] = _sigmoid(z)
        
        normalized[team] = perf
    
    return normalized


def _sigmoid(z: float) -> float:
    """Convert z-score to 0-1 probability."""
    return float(1.0 / (1.0 + np.exp(-z)))


# Quick test
if __name__ == "__main__":
    test_data = {
        'McLaren': {
            'bahrain_grand_prix_fp1': {
                'sector_times': {'s1': 29.546, 's2': 40.369, 's3': 23.289},
                'speed_profile': {'top_speed': 315.0},
                'consistency': {'std_lap_time': 16.08}
            }
        },
        'Ferrari': {
            'bahrain_grand_prix_fp1': {
                'sector_times': {'s1': 29.695, 's2': 40.477, 's3': 23.628},
                'speed_profile': {'top_speed': 320.0},
                'consistency': {'std_lap_time': 16.54}
            }
        },
        'Kick Sauber': {
            'bahrain_grand_prix_fp1': {
                'sector_times': {'s1': 29.724, 's2': 40.972, 's3': 23.566},
                'speed_profile': {'top_speed': 326.0},
                'consistency': {'std_lap_time': 18.33}
            }
        }
    }
    
    result = extract_all_teams_performance(test_data, 'fp1')
    
    print("Relative Performance:")
    print("=" * 60)
    for team, perf in result.items():
        print(f"\n{team}:")
        for k, v in perf.items():
            print(f"  {k}: {v:.3f}")

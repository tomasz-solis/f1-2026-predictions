"""
Session Order Extractor - ENHANCED FOR SPRINT SESSIONS

Added more Sprint Qualifying session name variations.
Better diagnostics for what's failing.
"""

import fastf1 as ff1
import pandas as pd
import numpy as np
import logging

logging.getLogger("fastf1").setLevel(logging.CRITICAL)


def extract_fp_order_from_laps(year, race_name, session_type, verbose=False):
    """
    Extract team order from FP/Sprint session using lap times.
    """
    # Session name variations
    variations = {
        'FP1': ['FP1', 'Practice 1', 'Free Practice 1', 'P1'],
        'FP2': ['FP2', 'Practice 2', 'Free Practice 2', 'P2'],
        'FP3': ['FP3', 'Practice 3', 'Free Practice 3', 'P3'],
        'Sprint Qualifying': [
            'Sprint Qualifying',
            'Sprint Shootout',
            'SQ',
            'Sprint Quali',
            'SprintQualifying',
            'Sprint_Qualifying'
        ],
        'Sprint': ['Sprint', 'S', 'Sprint Race']
    }
    
    session_variations = variations.get(session_type, [session_type])
    
    for variation in session_variations:
        try:
            session = ff1.get_session(year, race_name, variation)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            
            if not hasattr(session, 'laps') or session.laps is None or len(session.laps) == 0:
                continue
            
            laps = session.laps
            
            # Get fastest lap per team
            team_times = {}
            
            for team in laps['Team'].unique():
                if pd.isna(team):
                    continue
                
                team_laps = laps[laps['Team'] == team]
                
                # Get each driver's fastest lap
                driver_best_times = []
                for driver in team_laps['Driver'].unique():
                    driver_laps = team_laps[team_laps['Driver'] == driver]
                    
                    # Filter valid laps
                    valid_laps = driver_laps[
                        (driver_laps['LapTime'].notna()) &
                        (~driver_laps['IsAccurate'].isna() if 'IsAccurate' in driver_laps else True)
                    ]
                    
                    if len(valid_laps) > 0:
                        best_time = valid_laps['LapTime'].min()
                        driver_best_times.append(best_time.total_seconds())
                
                if driver_best_times:
                    team_times[team] = np.median(driver_best_times)
            
            if len(team_times) < 5:
                continue
            
            # Convert to ranks
            sorted_teams = sorted(team_times.items(), key=lambda x: x[1])
            team_ranks = {team: rank for rank, (team, _) in enumerate(sorted_teams, 1)}
            
            if verbose:
                print(f"  → Loaded via '{variation}': {len(team_ranks)} teams")
            
            return team_ranks
            
        except Exception as e:
            if verbose:
                print(f"  → Failed '{variation}': {str(e)[:50]}")
            continue
    
    return None


def extract_quali_order_from_positions(year, race_name, session_type, verbose=False):
    """
    Extract team order from Qualifying using positions.
    """
    variations = {
        'Q': ['Q', 'Qualifying', 'Quali'],
        'Sprint Qualifying': [
            'Sprint Qualifying',
            'Sprint Shootout',
            'SQ',
            'Sprint Quali',
            'SprintQualifying',
            'Sprint_Qualifying'
        ]
    }
    
    session_variations = variations.get(session_type, [session_type])
    
    for variation in session_variations:
        try:
            session = ff1.get_session(year, race_name, variation)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            
            if not hasattr(session, 'results') or session.results is None:
                continue
            
            results = session.results
            
            if 'Position' not in results.columns:
                continue
            
            valid_positions = results['Position'].notna().sum()
            if valid_positions < 5:
                continue
            
            # Extract team positions
            team_positions = {}
            
            for team in results['TeamName'].unique():
                if pd.isna(team):
                    continue
                
                team_results = results[results['TeamName'] == team]
                positions = team_results['Position'].dropna()
                
                if len(positions) > 0:
                    team_positions[team] = float(np.median(positions))
            
            if len(team_positions) < 5:
                continue
            
            # Convert to ranks
            sorted_teams = sorted(team_positions.items(), key=lambda x: x[1])
            team_ranks = {team: rank for rank, (team, _) in enumerate(sorted_teams, 1)}
            
            if verbose:
                print(f"  → Loaded via '{variation}': {len(team_ranks)} teams")
            
            return team_ranks
            
        except Exception as e:
            if verbose:
                print(f"  → Failed '{variation}': {str(e)[:50]}")
            continue
    
    return None


def extract_session_order_robust(year, race_name, session_type, verbose=False):
    """
    Extract team finishing order from any session.
    
    Args:
        year: Season year
        race_name: Race name
        session_type: Session type
        verbose: Print diagnostic info
        
    Returns:
        Dict mapping team -> rank, or None
    """
    fp_sessions = ['FP1', 'FP2', 'FP3']
    quali_sessions = ['Q', 'Sprint Qualifying']
    
    # Try quali method first for Sprint Qualifying
    if session_type == 'Sprint Qualifying':
        if verbose:
            print(f"Trying Sprint Qualifying with positions method first...")
        
        result = extract_quali_order_from_positions(year, race_name, session_type, verbose)
        if result:
            return result
        
        if verbose:
            print(f"Trying Sprint Qualifying with lap times method...")
        
        # Try lap times as fallback
        result = extract_fp_order_from_laps(year, race_name, session_type, verbose)
        if result:
            return result
        
        # Try Sprint session itself
        if verbose:
            print(f"Trying Sprint session with lap times...")
        
        result = extract_fp_order_from_laps(year, race_name, 'Sprint', verbose)
        return result
    
    elif session_type in fp_sessions:
        return extract_fp_order_from_laps(year, race_name, session_type, verbose)
    
    elif session_type in quali_sessions:
        return extract_quali_order_from_positions(year, race_name, session_type, verbose)
    
    else:
        # Try both methods
        result = extract_quali_order_from_positions(year, race_name, session_type, verbose)
        if result:
            return result
        return extract_fp_order_from_laps(year, race_name, session_type, verbose)


def calculate_order_mae(predicted_order, actual_order):
    """Calculate MAE between predicted and actual team order."""
    errors = []
    
    for team in predicted_order:
        if team in actual_order:
            error = abs(predicted_order[team] - actual_order[team])
            errors.append(error)
    
    return np.mean(errors) if errors else None


def test_session_as_predictor_fixed(
    year,
    race_name,
    predictor_session,
    target_session='Q',
    driver_ranker=None,
    lineups=None,
    actual_driver_results=None,
    verbose=False
):
    """
    Test how well a session predicts qualifying.
    """
    # Get predictor session order
    if verbose:
        print(f"\nExtracting predictor: {predictor_session}")
    
    predictor_order = extract_session_order_robust(year, race_name, predictor_session, verbose)
    
    if predictor_order is None:
        return {
            'status': 'failed',
            'reason': f'{predictor_session} data not available',
            'race': race_name
        }
    
    # Get actual qualifying order
    if verbose:
        print(f"Extracting target: {target_session}")
    
    actual_order = extract_session_order_robust(year, race_name, target_session, verbose)
    
    if actual_order is None:
        return {
            'status': 'failed',
            'reason': f'{target_session} data not available',
            'race': race_name
        }
    
    # Calculate team-level MAE
    team_mae = calculate_order_mae(predictor_order, actual_order)
    
    result = {
        'status': 'success',
        'race': race_name,
        'predictor_session': predictor_session,
        'target_session': target_session,
        'team_mae': team_mae,
        'predictor_order': predictor_order,
        'actual_order': actual_order
    }
    
    # If driver ranker provided, test driver-level
    if driver_ranker and lineups and actual_driver_results:
        try:
            driver_preds = driver_ranker.predict_positions(
                team_predictions=predictor_order,
                team_lineups=lineups,
                session_type='qualifying'
            )
            
            errors = []
            
            for pred in driver_preds['predictions']:
                actual_pos = next(
                    (p['position'] for p in actual_driver_results if p['driver'] == pred.driver),
                    None
                )
                
                if actual_pos and pd.notna(actual_pos):
                    errors.append(abs(pred.position - actual_pos))
            
            if errors:
                result['driver_mae'] = np.mean(errors)
                result['driver_within_1'] = sum(1 for e in errors if e <= 1) / len(errors)
                result['driver_within_2'] = sum(1 for e in errors if e <= 2) / len(errors)
                result['driver_within_3'] = sum(1 for e in errors if e <= 3) / len(errors)
        except Exception as e:
            result['driver_error'] = str(e)
    
    return result


if __name__ == '__main__':
    # Test on sprint weekend
    print("Testing Sprint Qualifying extraction...")
    print("="*70)
    
    # Test Chinese GP (sprint weekend)
    print("\nChinese Grand Prix (Sprint):")
    sq_order = extract_session_order_robust(2025, 'Chinese Grand Prix', 'Sprint Qualifying', verbose=True)
    
    if sq_order:
        print(f"\n🟢 Sprint Qualifying extracted: {len(sq_order)} teams")
        sorted_teams = sorted(sq_order.items(), key=lambda x: x[1])
        for team, rank in sorted_teams[:5]:
            print(f"  {rank}. {team}")
    else:
        print("\n🔴 Sprint Qualifying failed!")
    
    # Try Miami too
    print("\n" + "="*70)
    print("\nMiami Grand Prix (Sprint):")
    sq_order = extract_session_order_robust(2025, 'Miami Grand Prix', 'Sprint Qualifying', verbose=True)
    
    if sq_order:
        print(f"\n🟢 Sprint Qualifying extracted: {len(sq_order)} teams")
        sorted_teams = sorted(sq_order.items(), key=lambda x: x[1])
        for team, rank in sorted_teams[:5]:
            print(f"  {rank}. {team}")
    else:
        print("\n🔴 Sprint Qualifying failed!")

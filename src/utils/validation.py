"""
Validation metrics for F1 predictions

Compare predicted rankings vs actual qualifying results
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, kendalltau

try:
    from src.utils.team_mapping import canonicalize_team
except ImportError:
    from utils.team_mapping import canonicalize_team


def compare_rankings(
    predicted: List[str],
    actual: List[str]
) -> Dict[str, float]:
    """
    Compare predicted team ranking vs actual results
    
    Args:
        predicted: ['Mercedes', 'Red Bull', 'McLaren', ...]
        actual: ['Red Bull', 'Mercedes', 'Ferrari', ...]
    
    Returns:
        Dict with all metrics
    """
    # Normalize team names
    predicted = [canonicalize_team(t) for t in predicted]
    actual = [canonicalize_team(t) for t in actual]
    
    metrics = {}
    
    # Winner prediction
    if len(predicted) > 0 and len(actual) > 0:
        metrics['winner_correct'] = 1.0 if predicted[0] == actual[0] else 0.0
    
    # Top N accuracy
    for n in [3, 5, 10]:
        if len(predicted) >= n and len(actual) >= n:
            pred_top_n = set(predicted[:n])
            actual_top_n = set(actual[:n])
            overlap = len(pred_top_n & actual_top_n)
            metrics[f'top{n}_accuracy'] = overlap / n
    
    # Ranking correlation
    common_teams = [t for t in predicted if t in actual]
    
    if len(common_teams) >= 3:
        pred_positions = [predicted.index(t) + 1 for t in common_teams]
        actual_positions = [actual.index(t) + 1 for t in common_teams]
        
        # Spearman correlation
        rho, _ = spearmanr(pred_positions, actual_positions)
        metrics['spearman'] = float(rho)
        
        # Kendall's tau
        tau, _ = kendalltau(pred_positions, actual_positions)
        metrics['kendall_tau'] = float(tau)
        
        # Mean absolute error (positions off)
        mae = np.mean([abs(p - a) for p, a in zip(pred_positions, actual_positions)])
        metrics['mae_positions'] = float(mae)
    
    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple races
    
    Returns mean for each metric
    """
    if not all_metrics:
        return {}
    
    keys = all_metrics[0].keys()
    aggregated = {}
    
    for key in keys:
        values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        if values:
            aggregated[key] = np.mean(values)
    
    return aggregated


def confidence_calibration(
    predictions: List[Tuple[float, bool]]
) -> Dict[str, float]:
    """
    Check if confidence scores are well-calibrated
    
    Args:
        predictions: [(confidence, was_correct), ...]
    
    Returns:
        Calibration metrics
    """
    if not predictions:
        return {}
    
    # Group by confidence bins
    bins = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
    bin_accuracy = {}
    
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        bin_preds = [(conf, correct) for conf, correct in predictions 
                     if low <= conf < high]
        
        if bin_preds:
            accuracy = sum(correct for _, correct in bin_preds) / len(bin_preds)
            avg_conf = np.mean([conf for conf, _ in bin_preds])
            bin_accuracy[f'{low:.1f}-{high:.1f}'] = {
                'confidence': avg_conf,
                'accuracy': accuracy,
                'count': len(bin_preds)
            }
    
    # Brier score (lower is better, 0-1 range)
    brier = np.mean([(conf - (1.0 if correct else 0.0))**2 
                     for conf, correct in predictions])
    
    return {
        'brier_score': float(brier),
        'bins': bin_accuracy
    }


def analyze_by_track_type(
    results: Dict[str, Dict],
    track_types: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Group results by track type
    
    Args:
        results: {race_name: {metrics}}
        track_types: {race_name: 'street'/'high-speed'/'technical'/'mixed'}
    
    Returns:
        {track_type: aggregated_metrics}
    """
    by_type = {}
    
    for race, metrics in results.items():
        track_type = track_types.get(race, 'unknown')
        
        if track_type not in by_type:
            by_type[track_type] = []
        
        by_type[track_type].append(metrics)
    
    # Aggregate each type
    aggregated = {}
    for track_type, metrics_list in by_type.items():
        aggregated[track_type] = aggregate_metrics(metrics_list)
        aggregated[track_type]['count'] = len(metrics_list)
    
    return aggregated


def analyze_by_stage(
    results: Dict[str, Dict[str, Dict]]
) -> Dict[str, Dict]:
    """
    Compare prediction quality by stage
    
    Args:
        results: {race: {'post_fp1': metrics, 'post_fp2': metrics, ...}}
    
    Returns:
        {stage: aggregated_metrics}
    """
    by_stage = {}
    
    for race, stage_results in results.items():
        for stage, metrics in stage_results.items():
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(metrics)
    
    # Aggregate each stage
    aggregated = {}
    for stage, metrics_list in by_stage.items():
        aggregated[stage] = aggregate_metrics(metrics_list)
        aggregated[stage]['count'] = len(metrics_list)
    
    return aggregated


if __name__ == "__main__":
    # Quick test
    predicted = ['Mercedes', 'Red Bull', 'McLaren', 'Ferrari', 'Alpine']
    actual = ['Red Bull', 'Mercedes', 'Ferrari', 'McLaren', 'Aston Martin']
    
    metrics = compare_rankings(predicted, actual)
    
    print("Validation Metrics Test:")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

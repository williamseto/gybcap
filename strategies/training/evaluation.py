"""Model evaluation utilities."""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, Any]:
    """
    Evaluate model predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            results['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            results['auc'] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        results['true_positives'] = int(tp)
        results['true_negatives'] = int(tn)
        results['false_positives'] = int(fp)
        results['false_negatives'] = int(fn)

    return results


def precision_recall_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.4
) -> Dict[str, Any]:
    """
    Perform precision-recall analysis.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        min_precision: Minimum precision constraint

    Returns:
        Dictionary with precision-recall analysis results
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)

    # Find threshold that maximizes recall subject to precision constraint
    valid = np.where(prec >= min_precision)[0]

    if len(valid) > 0:
        best_idx = valid[np.argmax(rec[valid])]
        if best_idx >= len(thresholds):
            best_idx = len(thresholds) - 1
        best_thresh = thresholds[best_idx]
        best_prec = prec[best_idx]
        best_rec = rec[best_idx]
    else:
        best_thresh = 0.5
        best_idx = np.argmin(np.abs(thresholds - 0.5))
        best_prec = prec[best_idx]
        best_rec = rec[best_idx]

    # Metrics at 0.5 threshold
    half_idx = np.argmin(np.abs(thresholds - 0.5))

    return {
        'best_threshold': best_thresh,
        'best_precision': best_prec,
        'best_recall': best_rec,
        'precision_at_0.5': prec[half_idx],
        'recall_at_0.5': rec[half_idx],
        'precisions': prec,
        'recalls': rec,
        'thresholds': thresholds,
    }


def calculate_trading_metrics(
    trades: List,
    predictions: np.ndarray = None
) -> Dict[str, Any]:
    """
    Calculate trading performance metrics.

    Args:
        trades: List of Trade objects with pnl attribute
        predictions: Optional binary predictions to filter trades

    Returns:
        Dictionary with trading metrics
    """
    if predictions is not None:
        filtered_trades = [t for t, p in zip(trades, predictions) if p == 1]
    else:
        filtered_trades = trades

    if not filtered_trades:
        return {
            'total_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'max_pnl': 0.0,
            'min_pnl': 0.0,
            'profit_factor': 0.0,
        }

    pnls = [t.pnl for t in filtered_trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    total_profit = sum(winners) if winners else 0
    total_loss = abs(sum(losers)) if losers else 0

    return {
        'total_trades': len(filtered_trades),
        'total_pnl': sum(pnls),
        'win_rate': len(winners) / len(filtered_trades) if filtered_trades else 0,
        'avg_pnl': np.mean(pnls),
        'max_pnl': max(pnls),
        'min_pnl': min(pnls),
        'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
        'n_winners': len(winners),
        'n_losers': len(losers),
        'avg_win': np.mean(winners) if winners else 0,
        'avg_loss': np.mean(losers) if losers else 0,
    }


def compare_strategies(
    results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare multiple strategy results.

    Args:
        results: Dictionary mapping strategy name to metrics dict

    Returns:
        DataFrame comparing strategies
    """
    df = pd.DataFrame(results).T

    # Reorder columns for readability
    col_order = [
        'total_trades', 'total_pnl', 'win_rate', 'avg_pnl',
        'profit_factor', 'max_pnl', 'min_pnl'
    ]
    existing_cols = [c for c in col_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]

    return df[existing_cols + other_cols]

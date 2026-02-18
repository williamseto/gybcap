"""Isolation Forest-based anomaly detection with walk-forward discipline.

Fits on training window features, scores all data. Returns continuous
anomaly scores (higher = more anomalous).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class RollingAnomalyDetector:
    """Walk-forward anomaly detection using Isolation Forest.

    Fits on training window features, scores entire DataFrame.
    Returns continuous anomaly scores (higher = more anomalous).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        max_features: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.model_ = None

    def fit_score(self, features: pd.DataFrame, train_end_idx: int) -> pd.DataFrame:
        """Fit on features[:train_end_idx], score entire DataFrame.

        Args:
            features: Feature matrix (rows = days, columns = features).
            train_end_idx: Index of last training row (exclusive).

        Returns:
            DataFrame with columns:
              - anomaly_score: continuous score normalized to [0, 1], 1 = most anomalous
              - anomaly_label: binary (1=anomaly, 0=normal) from model's predict()
              - anomaly_percentile: rank percentile of score within training distribution
        """
        X = features.fillna(0).values
        X_train = X[:train_end_idx]

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_train)

        # Raw scores: lower = more anomalous in sklearn convention
        raw_scores = self.model_.decision_function(X)
        labels = self.model_.predict(X)  # 1=normal, -1=anomaly

        # Negate and normalize to [0, 1] where 1 = most anomalous
        negated = -raw_scores
        train_negated = negated[:train_end_idx]
        score_min = train_negated.min()
        score_max = train_negated.max()
        score_range = score_max - score_min
        if score_range > 0:
            normalized = (negated - score_min) / score_range
        else:
            normalized = np.zeros_like(negated)
        normalized = np.clip(normalized, 0, 1)

        # Percentile ranking against training distribution
        from scipy.stats import percentileofscore
        train_scores = normalized[:train_end_idx]
        percentiles = np.array([
            percentileofscore(train_scores, s, kind="rank") / 100.0
            for s in normalized
        ])

        out = pd.DataFrame(index=features.index)
        out["anomaly_score"] = normalized
        out["anomaly_label"] = (labels == -1).astype(int)
        out["anomaly_percentile"] = percentiles
        return out


def compute_anomaly_features_walkforward(
    feature_df: pd.DataFrame,
    train_end_idx: int,
    n_estimators: int = 200,
) -> pd.DataFrame:
    """Drop-in replacement for hmm_fn callback in walk_forward_cv.

    Fits Isolation Forest on feature_df[:train_end_idx], returns anomaly
    features for entire DataFrame.

    Args:
        feature_df: Full feature DataFrame (all days).
        train_end_idx: End of training window (exclusive).
        n_estimators: Number of trees in Isolation Forest.

    Returns:
        DataFrame with anomaly_score, anomaly_label, anomaly_percentile columns.
    """
    detector = RollingAnomalyDetector(n_estimators=n_estimators)
    return detector.fit_score(feature_df, train_end_idx)

"""
Reversal trainer with walk-forward cross-validation.

Supports training and comparing:
1. XGBoost path (hand-crafted features)
2. TCN path (sequence learning)
3. Hybrid path (combination)

Walk-forward validation ensures no look-ahead bias in time series.
"""

from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)


def _clear_gpu_memory():
    """Clear GPU memory to prevent OOM during walk-forward CV."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

from strategies.labeling.reversal_labels import ReversalLabeler, label_strong_reversals
from strategies.labeling.reversal_zones import TradeViableZoneLabeler
from strategies.reversal.predictor import (
    XGBoostReversalPredictor,
    TCNReversalPredictor,
    HybridReversalPredictor,
    PredictionResult
)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold: int
    train_days: int
    test_days: int
    train_samples: int
    test_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    reversal_precision: float  # Precision for reversal class
    reversal_recall: float     # Recall for reversal class
    confusion_matrix: np.ndarray


@dataclass
class PathResult:
    """Results from training a predictor path."""
    path_name: str
    fold_results: List[FoldResult]
    overall_accuracy: float
    overall_precision: float
    overall_recall: float
    overall_f1: float
    reversal_precision: float
    reversal_recall: float
    n_total_samples: int
    n_filtered_samples: int


class ReversalTrainer:
    """
    Trainer for reversal predictors with walk-forward validation.

    Supports training multiple predictor types and comparing results.
    """

    def __init__(
        self,
        min_move_pct: float = 0.002,
        slope_window: int = 20,
        validation_bars: int = 15,
        n_folds: int = 5,
        min_train_days: int = 100
    ):
        """
        Initialize trainer.

        Args:
            min_move_pct: Minimum move % for reversal labeling
            slope_window: Window for slope calculation in labeling
            validation_bars: Bars to validate reversal holds
            n_folds: Number of walk-forward folds
            min_train_days: Minimum training days per fold
        """
        self.min_move_pct = min_move_pct
        self.slope_window = slope_window
        self.validation_bars = validation_bars
        self.n_folds = n_folds
        self.min_train_days = min_train_days

        # Results storage
        self.labels_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.path_results: Dict[str, PathResult] = {}

    def prepare_labels(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute reversal labels for the dataset.

        Args:
            ohlcv: OHLCV DataFrame with trading_day column

        Returns:
            DataFrame with reversal labels
        """
        labeler = ReversalLabeler(
            min_move_pct=self.min_move_pct,
            slope_window=self.slope_window,
            validation_bars=self.validation_bars
        )

        self.labels_df = labeler.fit(ohlcv)
        labeler.print_summary()

        return self.labels_df

    def prepare_features(
        self,
        ohlcv: pd.DataFrame,
        include_htf: bool = True,
        include_volume: bool = True,
        include_quality: bool = True
    ) -> pd.DataFrame:
        """
        Compute all features for the dataset.

        Args:
            ohlcv: OHLCV DataFrame
            include_htf: Include higher-timeframe features
            include_volume: Include volume microstructure features
            include_quality: Include reversion quality features

        Returns:
            DataFrame with computed features
        """
        result = ohlcv.copy()

        if include_htf:
            print("Computing higher timeframe features...")
            from strategies.features.higher_timeframe import HigherTimeframeProvider
            htf = HigherTimeframeProvider()
            htf_features = htf._compute_impl(ohlcv)
            for col in htf.feature_names:
                if col in htf_features.columns:
                    result[col] = htf_features[col].values

        if include_volume:
            print("Computing volume microstructure features...")
            from strategies.features.volume_microstructure import VolumeMicrostructureProvider
            has_bidask = 'bidvolume' in ohlcv.columns
            vmp = VolumeMicrostructureProvider(include_bidask=has_bidask)
            vol_features = vmp._compute_impl(ohlcv)
            for col in vmp.feature_names:
                if col in vol_features.columns:
                    result[col] = vol_features[col].values

        if include_quality:
            print("Computing reversion quality features...")
            from strategies.features.reversion_quality import ReversionQualityProvider
            rqp = ReversionQualityProvider()
            qual_features = rqp._compute_impl(ohlcv)
            for col in rqp.feature_names:
                if col in qual_features.columns:
                    result[col] = qual_features[col].values

        self.features_df = result
        return result

    def _get_fold_splits(
        self,
        days: List,
        n_folds: int,
        min_train_days: int
    ) -> List[Tuple[List, List]]:
        """Get train/test day splits for walk-forward CV."""
        n_days = len(days)
        test_days_per_fold = (n_days - min_train_days) // n_folds

        splits = []
        for fold in range(n_folds):
            train_end_idx = min_train_days + fold * test_days_per_fold
            test_end_idx = train_end_idx + test_days_per_fold

            if fold == n_folds - 1:
                test_end_idx = n_days

            train_days = days[:train_end_idx]
            test_days = days[train_end_idx:test_end_idx]

            if len(test_days) > 0:
                splits.append((train_days, test_days))

        return splits

    def _evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold: int,
        train_days: int,
        test_days: int,
        train_samples: int,
        test_samples: int
    ) -> FoldResult:
        """Evaluate predictions and compute metrics."""
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Reversal-specific (class != 0)
        y_true_binary = (y_true != 0).astype(int)
        y_pred_binary = (y_pred != 0).astype(int)

        rev_precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rev_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        return FoldResult(
            fold=fold,
            train_days=train_days,
            test_days=test_days,
            train_samples=train_samples,
            test_samples=test_samples,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            reversal_precision=rev_precision,
            reversal_recall=rev_recall,
            confusion_matrix=cm
        )

    def train_xgboost_path(
        self,
        ohlcv: pd.DataFrame,
        labels: pd.DataFrame,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        verbose: bool = True
    ) -> PathResult:
        """
        Train XGBoost predictor with walk-forward validation.

        Args:
            ohlcv: OHLCV data
            labels: Reversal labels
            features_df: Pre-computed features
            feature_cols: Feature columns to use
            verbose: Print progress

        Returns:
            PathResult with all fold results
        """
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING XGBOOST PATH")
            print("=" * 60)

        # Get unique days
        days = sorted(ohlcv['trading_day'].unique())
        splits = self._get_fold_splits(days, self.n_folds, self.min_train_days)

        # Merge data
        df = features_df.copy()
        df['reversal_label'] = labels['reversal_label'].values
        df['reversal_magnitude'] = labels['reversal_magnitude'].values

        # Map labels: -1 (bear) -> 2
        df['label_class'] = df['reversal_label'].values.copy()
        df.loc[df['label_class'] == -1, 'label_class'] = 2

        fold_results = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_days, test_days) in enumerate(splits):
            if verbose:
                print(f"\nFold {fold + 1}/{len(splits)}:")
                print(f"  Train: {len(train_days)} days")
                print(f"  Test: {len(test_days)} days")

            # Split data
            train_mask = df['trading_day'].isin(train_days)
            test_mask = df['trading_day'].isin(test_days)

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(test_df) == 0:
                continue

            # Train predictor
            predictor = XGBoostReversalPredictor(feature_cols)
            predictor.train(
                train_df,
                train_df[['reversal_label', 'reversal_magnitude']].rename(
                    columns={'reversal_label': 'reversal_label'}
                ),
                train_df
            )

            # Predict on test
            result = predictor.predict(test_df)

            # Convert probabilities to class predictions
            # Class with highest probability
            probs = np.column_stack([
                1.0 - result.reversal_prob,  # none
                result.bull_prob,
                result.bear_prob
            ])
            y_pred = probs.argmax(axis=1)
            y_true = test_df['label_class'].values

            # Evaluate
            fold_result = self._evaluate_predictions(
                y_true, y_pred, fold,
                len(train_days), len(test_days),
                len(train_df), len(test_df)
            )
            fold_results.append(fold_result)

            all_y_true.extend(y_true.tolist())
            all_y_pred.extend(y_pred.tolist())

            if verbose:
                print(f"  Accuracy: {fold_result.accuracy:.2%}")
                print(f"  Reversal Precision: {fold_result.reversal_precision:.2%}")
                print(f"  Reversal Recall: {fold_result.reversal_recall:.2%}")

        # Overall results
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        result = PathResult(
            path_name='xgboost',
            fold_results=fold_results,
            overall_accuracy=accuracy_score(all_y_true, all_y_pred),
            overall_precision=precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            overall_recall=recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            overall_f1=f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            reversal_precision=precision_score(
                (all_y_true != 0).astype(int),
                (all_y_pred != 0).astype(int),
                zero_division=0
            ),
            reversal_recall=recall_score(
                (all_y_true != 0).astype(int),
                (all_y_pred != 0).astype(int),
                zero_division=0
            ),
            n_total_samples=len(all_y_true),
            n_filtered_samples=(all_y_pred != 0).sum()
        )

        self.path_results['xgboost'] = result

        if verbose:
            self._print_path_summary(result)

        return result

    def train_tcn_path(
        self,
        ohlcv: pd.DataFrame,
        labels: pd.DataFrame,
        verbose: bool = True
    ) -> PathResult:
        """
        Train TCN predictor with walk-forward validation.

        Note: Due to neural network training requirements, we use a simplified
        walk-forward where we train on all data up to test period.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING TCN PATH")
            print("=" * 60)

        days = sorted(ohlcv['trading_day'].unique())
        splits = self._get_fold_splits(days, self.n_folds, self.min_train_days)

        df = ohlcv.copy()
        df['reversal_label'] = labels['reversal_label'].values
        df['reversal_magnitude'] = labels['reversal_magnitude'].values

        # Map labels
        df['label_class'] = df['reversal_label'].values.copy()
        df.loc[df['label_class'] == -1, 'label_class'] = 2

        fold_results = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_days, test_days) in enumerate(splits):
            if verbose:
                print(f"\nFold {fold + 1}/{len(splits)}:")

            train_mask = df['trading_day'].isin(train_days)
            test_mask = df['trading_day'].isin(test_days)

            train_df = df[train_mask].reset_index(drop=True)
            test_df = df[test_mask].reset_index(drop=True)

            if len(test_df) == 0:
                continue

            # Train predictor
            predictor = TCNReversalPredictor(
                lookback_bars=60,
                num_epochs=20  # Reduced for CV
            )

            train_labels = train_df[['reversal_label', 'reversal_magnitude']]
            predictor.train(train_df, train_labels)

            # Predict on test
            result = predictor.predict(test_df)

            # Convert to class predictions
            probs = np.column_stack([
                1.0 - result.reversal_prob,
                result.bull_prob,
                result.bear_prob
            ])
            y_pred = probs.argmax(axis=1)
            y_true = test_df['label_class'].values

            # Handle length mismatch (sequences need lookback)
            min_len = min(len(y_pred), len(y_true))
            y_pred = y_pred[:min_len]
            y_true = y_true[:min_len]

            fold_result = self._evaluate_predictions(
                y_true, y_pred, fold,
                len(train_days), len(test_days),
                len(train_df), len(test_df)
            )
            fold_results.append(fold_result)

            all_y_true.extend(y_true.tolist())
            all_y_pred.extend(y_pred.tolist())

            if verbose:
                print(f"  Accuracy: {fold_result.accuracy:.2%}")
                print(f"  Reversal Precision: {fold_result.reversal_precision:.2%}")

            # Clean up memory after each fold
            del predictor
            del train_df, test_df
            _clear_gpu_memory()

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        result = PathResult(
            path_name='tcn',
            fold_results=fold_results,
            overall_accuracy=accuracy_score(all_y_true, all_y_pred),
            overall_precision=precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            overall_recall=recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            overall_f1=f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            reversal_precision=precision_score(
                (all_y_true != 0).astype(int),
                (all_y_pred != 0).astype(int),
                zero_division=0
            ),
            reversal_recall=recall_score(
                (all_y_true != 0).astype(int),
                (all_y_pred != 0).astype(int),
                zero_division=0
            ),
            n_total_samples=len(all_y_true),
            n_filtered_samples=(all_y_pred != 0).sum()
        )

        self.path_results['tcn'] = result

        if verbose:
            self._print_path_summary(result)

        return result

    def train_hybrid_path(
        self,
        ohlcv: pd.DataFrame,
        labels: pd.DataFrame,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        verbose: bool = True
    ) -> PathResult:
        """Train hybrid predictor with walk-forward validation."""
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING HYBRID PATH")
            print("=" * 60)

        days = sorted(ohlcv['trading_day'].unique())
        splits = self._get_fold_splits(days, self.n_folds, self.min_train_days)

        df = features_df.copy()
        df['reversal_label'] = labels['reversal_label'].values
        df['reversal_magnitude'] = labels['reversal_magnitude'].values
        df['label_class'] = df['reversal_label'].values.copy()
        df.loc[df['label_class'] == -1, 'label_class'] = 2

        fold_results = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_days, test_days) in enumerate(splits):
            if verbose:
                print(f"\nFold {fold + 1}/{len(splits)}:")

            train_mask = df['trading_day'].isin(train_days)
            test_mask = df['trading_day'].isin(test_days)

            train_df = df[train_mask].reset_index(drop=True)
            test_df = df[test_mask].reset_index(drop=True)

            if len(test_df) == 0:
                continue

            predictor = HybridReversalPredictor(
                feature_cols=feature_cols,
                num_epochs=20
            )

            train_labels = train_df[['reversal_label', 'reversal_magnitude']]
            predictor.train(train_df, train_labels, train_df)

            result = predictor.predict(test_df)

            probs = np.column_stack([
                1.0 - result.reversal_prob,
                result.bull_prob,
                result.bear_prob
            ])
            y_pred = probs.argmax(axis=1)
            y_true = test_df['label_class'].values

            min_len = min(len(y_pred), len(y_true))
            y_pred = y_pred[:min_len]
            y_true = y_true[:min_len]

            fold_result = self._evaluate_predictions(
                y_true, y_pred, fold,
                len(train_days), len(test_days),
                len(train_df), len(test_df)
            )
            fold_results.append(fold_result)

            all_y_true.extend(y_true.tolist())
            all_y_pred.extend(y_pred.tolist())

            if verbose:
                print(f"  Accuracy: {fold_result.accuracy:.2%}")

            # Clean up memory after each fold
            del predictor
            del train_df, test_df
            _clear_gpu_memory()

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        result = PathResult(
            path_name='hybrid',
            fold_results=fold_results,
            overall_accuracy=accuracy_score(all_y_true, all_y_pred),
            overall_precision=precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            overall_recall=recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            overall_f1=f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0),
            reversal_precision=precision_score(
                (all_y_true != 0).astype(int),
                (all_y_pred != 0).astype(int),
                zero_division=0
            ),
            reversal_recall=recall_score(
                (all_y_true != 0).astype(int),
                (all_y_pred != 0).astype(int),
                zero_division=0
            ),
            n_total_samples=len(all_y_true),
            n_filtered_samples=(all_y_pred != 0).sum()
        )

        self.path_results['hybrid'] = result

        if verbose:
            self._print_path_summary(result)

        return result

    def _print_path_summary(self, result: PathResult) -> None:
        """Print summary for a path result."""
        print(f"\n{'=' * 60}")
        print(f"{result.path_name.upper()} PATH SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total OOS samples: {result.n_total_samples:,}")
        print(f"Predicted reversals: {result.n_filtered_samples:,}")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {result.overall_accuracy:.2%}")
        print(f"  Precision: {result.overall_precision:.2%}")
        print(f"  Recall: {result.overall_recall:.2%}")
        print(f"  F1: {result.overall_f1:.2%}")
        print(f"\nReversal-Specific:")
        print(f"  Precision: {result.reversal_precision:.2%}")
        print(f"  Recall: {result.reversal_recall:.2%}")

    def compare_paths(self) -> pd.DataFrame:
        """
        Compare results across all trained paths.

        Returns:
            DataFrame with comparison metrics
        """
        if not self.path_results:
            raise ValueError("No paths trained. Call train_*_path() methods first.")

        rows = []
        for name, result in self.path_results.items():
            rows.append({
                'path': name,
                'accuracy': result.overall_accuracy,
                'precision': result.overall_precision,
                'recall': result.overall_recall,
                'f1': result.overall_f1,
                'reversal_precision': result.reversal_precision,
                'reversal_recall': result.reversal_recall,
                'n_samples': result.n_total_samples,
                'n_predicted_reversals': result.n_filtered_samples
            })

        comparison = pd.DataFrame(rows)
        comparison = comparison.sort_values('f1', ascending=False)

        print("\n" + "=" * 80)
        print("PATH COMPARISON (sorted by F1)")
        print("=" * 80)
        print(comparison.to_string(index=False))

        return comparison

    def analyze_feature_correlations(
        self,
        features_df: pd.DataFrame,
        labels: pd.DataFrame,
        feature_cols: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Analyze correlations between features and reversal labels.

        Args:
            features_df: Feature DataFrame
            labels: Label DataFrame
            feature_cols: Features to analyze
            top_n: Number of top features to show

        Returns:
            DataFrame with correlation analysis
        """
        # Binary reversal indicator
        y_reversal = (labels['reversal_label'] != 0).astype(float)

        correlations = []
        for col in feature_cols:
            if col not in features_df.columns:
                continue
            corr = features_df[col].fillna(0).corr(y_reversal)
            correlations.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)

        print("\n" + "=" * 60)
        print("FEATURE CORRELATION ANALYSIS")
        print("=" * 60)
        print(f"\nTop {top_n} features by correlation with reversal label:")
        print(corr_df.head(top_n).to_string(index=False))

        # Warn about weak features
        weak = corr_df[corr_df['abs_correlation'] < 0.05]
        if len(weak) > 0:
            print(f"\nWARNING: {len(weak)} features have |corr| < 0.05")
            print("These may not be predictive.")

        return corr_df

    def train_anomaly_path(
        self,
        ohlcv: pd.DataFrame,
        labels: pd.DataFrame,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        use_sequences: bool = False,
        target_recall: float = 0.7,
        epochs: int = 50,
        verbose: bool = True
    ) -> PathResult:
        """
        Train anomaly detection-based predictor with walk-forward validation.

        Key difference: Train on reversal samples ONLY, then detect similar patterns.
        This addresses the 499:1 class imbalance by learning "what reversals look like".

        Args:
            ohlcv: OHLCV data
            labels: Reversal labels
            features_df: Pre-computed features
            feature_cols: Feature columns to use
            use_sequences: Whether to use hybrid mode with sequences
            target_recall: Target recall for threshold tuning
            epochs: Training epochs per fold
            verbose: Print progress

        Returns:
            PathResult with all fold results
        """
        from strategies.reversal.anomaly_predictor import AnomalyReversalPredictor

        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING ANOMALY DETECTION PATH")
            print("=" * 60)
            print(f"Mode: {'Hybrid (sequences + features)' if use_sequences else 'Feature-only'}")
            print(f"Target recall: {target_recall:.0%}")

        # Get unique days
        days = sorted(ohlcv['trading_day'].unique())
        splits = self._get_fold_splits(days, self.n_folds, self.min_train_days)

        # Merge data
        df = features_df.copy()
        df['reversal_label'] = labels['reversal_label'].values
        df['reversal_magnitude'] = labels['reversal_magnitude'].values

        fold_results = []
        all_y_true = []
        all_y_pred = []
        all_scores = []

        for fold, (train_days, test_days) in enumerate(splits):
            if verbose:
                print(f"\nFold {fold + 1}/{len(splits)}:")
                print(f"  Train: {len(train_days)} days")
                print(f"  Test: {len(test_days)} days")

            # Split data
            train_mask = df['trading_day'].isin(train_days)
            test_mask = df['trading_day'].isin(test_days)

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            train_ohlcv = ohlcv[ohlcv['trading_day'].isin(train_days)].copy()
            test_ohlcv = ohlcv[ohlcv['trading_day'].isin(test_days)].copy()

            if len(test_df) == 0:
                continue

            # Count reversals in train
            n_train_reversals = (train_df['reversal_label'] != 0).sum()
            if verbose:
                print(f"  Train reversals: {n_train_reversals}")

            if n_train_reversals < 10:
                print(f"  Skipping fold: not enough reversals")
                continue

            # Train predictor
            predictor = AnomalyReversalPredictor(
                feature_cols=feature_cols,
                use_sequences=use_sequences,
                seq_lookback=60,
                dalton_integration='weighted' if 'dalton_trend_prob' in feature_cols else 'none'
            )

            train_labels = train_df[['reversal_label', 'reversal_magnitude']]
            predictor.train(
                train_ohlcv.reset_index(drop=True),
                train_df.reset_index(drop=True),
                train_labels.reset_index(drop=True),
                epochs=epochs,
                target_recall=target_recall,
                verbose=False
            )

            # Predict on test
            result = predictor.predict(
                test_ohlcv.reset_index(drop=True),
                test_df.reset_index(drop=True)
            )

            # Evaluate
            y_true_binary = (test_df['reversal_label'] != 0).astype(int).values
            y_pred_binary = result.is_reversal.astype(int)

            # Map to multiclass for consistency with other paths
            y_true = test_df['reversal_label'].values.copy()
            y_true[y_true == -1] = 2  # bear -> 2

            # For anomaly detection, we only predict reversal vs not
            # Map predictions: if anomaly detected, predict most likely direction
            # based on features (simplified: use sign of BB distance)
            y_pred = np.zeros(len(y_pred_binary), dtype=int)
            if 'intraday_bb_lower_dist' in test_df.columns:
                bb_lower = test_df['intraday_bb_lower_dist'].fillna(0).values
                bb_upper = test_df['intraday_bb_upper_dist'].fillna(0).values
                # Near lower BB -> bull reversal (1), near upper -> bear (2)
                direction = np.where(bb_lower < bb_upper, 1, 2)
                y_pred[y_pred_binary == 1] = direction[y_pred_binary == 1]

            fold_result = self._evaluate_predictions(
                y_true, y_pred, fold,
                len(train_days), len(test_days),
                len(train_df), len(test_df)
            )
            fold_results.append(fold_result)

            all_y_true.extend(y_true_binary.tolist())
            all_y_pred.extend(y_pred_binary.tolist())
            all_scores.extend(result.anomaly_score.tolist())

            if verbose:
                print(f"  Threshold: {result.threshold:.4f}")
                print(f"  Reversal Precision: {fold_result.reversal_precision:.2%}")
                print(f"  Reversal Recall: {fold_result.reversal_recall:.2%}")

            # Clean up memory after each fold to prevent OOM
            del predictor
            del train_df, test_df, train_ohlcv, test_ohlcv, train_labels
            _clear_gpu_memory()

        # Overall results
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        # Compute overall metrics
        tp = ((all_y_pred == 1) & (all_y_true == 1)).sum()
        fp = ((all_y_pred == 1) & (all_y_true == 0)).sum()
        fn = ((all_y_pred == 0) & (all_y_true == 1)).sum()

        overall_precision = tp / max(tp + fp, 1)
        overall_recall = tp / max(tp + fn, 1)
        overall_f1 = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-6)
        overall_accuracy = (all_y_pred == all_y_true).mean()

        result = PathResult(
            path_name='anomaly' + ('_hybrid' if use_sequences else '_feature'),
            fold_results=fold_results,
            overall_accuracy=overall_accuracy,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1=overall_f1,
            reversal_precision=overall_precision,
            reversal_recall=overall_recall,
            n_total_samples=len(all_y_true),
            n_filtered_samples=int((all_y_pred == 1).sum())
        )

        self.path_results[result.path_name] = result

        if verbose:
            self._print_path_summary(result)

        # Train final model on all data for threshold analysis/production use
        if verbose:
            print("\nTraining final model on all data for threshold analysis...")

        final_predictor = AnomalyReversalPredictor(
            feature_cols=feature_cols,
            use_sequences=use_sequences,
            seq_lookback=60,
            dalton_integration='weighted' if 'dalton_trend_prob' in feature_cols else 'none'
        )
        final_labels = df[['reversal_label', 'reversal_magnitude']]
        final_predictor.train(
            ohlcv.reset_index(drop=True),
            df.reset_index(drop=True),
            final_labels.reset_index(drop=True),
            epochs=epochs,
            target_recall=target_recall,
            verbose=False
        )
        self._last_anomaly_predictor = final_predictor

        return result

    def tune_threshold_for_recall(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        target_recall: float = 0.7
    ) -> Tuple[float, float, float]:
        """
        Find threshold that achieves target recall.

        Args:
            scores: Anomaly scores (higher = more likely reversal)
            y_true: True binary labels (1 = reversal, 0 = not)
            target_recall: Target recall to achieve

        Returns:
            threshold: Score threshold
            precision: Precision at this threshold
            f1: F1 score at this threshold
        """
        # Sort scores descending
        sorted_indices = np.argsort(-scores)

        n_positives = y_true.sum()
        if n_positives == 0:
            return scores.min(), 0.0, 0.0

        # Walk through thresholds until target recall achieved
        tp = 0
        threshold = scores.max()
        precision = 0.0

        for i, idx in enumerate(sorted_indices):
            if y_true[idx]:
                tp += 1
            recall = tp / n_positives
            if recall >= target_recall:
                threshold = scores[idx]
                precision = tp / (i + 1) if (i + 1) > 0 else 0.0
                break

        # Calculate F1
        if precision + target_recall > 0:
            f1 = 2 * (precision * target_recall) / (precision + target_recall)
        else:
            f1 = 0.0

        return threshold, precision, f1

    def compute_precision_recall_curve(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        n_thresholds: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve for anomaly scores.

        Args:
            scores: Anomaly scores
            y_true: True binary labels
            n_thresholds: Number of threshold points

        Returns:
            precisions: Precision at each threshold
            recalls: Recall at each threshold
            thresholds: Threshold values
        """
        thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

        precisions = []
        recalls = []

        n_positives = y_true.sum()

        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(n_positives, 1)

            precisions.append(precision)
            recalls.append(recall)

        return np.array(precisions), np.array(recalls), thresholds

    def prepare_zone_labels(
        self,
        ohlcv: pd.DataFrame,
        stop_loss_pct: float = 0.0015,
        min_reward_risk: float = 1.5,
        max_lookback_bars: int = 10
    ) -> pd.DataFrame:
        """
        Compute trade-viable zone labels for the dataset.

        Trade-viable zones label bars where entering a trade would be profitable,
        not just the exact reversal bar. This addresses the timing problem in
        exact-bar prediction.

        Args:
            ohlcv: OHLCV DataFrame with trading_day column
            stop_loss_pct: Maximum allowed MAE (0.15% = 3 ES points)
            min_reward_risk: Minimum reward:risk ratio
            max_lookback_bars: Max bars before reversal to check

        Returns:
            DataFrame with zone labels added
        """
        labeler = TradeViableZoneLabeler(
            stop_loss_pct=stop_loss_pct,
            min_reward_risk=min_reward_risk,
            max_lookback_bars=max_lookback_bars,
            min_move_pct=self.min_move_pct,
            slope_window=self.slope_window,
            validation_bars=self.validation_bars
        )

        self.zone_labels_df = labeler.fit(ohlcv)
        labeler.print_summary()

        return self.zone_labels_df


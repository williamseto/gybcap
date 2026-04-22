"""Walk-forward XGBoost training for range prediction.

Trains separate regressors for range_high_pct and range_low_pct per timeframe.
Uses walk-forward cross-validation to avoid look-ahead bias.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from strategies.range_predictor.config import RangePredictorConfig, TIMEFRAME_HORIZONS
from strategies.range_predictor.features import (
    aggregate_to_daily,
    compute_range_features,
    compute_targets,
    get_feature_names,
    prepare_dataset,
    prepare_newsletter_dataset,
    prepare_rth_dataset,
)


class RangeTrainer:
    """Walk-forward XGBoost trainer for range prediction."""

    def __init__(self, config: Optional[RangePredictorConfig] = None):
        self.config = config or RangePredictorConfig()
        self.models: Dict[str, Dict[str, XGBRegressor]] = {}
        self.results: Dict[str, dict] = {}

    def _train_single_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> XGBRegressor:
        """Train a single XGBoost regressor."""
        model = XGBRegressor(**self.config.xgb_params)
        model.fit(X_train, y_train)
        return model

    def walk_forward_cv(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        feature_names: List[str],
        target_col: str,
        n_folds: int = 5,
        min_train_days: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """Run walk-forward cross-validation for a single target.

        Args:
            features: Feature DataFrame (DatetimeIndex).
            targets: Target DataFrame with target_col column.
            feature_names: List of feature column names.
            target_col: Target column name ('range_high_pct' or 'range_low_pct').
            n_folds: Number of walk-forward folds.
            min_train_days: Minimum training set size.
            verbose: Print progress.

        Returns:
            Dict with fold results and summary metrics.
        """
        n_days = len(features)
        if n_days < min_train_days + n_folds:
            raise ValueError(
                f"Not enough days ({n_days}) for {n_folds} folds with "
                f"min_train_days={min_train_days}"
            )

        test_days_per_fold = (n_days - min_train_days) // n_folds

        if verbose:
            print(f"\n  Walk-forward CV for '{target_col}'")
            print(f"  Total days: {n_days}, folds: {n_folds}, "
                  f"~{test_days_per_fold} test days/fold")

        fold_results = []
        all_oos_preds = []
        all_oos_actuals = []

        X_all = features[feature_names].values
        y_all = targets[target_col].values

        for fold in range(n_folds):
            train_end = min_train_days + fold * test_days_per_fold
            test_end = train_end + test_days_per_fold
            if fold == n_folds - 1:
                test_end = n_days

            X_train = X_all[:train_end]
            y_train = y_all[:train_end]
            X_test = X_all[train_end:test_end]
            y_test = y_all[train_end:test_end]

            if len(X_test) == 0:
                continue

            model = self._train_single_model(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = np.mean(np.abs(y_pred - y_test))
            rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

            # Directional accuracy: did we predict the right relative magnitude?
            mean_target = np.mean(np.abs(y_test))
            mape = mae / mean_target if mean_target > 0 else float('inf')

            fold_results.append({
                'fold': fold,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
            })

            all_oos_preds.extend(y_pred.tolist())
            all_oos_actuals.extend(y_test.tolist())

            if verbose:
                print(f"    Fold {fold + 1}: MAE={mae:.6f}, RMSE={rmse:.6f}, MAPE={mape:.1%}")

        all_preds = np.array(all_oos_preds)
        all_actuals = np.array(all_oos_actuals)

        overall_mae = np.mean(np.abs(all_preds - all_actuals))
        overall_rmse = np.sqrt(np.mean((all_preds - all_actuals) ** 2))

        # R2
        ss_res = np.sum((all_actuals - all_preds) ** 2)
        ss_tot = np.sum((all_actuals - np.mean(all_actuals)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Hit rate: predicted range contains realized value
        # (evaluated at the model level, not yet the full range)
        correlation = np.corrcoef(all_preds, all_actuals)[0, 1]

        summary = {
            'fold_results': fold_results,
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_r2': r2,
            'correlation': correlation,
            'n_oos_samples': len(all_preds),
        }

        if verbose:
            print(f"  Overall: MAE={overall_mae:.6f}, R2={r2:.4f}, "
                  f"corr={correlation:.4f}")

        return summary

    def train_timeframe(
        self,
        daily: pd.DataFrame,
        timeframe: str = 'daily',
        verbose: bool = True,
    ) -> Dict[str, XGBRegressor]:
        """Train models for a single timeframe using width/center decomposition.

        Trains on width_pct (volatility-driven, total range) and center_pct
        (directional bias) instead of high/low independently. This separates
        the easy volatility problem from the hard directional problem.

        At prediction time, high/low are recovered:
            high_pct = width_pct/2 + center_pct
            low_pct  = width_pct/2 - center_pct

        Args:
            daily: Daily OHLCV DataFrame (DatetimeIndex).
            timeframe: One of 'daily', 'weekly', 'monthly', 'quarterly'.
            verbose: Print progress.

        Returns:
            Dict mapping target name to trained model.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING RANGE PREDICTOR — {timeframe.upper()}")
            print(f"{'='*60}")

        features_df, targets_df, feature_names = prepare_dataset(
            daily, timeframe=timeframe
        )

        if verbose:
            print(f"Dataset: {len(features_df)} days, {len(feature_names)} features")
            print(f"Horizon: {TIMEFRAME_HORIZONS[timeframe]} trading days")

        models = {}
        cv_results = {}

        for target in ['width_pct', 'center_pct']:
            if verbose:
                label = 'width (vol)' if target == 'width_pct' else 'center (dir)'
                print(f"\n--- Target: {target} [{label}] ---")

            cv = self.walk_forward_cv(
                features_df, targets_df, feature_names, target,
                n_folds=self.config.walk_forward_folds,
                min_train_days=self.config.min_train_days,
                verbose=verbose,
            )
            cv_results[target] = cv

            X = features_df[feature_names].values
            y = targets_df[target].values
            model = self._train_single_model(X, y)
            models[target] = model

            if verbose:
                importance = pd.Series(
                    model.feature_importances_, index=feature_names
                ).sort_values(ascending=False)
                total_imp = importance.sum()
                print(f"\n  Top 10 features for {target}:")
                for fname, imp in importance.head(10).items():
                    pct = imp / total_imp * 100 if total_imp > 0 else 0
                    print(f"    {fname}: {pct:.1f}%")

        self.models[timeframe] = models
        self.results[timeframe] = {
            'cv_results': cv_results,
            'feature_names': feature_names,
            'n_samples': len(features_df),
        }

        return models

    def train_all(
        self,
        daily: pd.DataFrame,
        verbose: bool = True,
    ) -> None:
        """Train models for all configured timeframes.

        Args:
            daily: Daily OHLCV DataFrame (DatetimeIndex).
            verbose: Print progress.
        """
        for tf in self.config.timeframes:
            self.train_timeframe(daily, timeframe=tf, verbose=verbose)

        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING SUMMARY")
            print(f"{'='*60}")
            for tf, res in self.results.items():
                for target, cv in res['cv_results'].items():
                    print(f"  {tf}/{target}: R2={cv['overall_r2']:.4f}, "
                          f"MAE={cv['overall_mae']:.6f}, "
                          f"corr={cv['correlation']:.4f}")

    def train_newsletter(
        self,
        daily: pd.DataFrame,
        newsletter: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict[str, XGBRegressor]:
        """Train models to replicate newsletter range predictions.

        Uses newsletter high/low as targets (width/center decomposition).
        Only trains a 'daily' timeframe since the newsletter publishes
        daily ES ranges.

        Args:
            daily: Daily OHLCV DataFrame (DatetimeIndex).
            newsletter: Newsletter predictions DataFrame.
            verbose: Print progress.

        Returns:
            Dict mapping target name to trained model.
        """
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING NEWSLETTER REPLICA — DAILY")
            print(f"{'='*60}")

        features_df, targets_df, feature_names = prepare_newsletter_dataset(
            daily, newsletter
        )

        if verbose:
            print(f"Dataset: {len(features_df)} days, {len(feature_names)} features")

        models = {}
        cv_results = {}

        # Train on all four targets to report full diagnostics
        for target in ['nl_width_pct', 'nl_center_pct', 'nl_high_pct', 'nl_low_pct']:
            if verbose:
                label = {
                    'nl_width_pct': 'width (vol)',
                    'nl_center_pct': 'center (dir)',
                    'nl_high_pct': 'high boundary',
                    'nl_low_pct': 'low boundary',
                }[target]
                print(f"\n--- Target: {target} [{label}] ---")

            cv = self.walk_forward_cv(
                features_df, targets_df, feature_names, target,
                n_folds=self.config.walk_forward_folds,
                min_train_days=self.config.min_train_days,
                verbose=verbose,
            )
            cv_results[target] = cv

            X = features_df[feature_names].values
            y = targets_df[target].values
            model = self._train_single_model(X, y)
            models[target] = model

            if verbose:
                importance = pd.Series(
                    model.feature_importances_, index=feature_names
                ).sort_values(ascending=False)
                total_imp = importance.sum()
                print(f"\n  Top 10 features for {target}:")
                for fname, imp in importance.head(10).items():
                    pct = imp / total_imp * 100 if total_imp > 0 else 0
                    print(f"    {fname}: {pct:.1f}%")

        # Store width/center as the primary models for prediction
        self.models['nl_daily'] = {
            'width_pct': models['nl_width_pct'],
            'center_pct': models['nl_center_pct'],
        }
        self.results['nl_daily'] = {
            'cv_results': cv_results,
            'feature_names': feature_names,
            'n_samples': len(features_df),
        }

        if verbose:
            print(f"\n  Newsletter Replica Summary:")
            for target, cv in cv_results.items():
                print(f"    {target}: R2={cv['overall_r2']:.4f}, "
                      f"MAE={cv['overall_mae']:.6f}, "
                      f"corr={cv['correlation']:.4f}")

        return models

    def train_rth(
        self,
        full_daily: pd.DataFrame,
        rth_daily: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict[str, XGBRegressor]:
        """Train RTH range prediction models.

        Uses full-session daily features + overnight gap features to predict
        RTH range (high/low relative to RTH open).

        Args:
            full_daily: Full-session daily OHLCV DataFrame (DatetimeIndex).
            rth_daily: RTH-only daily DataFrame with rth_open/high/low/close.
            verbose: Print progress.

        Returns:
            Dict mapping target name to trained model.
        """
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING RANGE PREDICTOR — RTH")
            print(f"{'='*60}")

        features_df, targets_df, feature_names = prepare_rth_dataset(
            full_daily, rth_daily
        )

        if verbose:
            print(f"Dataset: {len(features_df)} days, {len(feature_names)} features")
            print(f"Horizon: 1 trading day (RTH session only)")

        models = {}
        cv_results = {}

        for target in ['rth_range_high_pct', 'rth_range_low_pct']:
            if verbose:
                print(f"\n--- Target: {target} ---")

            # Walk-forward CV
            cv = self.walk_forward_cv(
                features_df, targets_df, feature_names, target,
                n_folds=self.config.walk_forward_folds,
                min_train_days=self.config.min_train_days,
                verbose=verbose,
            )
            cv_results[target] = cv

            # Train final model on all data
            X = features_df[feature_names].values
            y = targets_df[target].values
            model = self._train_single_model(X, y)
            models[target] = model

            if verbose:
                importance = pd.Series(
                    model.feature_importances_, index=feature_names
                ).sort_values(ascending=False)
                total_imp = importance.sum()
                print(f"\n  Top 10 features for {target}:")
                for fname, imp in importance.head(10).items():
                    pct = imp / total_imp * 100 if total_imp > 0 else 0
                    print(f"    {fname}: {pct:.1f}%")

        self.models['rth'] = models
        self.results['rth'] = {
            'cv_results': cv_results,
            'feature_names': feature_names,
            'n_samples': len(features_df),
        }

        if verbose:
            print(f"\n  RTH Training Summary:")
            for target, cv in cv_results.items():
                print(f"    {target}: R2={cv['overall_r2']:.4f}, "
                      f"MAE={cv['overall_mae']:.6f}, "
                      f"corr={cv['correlation']:.4f}")

        return models

    def save_models(self, model_dir: Optional[str] = None) -> None:
        """Save all trained models to disk.

        Args:
            model_dir: Directory to save models. Defaults to config.model_dir.
        """
        model_dir = model_dir or self.config.model_dir
        os.makedirs(model_dir, exist_ok=True)

        metadata = {}

        for tf, models in self.models.items():
            cv_results = self.results[tf]['cv_results']
            for target, model in models.items():
                if tf == 'rth':
                    fname = f"{target}.json"
                else:
                    fname = f"{tf}_{target}.json"
                fpath = os.path.join(model_dir, fname)
                model.save_model(fpath)

                # Look up CV results: exact key first, then nl_ prefix for newsletter models
                cv = cv_results.get(target) or cv_results.get(f'nl_{target}', {})
                metadata[f"{tf}/{target}"] = {
                    'file': fname,
                    'cv_results': {
                        k: v for k, v in cv.items()
                        if k != 'fold_results'
                    },
                }

        # Save metadata — use a standard timeframe for base feature names
        base_tfs = [tf for tf in self.results if tf not in ('rth', 'nl_daily')]
        if base_tfs:
            metadata['feature_names'] = self.results[base_tfs[0]]['feature_names']
        elif 'nl_daily' in self.results:
            metadata['feature_names'] = self.results['nl_daily']['feature_names']
        elif 'rth' in self.results:
            metadata['feature_names'] = self.results['rth']['feature_names']

        metadata['timeframes'] = list(self.models.keys())

        # Store RTH feature names separately (includes gap features)
        if 'rth' in self.results:
            metadata['rth_feature_names'] = self.results['rth']['feature_names']

        meta_path = os.path.join(model_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Models saved to {model_dir}/")

    def analyze_feature_correlations(
        self,
        daily: pd.DataFrame,
        timeframe: str = 'daily',
        min_correlation: float = 0.03,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Analyze feature correlations with targets.

        Args:
            daily: Daily OHLCV DataFrame.
            timeframe: Timeframe to analyze.
            min_correlation: Minimum useful correlation threshold.
            verbose: Print results.

        Returns:
            DataFrame with correlations.
        """
        features_df, targets_df, feature_names = prepare_dataset(
            daily, timeframe=timeframe
        )

        correlations = []
        for feat in feature_names:
            for target in ['range_high_pct', 'range_low_pct']:
                corr = features_df[feat].corr(targets_df[target])
                correlations.append({
                    'feature': feat,
                    'target': target,
                    'correlation': corr,
                    'abs_corr': abs(corr),
                })

        corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

        if verbose:
            print(f"\nFeature correlations with {timeframe} targets:")
            print(f"{'Feature':<30} {'high_corr':>10} {'low_corr':>10}")
            print("-" * 52)

            # Pivot for display
            pivot = corr_df.pivot(
                index='feature', columns='target', values='correlation'
            )
            pivot['max_abs'] = pivot.abs().max(axis=1)
            pivot = pivot.sort_values('max_abs', ascending=False)

            for feat, row in pivot.head(15).iterrows():
                print(f"  {feat:<28} {row.get('range_high_pct', 0):>10.4f} "
                      f"{row.get('range_low_pct', 0):>10.4f}")

            weak = pivot[pivot['max_abs'] < min_correlation]
            if len(weak) > 0:
                print(f"\nWARNING: {len(weak)} features with |corr| < {min_correlation}")

        return corr_df

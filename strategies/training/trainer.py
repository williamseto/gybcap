"""XGBoost model training for trading strategies with cross-validation.

Includes:
- Random CV (legacy, prone to overfitting on time series)
- Walk-forward CV (time-respecting, recommended for trading)
- Feature correlation analysis
"""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_curve, make_scorer, precision_score
import warnings


class Trainer:
    """
    XGBoost model trainer for trading strategy prediction.

    Trains a classifier to predict trade success based on features.
    Supports cross-validation for hyperparameter tuning.
    """

    DEFAULT_PARAMS = {
        'eval_metric': 'auc',
        'scale_pos_weight': 3.0,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'max_depth': 4,
        'min_child_weight': 5,
        'gamma': 0.5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'random_state': 27,
    }

    # Parameter grid for cross-validation
    PARAM_GRID = {
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5, 7],
        'scale_pos_weight': [1.0, 2.0, 3.0, 5.0],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [100, 300, 500, 1000],
        'gamma': [0, 0.1, 0.3, 0.5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }

    def __init__(
        self,
        feature_cols: List[str],
        params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.3,
        augment_data: bool = True,
        min_precision: float = 0.4,
        n_jobs: int = 1
    ):
        """
        Initialize trainer.

        Args:
            feature_cols: List of feature column names
            params: XGBoost parameters (overrides defaults)
            test_size: Fraction of data to use for validation
            augment_data: Whether to augment training data
            min_precision: Minimum precision for threshold selection
            n_jobs: Number of parallel jobs for CV (-1 uses all cores, can cause OOM)
        """
        self.feature_cols = feature_cols
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.test_size = test_size
        self.augment_data = augment_data
        self.min_precision = min_precision
        self.n_jobs = n_jobs

        self.model: Optional[XGBClassifier] = None
        self.best_threshold: float = 0.5
        self.training_results: Dict[str, Any] = {}
        self.cv_results: Optional[Dict[str, Any]] = None

    def prepare_data(
        self,
        trade_features_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data.

        Args:
            trade_features_df: DataFrame with features and y_succ column

        Returns:
            X_train, X_val, y_train, y_val
        """
        df = trade_features_df.fillna(0)

        X = df[self.feature_cols]
        y = df['y_succ']

        if self.augment_data:
            # Augment positive samples by flipping direction
            augmented_df = df[df['y_succ'] == 1].copy()
            augmented_X = augmented_df[self.feature_cols].copy()

            if 'bear' in augmented_X.columns:
                augmented_X['bear'] = 1 - augmented_X['bear']

            augmented_y = 1 - augmented_df['y_succ']  # Flip success

            X = pd.concat([X, augmented_X], axis=0)
            y = pd.concat([y, augmented_y], axis=0)

        X = X.values
        y = y.values

        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.params.get('random_state', 42),
            stratify=y
        )

    def cross_validate(
        self,
        trade_features_df: pd.DataFrame,
        n_folds: int = 5,
        n_iter: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform randomized cross-validation for hyperparameter tuning.

        Args:
            trade_features_df: DataFrame with features and y_succ column
            n_folds: Number of CV folds
            n_iter: Number of random parameter combinations to try
            verbose: Whether to print progress

        Returns:
            Dictionary with best parameters and CV scores
        """
        from sklearn.model_selection import RandomizedSearchCV

        df = trade_features_df.fillna(0)
        X = df[self.feature_cols].values
        y = df['y_succ'].values

        # Use precision as scoring metric
        scorer = make_scorer(precision_score, zero_division=0)

        base_model = XGBClassifier(
            eval_metric='auc',
            random_state=self.params.get('random_state', 27),
            use_label_encoder=False
        )

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            base_model,
            self.PARAM_GRID,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            verbose=2 if verbose else 0,
            random_state=42,
            n_jobs=self.n_jobs
        )

        if verbose:
            print(f"Running {n_iter} iterations of randomized CV with {n_folds} folds...")

        search.fit(X, y)

        self.cv_results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
        }

        if verbose:
            print(f"\nBest CV Score (Precision): {search.best_score_:.4f}")
            print(f"Best Parameters:")
            for k, v in search.best_params_.items():
                print(f"  {k}: {v}")

        # Update params with best found
        self.params.update(search.best_params_)

        return self.cv_results

    def grid_search_quick(
        self,
        trade_features_df: pd.DataFrame,
        n_folds: int = 3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Quick grid search over most important parameters.

        Args:
            trade_features_df: DataFrame with features
            n_folds: Number of CV folds
            verbose: Whether to print progress

        Returns:
            Best parameters found
        """
        df = trade_features_df.fillna(0)
        X = df[self.feature_cols].values
        y = df['y_succ'].values

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        best_score = 0
        best_params = {}

        # Quick grid over key parameters
        param_grid = {
            'max_depth': [3, 4, 5],
            'scale_pos_weight': [2.0, 3.0, 5.0],
            'n_estimators': [300, 500],
        }

        total = np.prod([len(v) for v in param_grid.values()])
        if verbose:
            print(f"Quick grid search over {total} combinations...")

        count = 0
        for max_depth in param_grid['max_depth']:
            for scale_pos_weight in param_grid['scale_pos_weight']:
                for n_estimators in param_grid['n_estimators']:
                    count += 1
                    params = {
                        **self.DEFAULT_PARAMS,
                        'max_depth': max_depth,
                        'scale_pos_weight': scale_pos_weight,
                        'n_estimators': n_estimators,
                    }

                    model = XGBClassifier(**params)
                    scores = cross_val_score(
                        model, X, y,
                        cv=cv,
                        scoring='precision',
                        n_jobs=self.n_jobs
                    )
                    mean_score = scores.mean()

                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params.copy()

                    if verbose and count % 6 == 0:
                        print(f"  {count}/{total}: best precision = {best_score:.4f}")

        if verbose:
            print(f"\nBest CV Precision: {best_score:.4f}")
            print(f"Best params: max_depth={best_params['max_depth']}, "
                  f"scale_pos_weight={best_params['scale_pos_weight']}, "
                  f"n_estimators={best_params['n_estimators']}")

        self.params.update(best_params)
        self.cv_results = {'best_params': best_params, 'best_score': best_score}

        return self.cv_results

    def train(
        self,
        trade_features_df: pd.DataFrame,
        verbose: bool = True
    ) -> XGBClassifier:
        """
        Train the model.

        Args:
            trade_features_df: DataFrame with features and y_succ column
            verbose: Whether to print training results

        Returns:
            Trained XGBClassifier
        """
        X_train, X_val, y_train, y_val = self.prepare_data(trade_features_df)

        self.model = XGBClassifier(**self.params)
        self.model.fit(X_train, y_train)

        # Predict probabilities on validation set
        y_proba = self.model.predict_proba(X_val)[:, 1]

        # Compute precision-recall curve
        prec, rec, thresholds = precision_recall_curve(y_val, y_proba)

        # Find optimal threshold
        valid = np.where(prec >= self.min_precision)[0]

        if len(valid) > 0:
            best_idx = valid[np.argmax(rec[valid])]
            if best_idx >= len(thresholds):
                best_idx = len(thresholds) - 1
            self.best_threshold = thresholds[best_idx]
        else:
            self.best_threshold = 0.5
            best_idx = np.argmin(np.abs(thresholds - 0.5))

        # Store results
        self.training_results = {
            'best_threshold': self.best_threshold,
            'recall_at_threshold': rec[best_idx],
            'precision_at_threshold': prec[best_idx],
            'n_train': len(y_train),
            'n_val': len(y_val),
            'val_positive_rate': y_val.mean(),
        }

        # Find metrics at 0.5 threshold
        half_idx = np.argmin(np.abs(thresholds - 0.5))
        self.training_results['recall_at_0.5'] = rec[half_idx]
        self.training_results['precision_at_0.5'] = prec[half_idx]

        if verbose:
            print(f"Chosen threshold: {self.best_threshold:.3f}")
            print(f"Recall at threshold: {rec[best_idx]:.3f}")
            print(f"Precision at threshold: {prec[best_idx]:.3f}")
            print(f"Recall at 0.5 threshold: {rec[half_idx]:.3f}")
            print(f"Precision at 0.5 threshold: {prec[half_idx]:.3f}")

        return self.model

    def train_with_cv(
        self,
        trade_features_df: pd.DataFrame,
        quick: bool = True,
        n_folds: int = 5,
        n_iter: int = 50,
        verbose: bool = True
    ) -> XGBClassifier:
        """
        Train with cross-validation for hyperparameter tuning.

        Args:
            trade_features_df: DataFrame with features
            quick: Use quick grid search (faster) vs randomized search
            n_folds: Number of CV folds
            n_iter: Number of iterations for randomized search
            verbose: Whether to print progress

        Returns:
            Trained XGBClassifier with optimized parameters
        """
        if verbose:
            print("=" * 50)
            print("STEP 1: Cross-validation for hyperparameter tuning")
            print("=" * 50)

        if quick:
            self.grid_search_quick(trade_features_df, n_folds=n_folds, verbose=verbose)
        else:
            self.cross_validate(trade_features_df, n_folds=n_folds, n_iter=n_iter, verbose=verbose)

        if verbose:
            print("\n" + "=" * 50)
            print("STEP 2: Training final model with best parameters")
            print("=" * 50)

        return self.train(trade_features_df, verbose=verbose)

    def predict(
        self,
        features: pd.DataFrame,
        use_threshold: bool = True
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            features: DataFrame with feature columns
            use_threshold: If True, use best_threshold; else use 0.5

        Returns:
            Binary predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = features[self.feature_cols].fillna(0).values
        proba = self.model.predict_proba(X)[:, 1]

        threshold = self.best_threshold if use_threshold else 0.5
        return (proba >= threshold).astype(int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            features: DataFrame with feature columns

        Returns:
            Array of probabilities for positive class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = features[self.feature_cols].fillna(0).values
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        self.model.save_model(path)

    def load_model(self, path: str) -> None:
        """Load model from file."""
        self.model = XGBClassifier()
        self.model.load_model(path)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.feature_importances_

        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def evaluate_trades(
        self,
        trades: list,
        trade_features_df: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on trades.

        Args:
            trades: List of Trade objects
            trade_features_df: DataFrame with features
            verbose: Whether to print results

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        preds = self.predict(trade_features_df)

        total_pnl = sum(t.pnl for t in trades)
        filtered_pnl = sum(
            trades[i].pnl for i in range(len(trades))
            if preds[i] == 1
        )
        n_filtered = preds.sum()

        # Calculate win rates
        total_winners = sum(1 for t in trades if t.pnl > 0)
        filtered_winners = sum(
            1 for i in range(len(trades))
            if preds[i] == 1 and trades[i].pnl > 0
        )

        results = {
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'total_win_rate': total_winners / len(trades) if trades else 0,
            'filtered_trades': n_filtered,
            'filtered_pnl': filtered_pnl,
            'filtered_win_rate': filtered_winners / n_filtered if n_filtered > 0 else 0,
            'avg_pnl_per_trade': total_pnl / len(trades) if trades else 0,
            'avg_filtered_pnl_per_trade': filtered_pnl / n_filtered if n_filtered > 0 else 0,
        }

        if verbose:
            print(f"Total trades: {results['total_trades']}, "
                  f"PnL: {results['total_pnl']:.2f}, "
                  f"WR: {results['total_win_rate']:.2%}")
            print(f"Filtered trades: {results['filtered_trades']}, "
                  f"PnL: {results['filtered_pnl']:.2f}, "
                  f"WR: {results['filtered_win_rate']:.2%}")

        return results

    def analyze_feature_correlations(
        self,
        trade_features_df: pd.DataFrame,
        min_correlation: float = 0.05,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Analyze correlations between features and target variable.

        Warns if features have near-zero correlation with y_succ, which
        indicates they may not be predictive.

        Args:
            trade_features_df: DataFrame with features and y_succ column
            min_correlation: Minimum absolute correlation to consider useful
            verbose: Whether to print warnings

        Returns:
            DataFrame with feature correlations sorted by absolute value
        """
        df = trade_features_df.fillna(0)
        y = df['y_succ']

        correlations = []
        weak_features = []

        for col in self.feature_cols:
            if col not in df.columns:
                continue
            corr = df[col].corr(y)
            correlations.append({'feature': col, 'correlation': corr})

            if abs(corr) < min_correlation:
                weak_features.append(col)

        corr_df = pd.DataFrame(correlations)
        corr_df['abs_corr'] = corr_df['correlation'].abs()
        corr_df = corr_df.sort_values('abs_corr', ascending=False)

        if verbose:
            print(f"\n{'='*50}")
            print("FEATURE CORRELATION ANALYSIS")
            print(f"{'='*50}")
            print(f"Minimum useful correlation threshold: {min_correlation}")
            print(f"\nTop features by correlation with y_succ:")
            for _, row in corr_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['correlation']:.4f}")

            if weak_features:
                print(f"\nWARNING: {len(weak_features)} features have |corr| < {min_correlation}:")
                for feat in weak_features[:10]:
                    corr_val = corr_df[corr_df['feature'] == feat]['correlation'].values[0]
                    print(f"  {feat}: {corr_val:.4f}")
                if len(weak_features) > 10:
                    print(f"  ... and {len(weak_features) - 10} more")
                print("\nThese features may not be predictive and could cause overfitting.")

        return corr_df

    def walk_forward_cv(
        self,
        trade_features_df: pd.DataFrame,
        day_col: str = 'trading_day',
        n_folds: int = 5,
        min_train_days: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform walk-forward cross-validation (time-respecting).

        Unlike random CV, this ensures training data always precedes test data,
        which is critical for time series to avoid look-ahead bias.

        Args:
            trade_features_df: DataFrame with features, y_succ, and day column
            day_col: Column containing trading day identifiers
            n_folds: Number of walk-forward folds
            min_train_days: Minimum number of days for initial training set
            verbose: Whether to print progress

        Returns:
            Dictionary with fold results and summary statistics
        """
        df = trade_features_df.fillna(0)

        # Get unique days sorted
        if day_col in df.columns:
            days = sorted(df[day_col].unique())
        else:
            # Infer days from index if possible
            if isinstance(df.index, pd.DatetimeIndex):
                df['_day'] = df.index.date
                days = sorted(df['_day'].unique())
                day_col = '_day'
            else:
                raise ValueError(f"No '{day_col}' column and index is not DatetimeIndex")

        n_days = len(days)
        if n_days < min_train_days + n_folds:
            raise ValueError(
                f"Not enough days ({n_days}) for {n_folds} folds with "
                f"min_train_days={min_train_days}"
            )

        # Calculate step size for sliding window
        test_days_per_fold = (n_days - min_train_days) // n_folds

        if verbose:
            print(f"\n{'='*50}")
            print("WALK-FORWARD CROSS-VALIDATION")
            print(f"{'='*50}")
            print(f"Total days: {n_days}")
            print(f"Min train days: {min_train_days}")
            print(f"Folds: {n_folds}")
            print(f"Test days per fold: ~{test_days_per_fold}")

        fold_results = []
        all_oos_preds = []
        all_oos_actuals = []
        all_oos_indices = []

        for fold in range(n_folds):
            train_end_idx = min_train_days + fold * test_days_per_fold
            test_end_idx = train_end_idx + test_days_per_fold

            if fold == n_folds - 1:
                # Last fold: use all remaining days
                test_end_idx = n_days

            train_days = days[:train_end_idx]
            test_days = days[train_end_idx:test_end_idx]

            train_mask = df[day_col].isin(train_days)
            test_mask = df[day_col].isin(test_days)

            X_train = df.loc[train_mask, self.feature_cols].values
            y_train = df.loc[train_mask, 'y_succ'].values
            X_test = df.loc[test_mask, self.feature_cols].values
            y_test = df.loc[test_mask, 'y_succ'].values

            if len(X_test) == 0 or len(X_train) == 0:
                continue

            # Train model
            model = XGBClassifier(**self.params)
            model.fit(X_train, y_train)

            # Predict
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            # Metrics
            accuracy = (y_pred == y_test).mean()
            n_pos_pred = y_pred.sum()
            n_pos_actual = y_test.sum()

            if n_pos_pred > 0:
                precision = (y_pred & y_test).sum() / n_pos_pred
            else:
                precision = 0.0

            if n_pos_actual > 0:
                recall = (y_pred & y_test).sum() / n_pos_actual
            else:
                recall = 0.0

            win_rate_oos = y_test.mean()

            fold_result = {
                'fold': fold,
                'train_days': len(train_days),
                'test_days': len(test_days),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'win_rate_oos': win_rate_oos,
            }
            fold_results.append(fold_result)

            all_oos_preds.extend(y_pred.tolist())
            all_oos_actuals.extend(y_test.tolist())
            all_oos_indices.extend(df.loc[test_mask].index.tolist())

            if verbose:
                print(f"\nFold {fold + 1}/{n_folds}:")
                print(f"  Train: {len(train_days)} days, {len(X_train)} samples")
                print(f"  Test:  {len(test_days)} days, {len(X_test)} samples")
                print(f"  OOS WR: {win_rate_oos:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")

        # Summary
        all_oos_preds = np.array(all_oos_preds)
        all_oos_actuals = np.array(all_oos_actuals)

        overall_accuracy = (all_oos_preds == all_oos_actuals).mean()
        overall_win_rate = all_oos_actuals.mean()

        n_filtered = all_oos_preds.sum()
        if n_filtered > 0:
            filtered_win_rate = all_oos_actuals[all_oos_preds == 1].mean()
        else:
            filtered_win_rate = 0.0

        results = {
            'fold_results': fold_results,
            'overall_accuracy': overall_accuracy,
            'overall_win_rate_oos': overall_win_rate,
            'filtered_win_rate_oos': filtered_win_rate,
            'n_total_oos': len(all_oos_preds),
            'n_filtered_oos': n_filtered,
            'oos_predictions': all_oos_preds,
            'oos_actuals': all_oos_actuals,
            'oos_indices': all_oos_indices,
        }

        if verbose:
            print(f"\n{'='*50}")
            print("WALK-FORWARD SUMMARY")
            print(f"{'='*50}")
            print(f"Total OOS samples: {results['n_total_oos']}")
            print(f"Overall OOS win rate: {results['overall_win_rate_oos']:.2%}")
            print(f"Filtered OOS samples: {results['n_filtered_oos']}")
            print(f"Filtered OOS win rate: {results['filtered_win_rate_oos']:.2%}")

            # Check for overfitting
            avg_fold_precision = np.mean([f['precision'] for f in fold_results])
            print(f"\nAvg fold precision: {avg_fold_precision:.2%}")
            if filtered_win_rate < overall_win_rate:
                print("\nWARNING: Filtered win rate < raw win rate!")
                print("ML filtering may not be adding value.")

        self.walk_forward_results = results
        return results

    def train_with_walk_forward(
        self,
        trade_features_df: pd.DataFrame,
        day_col: str = 'trading_day',
        n_folds: int = 5,
        min_train_days: int = 100,
        analyze_correlations: bool = True,
        verbose: bool = True
    ) -> XGBClassifier:
        """
        Train with walk-forward validation (recommended for trading).

        This method:
        1. Optionally analyzes feature correlations
        2. Runs walk-forward CV to estimate out-of-sample performance
        3. Trains final model on all data

        Args:
            trade_features_df: DataFrame with features
            day_col: Column containing trading day
            n_folds: Number of walk-forward folds
            min_train_days: Minimum days for initial training
            analyze_correlations: Whether to check feature correlations
            verbose: Whether to print progress

        Returns:
            Trained XGBClassifier
        """
        if analyze_correlations:
            self.analyze_feature_correlations(trade_features_df, verbose=verbose)

        if verbose:
            print("\n" + "=" * 50)
            print("STEP 1: Walk-Forward Validation")
            print("=" * 50)

        try:
            self.walk_forward_cv(
                trade_features_df,
                day_col=day_col,
                n_folds=n_folds,
                min_train_days=min_train_days,
                verbose=verbose
            )
        except ValueError as e:
            if verbose:
                print(f"Walk-forward CV skipped: {e}")

        if verbose:
            print("\n" + "=" * 50)
            print("STEP 2: Training Final Model on All Data")
            print("=" * 50)

        return self.train(trade_features_df, verbose=verbose)

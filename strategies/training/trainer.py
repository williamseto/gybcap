"""XGBoost model training for trading strategies."""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve


class Trainer:
    """
    XGBoost model trainer for trading strategy prediction.

    Trains a classifier to predict trade success based on features.
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

    def __init__(
        self,
        feature_cols: List[str],
        params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.3,
        augment_data: bool = True,
        min_precision: float = 0.4
    ):
        """
        Initialize trainer.

        Args:
            feature_cols: List of feature column names
            params: XGBoost parameters (overrides defaults)
            test_size: Fraction of data to use for validation
            augment_data: Whether to augment training data
            min_precision: Minimum precision for threshold selection
        """
        self.feature_cols = feature_cols
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.test_size = test_size
        self.augment_data = augment_data
        self.min_precision = min_precision

        self.model: Optional[XGBClassifier] = None
        self.best_threshold: float = 0.5
        self.training_results: Dict[str, Any] = {}

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
            random_state=self.params.get('random_state', 42)
        )

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

        results = {
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'filtered_trades': n_filtered,
            'filtered_pnl': filtered_pnl,
            'avg_pnl_per_trade': total_pnl / len(trades) if trades else 0,
            'avg_filtered_pnl_per_trade': filtered_pnl / n_filtered if n_filtered > 0 else 0,
        }

        if verbose:
            print(f"Total trades: {results['total_trades']}, PnL: {results['total_pnl']:.2f}")
            print(f"Filtered trades: {results['filtered_trades']}, PnL: {results['filtered_pnl']:.2f}")

        return results

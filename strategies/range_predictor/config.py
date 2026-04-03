"""Configuration for ES range predictor."""

from dataclasses import dataclass, field
from typing import List


# Horizons in trading days for each timeframe
TIMEFRAME_HORIZONS = {
    'daily': 1,
    'weekly': 5,
    'monthly': 21,
    'quarterly': 63,
}


@dataclass
class RangePredictorConfig:
    """Configuration for range prediction models."""

    timeframes: List[str] = field(
        default_factory=lambda: ['daily', 'weekly', 'monthly', 'quarterly']
    )
    model_dir: str = 'models/range_predictor'
    data_path: str = 'raw_data/es_min_3y_clean_td_gamma.csv'

    # Feature config
    atr_window: int = 14
    rv_window: int = 21

    # Training config
    walk_forward_folds: int = 5
    min_train_days: int = 100
    test_frac: float = 0.2

    # XGBoost defaults (regression, not classification)
    # Targets are small floats (~0.005), so regularization must be light
    xgb_params: dict = field(default_factory=lambda: {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'n_estimators': 300,
        'max_depth': 4,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0.0,
        'random_state': 27,
    })

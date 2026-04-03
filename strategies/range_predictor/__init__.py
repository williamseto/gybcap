"""ES range prediction module.

Predicts daily/weekly/monthly/quarterly ES price ranges using XGBoost regressors.
Newsletter comparison tools are in the `newsletter` subpackage.
"""

from strategies.range_predictor.config import RangePredictorConfig
from strategies.range_predictor.predictor import RangePredictor

__all__ = ['RangePredictorConfig', 'RangePredictor']

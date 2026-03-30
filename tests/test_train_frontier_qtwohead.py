import pandas as pd

from sandbox.train_frontier_qtwohead_models import _strict_feature_subset


def test_strict_feature_subset_excludes_weight_and_target_columns() -> None:
    rows = pd.DataFrame(
        {
            "minute_of_day": [390, 391],
            "event_base_prob": [0.5, 0.6],
            "tp_now_target": [1.0, 0.0],
            "fail_target": [0.0, 1.0],
            "w_tp": [1.0, 2.0],
            "w_fail": [1.0, 2.0],
            "custom_weight": [1.0, 1.0],
            "feature_a": [0.1, 0.2],
            "feature_b": [1.0, 2.0],
        }
    )
    cols = _strict_feature_subset(rows, list(rows.columns))
    assert "feature_a" in cols
    assert "feature_b" in cols
    assert "event_base_prob" in cols
    assert "tp_now_target" not in cols
    assert "fail_target" not in cols
    assert "w_tp" not in cols
    assert "w_fail" not in cols
    assert "custom_weight" not in cols

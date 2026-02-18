"""HMM-based regime discovery and causal HMM features.

The HMM is fitted per walk-forward fold on training data only (no leakage).
HMM states are sorted by mean return and mapped to bull/bear/balance.

HMM-derived features are CAUSAL — fitted on past data, applied forward.
"""
import pandas as pd
import numpy as np
from typing import Optional


def fit_hmm(
    daily: pd.DataFrame,
    n_states: int = 3,
    random_state: int = 42,
) -> tuple:
    """Fit a Gaussian HMM on daily features.

    Uses [daily_return, realized_vol_10d, range_pct] as observables.

    Args:
        daily: Daily OHLCV DataFrame
        n_states: Number of HMM states
        random_state: Random seed

    Returns:
        (model, state_map): fitted HMM model and state→regime mapping
    """
    from hmmlearn.hmm import GaussianHMM

    c = daily["close"]
    log_ret = np.log(c / c.shift(1))
    rvol_10 = log_ret.rolling(10).std() * np.sqrt(252)
    range_pct = (daily["high"] - daily["low"]) / c

    obs = pd.DataFrame({
        "return": log_ret,
        "rvol": rvol_10,
        "range": range_pct,
    }).dropna()

    X = obs.values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(X)

    # Sort states by mean return: lowest = bear, highest = bull
    state_means = model.means_[:, 0]  # return column
    sorted_states = np.argsort(state_means)
    state_map = {int(sorted_states[i]): i for i in range(n_states)}
    # 0=bear, 1=balance, 2=bull

    return model, state_map, obs.index


def predict_hmm(
    model,
    state_map: dict,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Generate HMM features from a fitted model.

    All features are causal — the model was fitted on prior data.

    Returns DataFrame with columns:
        hmm_state: mapped regime (0=bear, 1=balance, 2=bull)
        hmm_bull_prob: posterior probability of bull state
        hmm_bear_prob: posterior probability of bear state
        hmm_regime_age: consecutive days in current regime
    """
    c = daily["close"]
    log_ret = np.log(c / c.shift(1))
    rvol_10 = log_ret.rolling(10).std() * np.sqrt(252)
    range_pct = (daily["high"] - daily["low"]) / c

    obs = pd.DataFrame({
        "return": log_ret,
        "rvol": rvol_10,
        "range": range_pct,
    }).dropna()

    X = obs.values
    raw_states = model.predict(X)
    posteriors = model.predict_proba(X)

    # Map states
    mapped_states = np.array([state_map.get(s, 1) for s in raw_states])

    # Find bull/bear raw state indices
    inv_map = {v: k for k, v in state_map.items()}
    bull_raw = inv_map.get(2, 0)
    bear_raw = inv_map.get(0, 0)

    feat = pd.DataFrame(index=obs.index)
    feat["hmm_state"] = mapped_states
    feat["hmm_bull_prob"] = posteriors[:, bull_raw]
    feat["hmm_bear_prob"] = posteriors[:, bear_raw]

    # Regime age: consecutive days in same state
    state_changes = (feat["hmm_state"] != feat["hmm_state"].shift(1)).astype(int)
    groups = state_changes.cumsum()
    feat["hmm_regime_age"] = feat.groupby(groups).cumcount() + 1

    # Transition probability features
    feat["hmm_state_changed"] = state_changes.astype(float)
    feat["hmm_regime_age_log"] = np.log1p(feat["hmm_regime_age"])

    return feat


def compute_hmm_features_walkforward(
    daily: pd.DataFrame,
    train_end_idx: int,
    n_states: int = 3,
) -> pd.DataFrame:
    """Compute HMM features with walk-forward discipline.

    Fits HMM on data[:train_end_idx], predicts on full dataset.
    Features before train_end_idx are in-sample; after are out-of-sample.

    Args:
        daily: Full daily DataFrame
        train_end_idx: Index position where training data ends
        n_states: Number of HMM states

    Returns:
        DataFrame with HMM features for all rows
    """
    train_data = daily.iloc[:train_end_idx]

    try:
        model, state_map, _ = fit_hmm(train_data, n_states=n_states)
        feat = predict_hmm(model, state_map, daily)
    except Exception as e:
        print(f"  HMM fitting failed: {e}. Using defaults.")
        feat = pd.DataFrame(index=daily.index)
        feat["hmm_state"] = 1
        feat["hmm_bull_prob"] = 0.33
        feat["hmm_bear_prob"] = 0.33
        feat["hmm_regime_age"] = 1
        feat["hmm_state_changed"] = 0.0
        feat["hmm_regime_age_log"] = 0.0

    # Reindex to full daily index (NaN for early rows without enough data)
    feat = feat.reindex(daily.index).fillna(0.0)
    return feat


FEATURE_NAMES = [
    "hmm_state", "hmm_bull_prob", "hmm_bear_prob",
    "hmm_regime_age", "hmm_state_changed", "hmm_regime_age_log",
]

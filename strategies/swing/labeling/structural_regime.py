"""Structural regime labels from swing high/low detection.

Labels are NON-CAUSAL (use future data) — they define ground truth for what
the regime actually was. The model's features are strictly causal.

Three label types:
  - Macro regime (BULL/BEAR/BALANCE): from swing structure over weeks/months
  - Micro regime (UP/DOWN/BALANCE): from next-day return with dead zone
  - Structural regime (BULL/BEAR/BALANCE): from percentage-change zigzag
"""
import pandas as pd
import numpy as np

from strategies.swing.config import MacroRegime, MicroRegime


def detect_swing_points(
    daily: pd.DataFrame, lookback: int = 10
) -> tuple[pd.Series, pd.Series]:
    """Detect swing highs and lows using N-bar window.

    A swing high at bar t means high[t] == max(high[t-N:t+N+1]).
    A swing low at bar t means low[t] == min(low[t-N:t+N+1]).

    These are NON-CAUSAL — they use future bars for confirmation.

    Args:
        daily: Daily OHLCV DataFrame
        lookback: N bars on each side for window

    Returns:
        swing_highs: Series of bool (True at swing high days)
        swing_lows: Series of bool (True at swing low days)
    """
    highs = daily["high"].values
    lows = daily["low"].values
    n = len(highs)

    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback: i + lookback + 1]
        window_low = lows[i - lookback: i + lookback + 1]

        if highs[i] == window_high.max():
            is_swing_high[i] = True
        if lows[i] == window_low.min():
            is_swing_low[i] = True

    return (
        pd.Series(is_swing_high, index=daily.index),
        pd.Series(is_swing_low, index=daily.index),
    )


def assign_macro_regime(
    daily: pd.DataFrame, lookback: int = 10
) -> pd.Series:
    """Assign macro regime labels based on swing high/low structure.

    Logic:
      - Track the most recent swing high price and swing low price
      - If the latest swing low is higher than the previous swing low
        AND price is above the latest swing low → BULL
      - If the latest swing high is lower than the previous swing high
        AND price is below the latest swing high → BEAR
      - Otherwise → BALANCE

    This is non-causal because swing detection uses future bars.

    Returns:
        Series of MacroRegime values (0=BEAR, 1=BALANCE, 2=BULL)
    """
    swing_highs, swing_lows = detect_swing_points(daily, lookback)

    n = len(daily)
    regime = np.full(n, MacroRegime.BALANCE, dtype=int)

    # Collect swing high/low prices and their indices
    sh_idx = np.where(swing_highs.values)[0]
    sl_idx = np.where(swing_lows.values)[0]

    highs = daily["high"].values
    lows = daily["low"].values
    closes = daily["close"].values

    # For each day, find the two most recent swing highs and two most recent swing lows
    for i in range(n):
        # Recent swing lows before or at i
        past_sl = sl_idx[sl_idx <= i]
        past_sh = sh_idx[sh_idx <= i]

        if len(past_sl) < 2 or len(past_sh) < 2:
            continue

        # Most recent two swing lows
        sl_curr = lows[past_sl[-1]]
        sl_prev = lows[past_sl[-2]]

        # Most recent two swing highs
        sh_curr = highs[past_sh[-1]]
        sh_prev = highs[past_sh[-2]]

        # Higher lows = uptrend
        higher_lows = sl_curr > sl_prev
        # Lower highs = downtrend
        lower_highs = sh_curr < sh_prev

        if higher_lows and not lower_highs:
            regime[i] = MacroRegime.BULL
        elif lower_highs and not higher_lows:
            regime[i] = MacroRegime.BEAR
        else:
            regime[i] = MacroRegime.BALANCE

    return pd.Series(regime, index=daily.index, name="macro_regime")


def assign_micro_regime(
    daily: pd.DataFrame, threshold_pct: float = 0.003
) -> pd.Series:
    """Assign next-day micro regime labels.

    Uses day t+1 return to label day t:
      return > threshold  → UP
      return < -threshold → DOWN
      otherwise           → BALANCE

    This is non-causal — it uses the next day's return.

    Args:
        daily: Daily OHLCV DataFrame
        threshold_pct: Dead-zone threshold (default 0.3%)

    Returns:
        Series of MicroRegime values (0=DOWN, 1=BALANCE, 2=UP), NaN for last day
    """
    next_day_return = daily["close"].pct_change().shift(-1)

    regime = pd.Series(MicroRegime.BALANCE, index=daily.index, dtype=int, name="micro_regime")
    regime[next_day_return > threshold_pct] = MicroRegime.UP
    regime[next_day_return < -threshold_pct] = MicroRegime.DOWN

    # Last day has no next-day return — mark as NaN
    regime.iloc[-1] = -1  # will be filtered out in training

    return regime


def find_zigzag_turning_points(close: np.ndarray, pct_threshold: float = 0.05) -> list[dict]:
    """Find peaks/troughs where price reverses by at least pct_threshold.

    Global, forward-looking — no fixed window. Non-causal by design.

    Walks forward tracking running highs/lows:
    - A peak is confirmed when price drops pct_threshold from the running high.
    - A trough is confirmed when price rises pct_threshold from the running low.

    Args:
        close: Array of close prices.
        pct_threshold: Minimum reversal percentage to confirm a turning point.

    Returns:
        List of dicts with keys: idx, price, type ('peak' or 'trough').
    """
    n = len(close)
    if n < 2:
        return []

    # Start by looking for first direction
    turning_points = []
    # Initialize: determine initial direction from first move
    state = "seeking"  # 'seeking_peak' or 'seeking_trough'
    high_idx = 0
    low_idx = 0

    # Find initial direction
    for i in range(1, n):
        if close[i] / close[low_idx] - 1.0 >= pct_threshold:
            # Price rose enough from low — the low was a trough, now seek peak
            turning_points.append({"idx": low_idx, "price": close[low_idx], "type": "trough"})
            state = "seeking_peak"
            high_idx = i
            break
        elif 1.0 - close[i] / close[high_idx] >= pct_threshold:
            # Price fell enough from high — the high was a peak, now seek trough
            turning_points.append({"idx": high_idx, "price": close[high_idx], "type": "peak"})
            state = "seeking_trough"
            low_idx = i
            break
        # Track running extremes
        if close[i] > close[high_idx]:
            high_idx = i
        if close[i] < close[low_idx]:
            low_idx = i

    if state == "seeking":
        return []

    # Main loop
    for i in range(turning_points[-1]["idx"] + 1, n):
        if state == "seeking_peak":
            if close[i] > close[high_idx]:
                high_idx = i
            # Check for reversal down from peak
            if close[high_idx] > 0 and 1.0 - close[i] / close[high_idx] >= pct_threshold:
                turning_points.append({"idx": high_idx, "price": close[high_idx], "type": "peak"})
                state = "seeking_trough"
                low_idx = i
        elif state == "seeking_trough":
            if close[i] < close[low_idx]:
                low_idx = i
            # Check for reversal up from trough
            if close[low_idx] > 0 and close[i] / close[low_idx] - 1.0 >= pct_threshold:
                turning_points.append({"idx": low_idx, "price": close[low_idx], "type": "trough"})
                state = "seeking_peak"
                high_idx = i

    return turning_points


def assign_pct_regime(
    daily: pd.DataFrame,
    detect_threshold: float = 0.05,
    bull_threshold: float = 0.10,
    bear_threshold: float = 0.07,
) -> pd.Series:
    """Tiered zigzag regime: classify segments by magnitude.

    BULL: trough→peak rally > bull_threshold
    BEAR: peak→trough drawdown > bear_threshold
    BALANCE: segments smaller than bull/bear thresholds

    Non-causal by design — uses global zigzag structure.

    Returns:
        Series of MacroRegime values (0=BEAR, 1=BALANCE, 2=BULL)
    """
    close = daily["close"].values
    tps = find_zigzag_turning_points(close, detect_threshold)

    n = len(daily)
    regime = np.full(n, MacroRegime.BALANCE, dtype=int)

    if len(tps) < 2:
        return pd.Series(regime, index=daily.index, name="structural_regime")

    # Classify each segment between consecutive turning points
    for i in range(len(tps) - 1):
        tp_start = tps[i]
        tp_end = tps[i + 1]
        seg_start = tp_start["idx"]
        seg_end = tp_end["idx"]

        pct_change = (tp_end["price"] - tp_start["price"]) / tp_start["price"]

        if tp_start["type"] == "trough" and tp_end["type"] == "peak":
            # Rally segment
            if pct_change > bull_threshold:
                regime[seg_start: seg_end + 1] = MacroRegime.BULL
            # else stays BALANCE
        elif tp_start["type"] == "peak" and tp_end["type"] == "trough":
            # Drawdown segment
            if abs(pct_change) > bear_threshold:
                regime[seg_start: seg_end + 1] = MacroRegime.BEAR
            # else stays BALANCE

    # Classify tail segment from last turning point to end of series
    if len(tps) >= 1:
        last_tp = tps[-1]
        last_idx = last_tp["idx"]
        last_price = last_tp["price"]
        end_price = close[-1]
        pct_move = (end_price - last_price) / last_price

        if last_tp["type"] == "trough" and pct_move > bull_threshold:
            regime[last_idx:] = MacroRegime.BULL
        elif last_tp["type"] == "peak" and abs(pct_move) > bear_threshold:
            regime[last_idx:] = MacroRegime.BEAR
        else:
            regime[last_idx:] = MacroRegime.BALANCE

    return pd.Series(regime, index=daily.index, name="structural_regime")


def compute_labels(
    daily: pd.DataFrame,
    swing_lookback: int = 10,
    micro_threshold_pct: float = 0.003,
    detect_threshold: float = 0.05,
    bull_threshold: float = 0.10,
    bear_threshold: float = 0.07,
) -> pd.DataFrame:
    """Compute macro, micro, and structural regime labels.

    Returns DataFrame with columns: y_macro, y_micro, y_structural
    """
    macro = assign_macro_regime(daily, swing_lookback)
    micro = assign_micro_regime(daily, micro_threshold_pct)
    structural = assign_pct_regime(daily, detect_threshold, bull_threshold, bear_threshold)

    labels = pd.DataFrame({
        "y_macro": macro,
        "y_micro": micro,
        "y_structural": structural,
    }, index=daily.index)

    return labels

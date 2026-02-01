"""
breakout_retest_mplfinance.py

Produces an mplfinance candlestick plot of HTF bars with:
 - previous-day levels (high / low / midpoint)
 - trade entries (triangles) and exits (x)
 - optional zoom range

Requires: pandas, numpy, mplfinance

Run: python breakout_retest_mplfinance.py
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import sys
import pickle
import talib
import numba as nb

sys.path.append('../gex')
import gex_utils

# ---------------------------
# Data helper (synthetic / or load CSV)
# ---------------------------

def generate_synthetic_ohlcv(start='2025-07-01', days=20, freq='1min', seed=42):
    np.random.seed(seed)
    idx = pd.date_range(start=start, periods=days*24*60, freq=freq)
    r = 0.00001 + 0.0005 * np.random.randn(len(idx))
    price = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame(index=idx)
    df['close'] = price
    df['open']  = df['close'].shift(1).fillna(df['close'])
    df['high']  = np.maximum(df['open'], df['close']) + np.abs(0.01 * price * np.random.rand(len(idx)))
    df['low']   = np.minimum(df['open'], df['close']) - np.abs(0.01 * price * np.random.rand(len(idx)))
    df['volume']= (100 * np.random.rand(len(idx))).astype(int)
    return df[['open','high','low','close','volume']]

def load_your_data(csv_path: str) -> pd.DataFrame:
    """
    Replace with your CSV loader if you have one.
    Ensure the CSV has a datetime column or index and columns: open,high,low,close,volume
    Example:
      df = pd.read_csv('mydata.csv', parse_dates=['datetime_col'], index_col='datetime_col')
    """
    return generate_synthetic_ohlcv()  # placeholder for demo


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    dt_format_str = "%m/%d/%Y %H:%M:%S"
    df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format=dt_format_str)

    df.columns = df.columns.str.lower()

    # min15_df = df.set_index('dt').groupby('trading_day').resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()

    return df

# ---------------------------
# Price level provider (previous day high/low/mid)
# ---------------------------


class PriceLevelProvider:
    def __init__(self, ohlcv: pd.DataFrame):
        self.ohlcv = ohlcv
        # if not isinstance(self.ohlcv.index, pd.DatetimeIndex):
        #     raise ValueError("ohlcv must have a DatetimeIndex")


        self.ohlcv_feat = self.ohlcv.groupby('trading_day').apply(self.compute_bar_stats, include_groups=False).reset_index()

    def prev_day_levels(self) -> pd.DataFrame:

        daily = self.ohlcv.groupby('trading_day').agg({'open':'first','high':'max','low':'min','close':'last'})

        # daily = self.ohlcv.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last'})
        daily = daily[['high','low']].rename(columns={'high':'day_high','low':'day_low'})
        prev = daily.shift(1).rename(columns={'day_high':'prev_high','day_low':'prev_low'})
        prev['prev_mid'] = (prev['prev_high'] + prev['prev_low']) / 2.0
        return prev[['prev_high','prev_low','prev_mid']]


    def compute_bar_stats(self, group) -> pd.DataFrame:

        # group["sma5"] = talib.SMA(group["close"], timeperiod=5)
        group["rsi"] = talib.RSI(group["close"], timeperiod=14)

        ovn_hi = group[group['ovn'] == 1]['high'].max()
        ovn_lo = group[group['ovn'] == 1]['low'].min()

        group['ovn_hi'] = ovn_hi
        group['ovn_lo'] = ovn_lo

        avg_price = (group["open"] + group["high"] + group["low"] + group["close"]).div(4)
        vwap = avg_price.mul(group["volume"]).cumsum().div(group["volume"].cumsum())
        vwap_std = avg_price.sub(vwap).pow(2).cumsum().div(np.arange(1,len(vwap)+1)).apply(np.sqrt).clip(lower=1e-6)

        group['vwap'] = vwap
        group['vwap_z'] = (group['close'] - vwap) / vwap_std


        sma20 = talib.SMA(group["close"], timeperiod=20)
        std20 = group['close'].rolling(window=20).std().fillna(0.000001)

        

        group['close_z20'] = (group['close'] - sma20) / std20
        group['ovn_lo_z'] = (group['close'] - ovn_lo) / vwap_std
        group['ovn_hi_z'] = (ovn_hi - group['close']) / vwap_std

        time_df = group.set_index("dt")

        rth_df = time_df.between_time('6:30', '12:59')

        ib_df = time_df.between_time('6:30', '7:30')


        ib_lo = ib_df['low'].min()
        ib_hi = ib_df['high'].max()

        ib_lo_z = (group['close'] - ib_lo) / vwap_std
        ib_hi_z = (ib_hi - group['close']) / vwap_std

        group['ib_lo_z'] = 0.0
        group['ib_hi_z'] = 0.0


        rth_after_ib_idx = group.index[time_df.index.indexer_between_time('7:30', '12:59')]


        group.loc[rth_after_ib_idx, 'ib_lo_z'] = ib_lo_z.loc[rth_after_ib_idx]
        group.loc[rth_after_ib_idx, 'ib_hi_z'] = ib_hi_z.loc[rth_after_ib_idx]


        group['rth_lo'] = 0.0
        group['rth_hi'] = 0.0
        group.loc[rth_after_ib_idx, 'rth_lo'] = group.loc[rth_after_ib_idx, 'low'].cummin()
        group.loc[rth_after_ib_idx, 'rth_hi'] = group.loc[rth_after_ib_idx, 'high'].cummax()


        group["rsi"] = talib.RSI(group["close"], timeperiod=14)

        vol_mean = group['volume'].rolling(window=20, center=True, min_periods=1).mean()
        vol_std  = group['volume'].rolling(window=20, center=True, min_periods=1).std().add(1e-6)

        group['vol_z'] = (group['volume'] - vol_mean) / vol_std

        group['adx'] = talib.ADX(group['high'], group['low'], group['close'], timeperiod=14)


        ofi_pct = (group['askvolume'] - group['bidvolume']) / (group['askvolume'] + group['bidvolume'])
        window = 60  # minutes or bars
        ofi_pct_series = ofi_pct.fillna(0)
        ofi_z = (ofi_pct_series - ofi_pct_series.rolling(window).mean()) / ofi_pct_series.rolling(window).std()
        group['ofi_z'] = ofi_z


        # def get_vol_percentile(x):

        #     if len(x) > 1320:
        #         return 0

        #     price_vol = group[['close', 'volume']].iloc[:len(x)+1]

        #     volume_profile = price_vol.groupby("close").agg({"volume": "sum"}).reset_index()

        #     close_idx = volume_profile[volume_profile['close'] == x.iloc[-1]].index

        #     window_size = 3
        #     window_min_idx = max(close_idx[0]-window_size, 0)
        #     window_max_idx = min(close_idx[0]+window_size, len(volume_profile))

        #     vp_window = volume_profile.iloc[window_min_idx:window_max_idx]

        #     return (vp_window["volume"].sum() / volume_profile["volume"].sum())


        # group['vol_percentile'] = group['close'].expanding(min_periods=930).apply(get_vol_percentile)

        return group

    def attach_levels_to_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        levels = self.prev_day_levels()

        # bars = bars.copy()
        # bar_days = bars.index.floor('D')

        bars['prev_high'] = bars['trading_day'].map(levels['prev_high'])
        bars['prev_low']  = bars['trading_day'].map(levels['prev_low'])
        bars['prev_mid']  = bars['trading_day'].map(levels['prev_mid'])


        # self.level_cols = ['vwap','ovn_lo','ovn_hi', 'ovn']
        self.feat_cols = ['close_z20','ovn_lo_z','ovn_hi_z','ib_lo_z','ib_hi_z', 'vwap_z', 'rsi', 'vol_z', 'adx', 'ofi_z', 'nearby_gamma_score']
        feat_1m = self.ohlcv_feat[['dt','vwap','ovn_lo','ovn_hi', 'ovn', 'rth_lo', 'rth_hi'] + self.feat_cols]

        bars = pd.merge(bars,
                        feat_1m,
                        left_on=['dt'],
                        right_on=['dt'],
                        how='left')


        # map by day
        # bars['prev_high'] = bar_days.map(levels['prev_high'])
        # bars['prev_low']  = bar_days.map(levels['prev_low'])
        # bars['prev_mid']  = bar_days.map(levels['prev_mid'])

        return bars

# ---------------------------
# Strategy (breakout detection + retest -> trades)
# ---------------------------

@dataclass
class Trade:
    entry_ts: pd.Timestamp
    entry_price: float
    direction: str
    size: float
    stop: float
    take: Optional[float]
    exit_ts: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

class BreakoutRetestStrategy:
    def __init__(self, bars: pd.DataFrame,
                 level_cols=['prev_high','prev_low','prev_mid'],
                 threshold_pct=0.0012,
                 lookahead_bars=40):
        self.bars = bars.copy()


        self.level_cols = level_cols
        self.threshold_pct = threshold_pct
        self.lookahead_bars = lookahead_bars

    def detect_breakouts(self):

        events = []
        c = self.bars['close']
        prev_close = c.shift(1)

        rth_mask = self.bars['ovn'] == 0


        for lvl in self.level_cols:
            if lvl not in self.bars.columns:
                continue
            L = self.bars[lvl]

            bull_mask = (prev_close <= L) & (c >= L) & rth_mask
            bear_mask = (prev_close >= L) & (c <= L) & rth_mask
            for ts in self.bars.index[bull_mask.fillna(False)]:
                events.append((lvl, float(L.loc[ts]), ts, 'bull'))
            for ts in self.bars.index[bear_mask.fillna(False)]:
                events.append((lvl, float(L.loc[ts]), ts, 'bear'))
        events.sort(key=lambda e: e[2])

        return events

    def find_retest_and_build_trades(self, stop_buffer_pct=0.0025, rr=1.5, fixed_size=1.0):
        trades = []
        events = self.detect_breakouts()

        bar_index = self.bars.index

        trade_features = []
        trade_succ = []
        trade_bear = []
        for lvl_name, L, breakout_ts, direction in events:

            try:
                idx0 = bar_index.get_loc(breakout_ts)
            except KeyError:
                continue
            look_slice = self.bars.iloc[idx0+1: idx0+1 + self.lookahead_bars]
            if look_slice.empty:
                continue

            # ToDo: try to optimize this threshold
            risk = min(stop_buffer_pct * L, 10.0)

            threshold = min(self.threshold_pct * L, risk * 0.5)
            distances = (look_slice['close'] - L).abs()

            hits = distances <= threshold
            if not hits.any():
                continue
            hit_idx = hits.idxmax()
            entry_price = float(look_slice.loc[hit_idx, 'close'])

            if direction == 'bull':
                # stop = L - stop_buffer
                # risk = entry_price - stop
                stop = entry_price - risk
                take = entry_price + rr * risk
            else:
                # stop = L + stop_buffer
                # risk = stop - entry_price
                stop = entry_price + risk
                take = entry_price - rr * risk
            trade = Trade(entry_ts=hit_idx, entry_price=entry_price, direction=direction,
                          size=fixed_size, stop=stop, take=take)

            # simulate exit (simple intrabar hi/lo check)
            exit_price = None
            exit_ts = None
            for ts, row in self.bars.loc[hit_idx:].iterrows():
                hi, lo = row['high'], row['low']
                if direction == 'bull':
                    if lo <= stop:
                        exit_price = stop; exit_ts = ts; break
                    if hi >= take:
                        exit_price = take; exit_ts = ts; break
                else:
                    if hi >= stop:
                        exit_price = stop; exit_ts = ts; break
                    if lo <= take:
                        exit_price = take; exit_ts = ts; break


            if exit_price is None:
                exit_ts = self.bars.index[-1]
                exit_price = float(self.bars.loc[exit_ts, 'close'])
            trade.exit_ts = exit_ts
            trade.exit_price = exit_price
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.size if direction == 'bull' else (trade.entry_price - trade.exit_price) * trade.size
            trades.append(trade)


            trade_row = self.bars.loc[hit_idx]
            trade_features.append(trade_row)

            if trade.pnl > 0:
                trade_succ.append(1)
            else:
                trade_succ.append(0)

            if direction == 'bear':
                trade_bear.append(1)
            else:
                trade_bear.append(0)


        trade_features_df = pd.DataFrame(trade_features).reset_index(drop=True)

        trade_features_df['y_succ'] = trade_succ
        trade_features_df['bear'] = trade_bear

        return trades, trade_features_df


class ReversionStrategy:
    def __init__(self, bars: pd.DataFrame,
                 level_cols=['prev_high','prev_low','prev_mid'],
                 threshold_pct=0.0012,
                 lookahead_bars=40):
        self.bars = bars.copy()


        self.level_cols = level_cols
        self.threshold_pct = threshold_pct
        self.lookahead_bars = lookahead_bars

    def get_trigger_events(self):

        events = []

        rth_mask = self.bars['ovn'] == 0

        for lvl in self.level_cols:
            if lvl not in self.bars.columns:
                continue
            L = self.bars[lvl]

            bull_mask = (self.bars['open'] > L) & (self.bars['low'] <= L) & (self.bars['close'] > L) & rth_mask
            bear_mask = (self.bars['open'] < L) & (self.bars['high'] >= L) & (self.bars['close'] < L) & rth_mask
            for ts in self.bars.index[bull_mask.fillna(False)]:
                events.append((lvl, float(L.loc[ts]), ts, 'bull'))
            for ts in self.bars.index[bear_mask.fillna(False)]:
                events.append((lvl, float(L.loc[ts]), ts, 'bear'))
        events.sort(key=lambda e: e[2])
        return events

    def find_retest_and_build_trades(self, stop_buffer_pct=0.0025, rr=1.5, fixed_size=1.0):
        trades = []
        events = self.get_trigger_events()
        bar_index = self.bars.index

        trade_features = []
        trade_succ = []
        trade_bear = []
        for lvl_name, L, breakout_ts, direction in events:

            try:
                idx0 = bar_index.get_loc(breakout_ts)
            except KeyError:
                continue
            look_slice = self.bars.iloc[idx0+1: idx0+1 + self.lookahead_bars]
            if look_slice.empty:
                continue

            # ToDo: try to optimize this threshold
            risk = min(stop_buffer_pct * L, 10.0)

            threshold = min(self.threshold_pct * L, risk * 0.5)
            distances = (look_slice['close'] - L).abs()

            hits = distances <= threshold
            if not hits.any():
                continue
            hit_idx = hits.idxmax()
            entry_price = float(look_slice.loc[hit_idx, 'close'])

            if direction == 'bull':
                # stop = L - stop_buffer
                # risk = entry_price - stop
                stop = entry_price - risk
                take = entry_price + rr * risk
            else:
                # stop = L + stop_buffer
                # risk = stop - entry_price
                stop = entry_price + risk
                take = entry_price - rr * risk
            trade = Trade(entry_ts=hit_idx, entry_price=entry_price, direction=direction,
                          size=fixed_size, stop=stop, take=take)

            # simulate exit (simple intrabar hi/lo check)
            exit_price = None
            exit_ts = None
            for ts, row in self.bars.loc[hit_idx:].iterrows():
                hi, lo = row['high'], row['low']
                if direction == 'bull':
                    if lo <= stop:
                        exit_price = stop; exit_ts = ts; break
                    if hi >= take:
                        exit_price = take; exit_ts = ts; break
                else:
                    if hi >= stop:
                        exit_price = stop; exit_ts = ts; break
                    if lo <= take:
                        exit_price = take; exit_ts = ts; break


            if exit_price is None:
                exit_ts = self.bars.index[-1]
                exit_price = float(self.bars.loc[exit_ts, 'close'])
            trade.exit_ts = exit_ts
            trade.exit_price = exit_price
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.size if direction == 'bull' else (trade.entry_price - trade.exit_price) * trade.size
            trades.append(trade)


            trade_row = self.bars.loc[hit_idx]
            trade_features.append(trade_row)

            if trade.pnl > 0:
                trade_succ.append(1)
            else:
                trade_succ.append(0)

            if direction == 'bear':
                trade_bear.append(1)
            else:
                trade_bear.append(0)


        trade_features_df = pd.DataFrame(trade_features).reset_index(drop=True)

        trade_features_df['y_succ'] = trade_succ
        trade_features_df['bear'] = trade_bear

        return trades, trade_features_df

# ---------------------------
# Plot with mplfinance
# ---------------------------

def plot_with_mplfinance(bars: pd.DataFrame, trades: List[Trade], plot_range: Optional[tuple]=None):
    """
    bars: HTF bars (DatetimeIndex, with open/high/low/close/volume and prev_high/prev_low/prev_mid)
    trades: list of Trade dataclass instances
    plot_range: optional tuple(start_dt, end_dt) to zoom (strings or pd.Timestamp)
    """
    plot_bars = bars.copy()
    if plot_range is not None:
        start, end = pd.to_datetime(plot_range[0]), pd.to_datetime(plot_range[1])
        plot_bars = plot_bars.loc[start:end]

    # Prepare addplots:
    ap = []

    # 1) previous-day levels as lines (these series are NaN for first day)
    for col, style in [('prev_high','--'), ('prev_low','--'), ('prev_mid',':')]:
        if col in plot_bars.columns:
            # make a full-length series for mplfinance
            s = plot_bars[col].astype(float)
            ap.append(mpf.make_addplot(s, type='line', linestyle=style, width=0.8))

    # 2) entries and exits as scatter series
    entry_s = pd.Series(np.nan, index=plot_bars.index)
    entry_colors = []
    exit_s  = pd.Series(np.nan, index=plot_bars.index)
    entry_markers = []
    exit_markers = []
    for t in trades:
        if t.entry_ts in entry_s.index:
            entry_s.loc[t.entry_ts] = t.entry_price
        # exits might be outside plot_range; only map if present
        if t.exit_ts in exit_s.index:
            exit_s.loc[t.exit_ts] = t.exit_price

    if entry_s.notna().any():
        ap.append(mpf.make_addplot(entry_s, type='scatter', markersize=100, marker='^', color='g'))
    if exit_s.notna().any():
        ap.append(mpf.make_addplot(exit_s, type='scatter', markersize=80, marker='x', color='r'))

    # 3) optional connecting lines (entry->exit): mplfinance doesn't accept per-point line adds easily,
    # so we draw connecting lines on the matplotlib Axes after mpf.plot returns the axes.
    fig, axes = mpf.plot(plot_bars,
                         type='candle',
                         volume=True,
                         addplot=ap,
                         returnfig=True,
                         figscale=1.3,
                         title='Breakout + Retest (candles) with prev-day levels and trades',
                         style='yahoo')

    ax = axes[0]  # price axes
    # draw connecting lines for trades that lie inside the plotted window
    for t in trades:
        if (t.entry_ts in plot_bars.index) and (t.exit_ts in plot_bars.index):
            ax.plot([t.entry_ts, t.exit_ts], [t.entry_price, t.exit_price], linewidth=0.8, alpha=0.9, zorder=2)

    plt.show()

def simple_plot(bars: pd.DataFrame, trades: List[Trade]):
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(bars.index, bars['close'], label='close', linewidth=1)

    # draw prev-day levels (one horizontal per day)
    days = sorted(set(bars.index.floor('D')))
    first_label_plotted = set()
    for d in days:
        day_mask = (bars.index.floor('D') == d)
        if not day_mask.any(): continue
        start = bars.index[day_mask].min()
        end   = bars.index[day_mask].max()
        ph = bars.loc[day_mask, 'prev_high'].iloc[0]
        pl = bars.loc[day_mask, 'prev_low'].iloc[0]
        pm = bars.loc[day_mask, 'prev_mid'].iloc[0]
        if not pd.isna(ph):
            lbl = 'prev_high' if 'prev_high' not in first_label_plotted else None
            ax.hlines(ph, start, end, linewidth=1, label=lbl); first_label_plotted.add('prev_high')
        if not pd.isna(pl):
            lbl = 'prev_low' if 'prev_low' not in first_label_plotted else None
            ax.hlines(pl, start, end, linewidth=1, label=lbl); first_label_plotted.add('prev_low')
        if not pd.isna(pm):
            lbl = 'prev_mid' if 'prev_mid' not in first_label_plotted else None
            ax.hlines(pm, start, end, linewidth=1, label=lbl); first_label_plotted.add('prev_mid')

    # plot trades
    first_entry_plotted = False
    first_exit_plotted  = False
    for t in trades:
        mk = '^' if t.direction == 'bull' else 'v'
        ax.scatter([t.entry_ts], [t.entry_price], marker=mk, s=80, label='entry' if not first_entry_plotted else None)
        first_entry_plotted = True
        ax.scatter([t.exit_ts], [t.exit_price], marker='x', s=80, label='exit' if not first_exit_plotted else None)
        first_exit_plotted = True
        ax.plot([t.entry_ts, t.exit_ts], [t.entry_price, t.exit_price], linewidth=0.7)

    ax.set_title('Breakout + Retest: price, prev-day levels, trade entries/exits')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def get_train_model(trade_features_df, feature_cols):

    trade_features_df = trade_features_df.fillna(0)

    # X = trade_features_df[feature_cols].values
    # y = trade_features_df['y_succ'].values

    X = trade_features_df[feature_cols]
    y = trade_features_df['y_succ']

    augmented_df = trade_features_df[trade_features_df['y_succ']==1] #trade_features_df.sample(frac=0.95, random_state=42)

    augmented_X = augmented_df[feature_cols]
    augmented_X.loc[:, 'bear'] = 1 - augmented_X.loc[:, 'bear']

    augmented_y = augmented_df['y_succ']
    augmented_y = 1 - augmented_y

    combined_X = pd.concat([X, augmented_X], axis=0)
    combined_y = pd.concat([y, augmented_y], axis=0)

    X = combined_X.values
    y = combined_y.values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)



    model = XGBClassifier(
        eval_metric="auc",
        scale_pos_weight=3.0,
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=5,
        gamma=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        seed=27
    )
    
    model.fit(X_train, y_train)



    y_proba = model.predict_proba(X_val)[:,1]




    # --- Step 2: predict probabilities on validation ---
    # dval = xgb.DMatrix(X_val, label=y_val)
    # y_proba = bst.predict(dval)


    # --- Step 3: precision-recall curve ---
    prec, rec, thresholds = precision_recall_curve(y_val, y_proba)

    # np.set_printoptions(threshold=sys.maxsize)
    # print(prec)
    # print(rec)
    # print(thresholds)


    # choose the threshold with recall as high as possible
    # subject to a minimum precision constraint
    min_precision = 0.4   # <-- tune this depending on your tolerance
    valid = np.where(prec >= min_precision)[0]

    half_threshold_idx = np.where(thresholds >= 0.5)[0][0]

    if len(valid) > 0:
        best_idx = valid[np.argmax(rec[valid])]
        if best_idx >= len(thresholds):
            best_idx = len(thresholds) - 1
        best_thresh = thresholds[best_idx]
    else:
        best_thresh = 0.5  # fallback if constraint can’t be met

    print(f"Chosen threshold: {best_thresh:.3f}")
    print(f"Recall at threshold: {rec[best_idx]:.3f}")
    print(f"Precision at threshold: {prec[best_idx]:.3f}")

    print(f"Recall at 0.5 threshold: {rec[half_threshold_idx]:.3f}")
    print(f"Precision at 0.5 threshold: {prec[half_threshold_idx]:.3f}")


    # X_train = trade_features_df[feature_cols].values
    # y_train = trade_features_df['y_succ'].values
    # model = XGBClassifier(
    #     # n_estimators=69,
    #     # max_depth=2,
    #     # min_child_weight=5,
    #     # reg_alpha=0.1,
    #     # objective= 'binary:logistic',
    #     eval_metric="auc",
    #     scale_pos_weight=5.0
    # )
    # model.fit(X_train, y_train)

    # print(model.get_params())


    

    
   

    return model

def modelfit(alg, X, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)

        print('best n_estimators: ', cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X, y)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
                    
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    # plt.show()

    return alg

# ---------------------------
# Demo driver
# ---------------------------
def main():
    # load/generate data
    # df = load_your_data(None)  # replace None with your csv path if desired

    # # resample to higher timeframe (e.g., 15 min)
    # tf = '15T'
    # bars = df.resample(tf).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

    # df2 = load_data('../es_min_20220911_20250905_td.csv')
    df2 = load_data('../raw_data/es_min_3y_clean_td_gamma.csv')

    agg_dict = {
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'volume':'sum'
    }
    bars = df2.set_index('dt').groupby('trading_day').resample('15min').agg(agg_dict).dropna().reset_index()

    # attach prev-day levels
    plp = PriceLevelProvider(df2)
    bars = plp.attach_levels_to_bars(bars)


    strat = BreakoutRetestStrategy(bars, level_cols=['prev_high','prev_low','vwap', 'ovn_lo', 'ovn_hi', 'ib_lo', 'ib_hi', 'rth_lo', 'rth_hi'],
                                   threshold_pct=0.0012, lookahead_bars=10)
    trades, trade_features_df = strat.find_retest_and_build_trades(stop_buffer_pct=0.0025, rr=2.0, fixed_size=1.0)

    # strat = ReversionStrategy(bars, level_cols=['prev_high','prev_low','vwap', 'ovn_lo', 'ovn_hi', 'ib_lo', 'ib_hi','rth_lo', 'rth_hi'],
    #                                threshold_pct=0.0012, lookahead_bars=10)
    # trades, trade_features_df = strat.find_retest_and_build_trades(stop_buffer_pct=0.0025, rr=2.0, fixed_size=1.0)


    total_pnl = 0
    for trade in trades:
        total_pnl += trade.pnl
    print(f"Total PNL: {total_pnl}")
    print(f"Total Trades: {len(trades)}")
    print(f"number of trading days: {max(bars['trading_day'])}")

    train_pnl = 0
    train_trades = 0
    feature_cols = plp.feat_cols + ['bear']
    model = get_train_model(trade_features_df, feature_cols)
    for idx, row in trade_features_df.iterrows():
        pred = model.predict(row[feature_cols].values.reshape(1, -1))

        # pred_proba = model.predict_proba(row[feature_cols].values.reshape(1, -1))

        if pred[0] == 1:
        # if pred_proba[0][1] >= 0.3:
            train_pnl += trades[idx].pnl
            train_trades += 1

    print(f"Train PNL: {train_pnl} trades: {train_trades}")

    model.save_model('bo_retest_model.json')
    

    exit()

    # Plot full range (may be cluttered). To zoom, pass plot_range=(start,end)
    # Example zoom: use a smaller window around a certain date string: ('2025-07-05','2025-07-07')
    # plot_with_mplfinance(bars, trades, plot_range=None)
    # simple_plot(bars, trades)

if __name__ == "__main__":
    main()

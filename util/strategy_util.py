import time
from client_util import *
import pandas as pd
import talib
import numpy as np
import xgboost as xgb

class DayPriceLevelProvider:
    def __init__(self):
        # self.ohlcv = ohlcv.copy()
        # self.ohlcv_feat = self.compute_bar_stats(self.ohlcv)

        self.prev_day_levels = self.calc_prev_day_levels()

    @staticmethod
    def calc_prev_day_levels() -> pd.DataFrame:

        prev_day_df = get_prev_day_df()

        levels = {}
        levels['prev_high'] = prev_day_df['price'].max()
        levels['prev_low']  = prev_day_df['price'].min()
        levels['prev_mid']  = (levels['prev_high'] + levels['prev_low']) / 2

        return levels

    @staticmethod
    def compute_bar_stats(group) -> pd.DataFrame:

        rth_open_hr = 6
        rth_close_hr = 13

        def check_ovn(row):

            dt = row.name

            if dt.hour < rth_open_hr or dt.hour > rth_close_hr or \
                (dt.hour == rth_open_hr and dt.minute < 30) or \
                (dt.hour == rth_close_hr and dt.second > 0):
                    return 1
            return 0

        # group["sma5"] = talib.SMA(group["close"], timeperiod=5)
        group["rsi"] = talib.RSI(group["close"], timeperiod=14)

        pd.set_option('display.max_rows', None)


        group['ovn'] = group.apply(check_ovn, axis=1)
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

        # time_df = group.set_index("dt")

        # rth_df = time_df.between_time('6:30', '12:59')

        ib_df = group.between_time('6:30', '7:30')

        ib_lo = ib_df['low'].min()
        ib_hi = ib_df['high'].max()

        ib_lo_z = (group['close'] - ib_lo) / vwap_std
        ib_hi_z = (ib_hi - group['close']) / vwap_std

        group['ib_lo_z'] = 0.0
        group['ib_hi_z'] = 0.0

        group['ib_lo'] = 0.0
        group['ib_hi'] = 0.0
        

        rth_after_ib_idx = group.between_time('7:30', '12:59').index

        group.loc[rth_after_ib_idx, 'ib_lo_z'] = ib_lo_z.loc[rth_after_ib_idx]
        group.loc[rth_after_ib_idx, 'ib_hi_z'] = ib_hi_z.loc[rth_after_ib_idx]

        group.loc[rth_after_ib_idx, 'ib_lo'] = ib_lo
        group.loc[rth_after_ib_idx, 'ib_hi'] = ib_hi


        group['rth_lo'] = 0.0
        group['rth_hi'] = 0.0
        group.loc[rth_after_ib_idx, 'rth_lo'] = group.loc[rth_after_ib_idx, 'low'].cummin()
        group.loc[rth_after_ib_idx, 'rth_hi'] = group.loc[rth_after_ib_idx, 'high'].cummax()


        group["rsi"] = talib.RSI(group["close"], timeperiod=14)

        vol_mean = group['volume'].rolling(window=20, center=True, min_periods=1).mean()
        vol_std  = group['volume'].rolling(window=20, center=True, min_periods=1).std().add(1e-6)

        group['vol_z'] = (group['volume'] - vol_mean) / vol_std

        group['adx'] = talib.ADX(group['high'], group['low'], group['close'], timeperiod=14)


        ofi_pct = (group['buys'] - group['sells']) / (group['buys'] + group['sells'])
        window = 60  # minutes or bars
        ofi_pct_series = ofi_pct.fillna(0)
        ofi_z = (ofi_pct_series - ofi_pct_series.rolling(window).mean()) / ofi_pct_series.rolling(window).std()
        group['ofi_z'] = ofi_z


        return group

    def attach_levels_to_bars(self, ohlcv: pd.DataFrame, **kwargs) -> pd.DataFrame:

        ohlcv_feat = self.compute_bar_stats(ohlcv)

        t_samp = kwargs.get('t_samp', '15Min')
        bars = ohlcv.resample(t_samp).agg({
            'price': 'last',
            'buys': 'sum',
            'sells': 'sum',
            'volume': 'sum',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        })

        bars['prev_high'] = self.prev_day_levels['prev_high']
        bars['prev_low']  = self.prev_day_levels['prev_low']
        bars['prev_mid']  = self.prev_day_levels['prev_mid']

        self.feat_cols = ['close_z20','ovn_lo_z','ovn_hi_z','ib_lo_z','ib_hi_z','vwap_z','rsi','vol_z', 'adx', 'ofi_z']
        feat_1m = ohlcv_feat[['vwap','ovn_lo','ovn_hi', 'ovn','ib_lo','ib_hi'] + self.feat_cols]

        bars = pd.merge(bars, feat_1m, left_index=True, right_index=True, how='left')

        return bars


class BreakoutRetestStrategy:
    def __init__(self, level_cols=['prev_high','prev_low','prev_mid'],
                 threshold_pct=0.0012,
                 lookahead_bars=40):

        self.plp = DayPriceLevelProvider()

        self.level_cols = level_cols
        self.threshold_pct = threshold_pct
        self.lookahead_bars = lookahead_bars

        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model('../bo_retest_model.json')

        

    def detect_breakouts(self, ohlcv: pd.DataFrame):

        self.bars = self.plp.attach_levels_to_bars(ohlcv)

        events = []
        c = self.bars['close']
        prev_close = c.shift(1)

        rth_mask = self.bars['ovn'] == 0

        for lvl in self.level_cols:
            if lvl not in self.bars.columns:
                continue
            L = self.bars[lvl]

            bull_mask = (prev_close <= L) & (c > L) & rth_mask
            bear_mask = (prev_close >= L) & (c < L) & rth_mask
            for ts in self.bars.index[bull_mask.fillna(False)]:
                events.append((lvl, float(L.loc[ts]), ts, 'bull'))
            for ts in self.bars.index[bear_mask.fillna(False)]:
                events.append((lvl, float(L.loc[ts]), ts, 'bear'))
        events.sort(key=lambda e: e[2])
        return events
    
    def find_retest_and_build_trades(self, ohlcv: pd.DataFrame, stop_buffer_pct=0.0025):
        trades = []

        # detect breakouts on 15min bars, but retest/trigger on 5min bars
        events = self.detect_breakouts(ohlcv)

        trigger_bars = self.plp.attach_levels_to_bars(ohlcv, t_samp='5Min')

        for lvl_name, L, breakout_ts, direction in events:

            # print(f"Processing {direction} breakout at {breakout_ts} of level {lvl_name}={L}")

            try:
                idx0 = trigger_bars.index.get_loc(breakout_ts)
            except KeyError:
                continue
            look_slice = trigger_bars.iloc[idx0+1: idx0+1 + self.lookahead_bars]

            # is breakout within last lookahead_bars?
            if look_slice.empty or len(look_slice) == self.lookahead_bars:
                continue
            

            # ToDo: try to optimize this threshold
            risk = min(stop_buffer_pct * L, 10.0)

            threshold = min(self.threshold_pct * L, risk * 0.5)
            distances = (look_slice['close'] - L).abs()

            # if np.any(distances > 10.0):
            #     continue

            hits = distances <= threshold

            # if not hits.any():
            #     continue
            # hit_idx = hits.idxmax()

            if not hits.iloc[-1]:
                continue
            hit_idx = hits.index[-1]
            entry_price = float(look_slice.loc[hit_idx, 'close'])

            trigger_row = trigger_bars.iloc[-1]

            feature_cols = self.plp.feat_cols
            bear = 1 if direction == 'bear' else 0

            feature_vec = np.append(trigger_row[feature_cols].values, bear)

            pred = self.xgb_model.predict_proba(feature_vec.reshape(1, -1))

            if pred[0][1] < 0.4:
                continue

            trades.append({
                'trigger_ts': breakout_ts,
                'entry_ts': hit_idx,
                'entry_price': entry_price,
                'direction': direction,
                'level_name': lvl_name,
                'level_value': L,
                'pred' : float(pred[0][1])
            })

        return trades

class ReversionStrategy:
    def __init__(self, level_cols=['prev_high','prev_low','prev_mid'],
                 threshold_pct=0.0012,
                 lookahead_bars=40):

        self.plp = DayPriceLevelProvider()

        self.level_cols = level_cols
        self.threshold_pct = threshold_pct
        self.lookahead_bars = lookahead_bars

        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model('../reversion_model.json')

        

    def get_trigger_events(self, ohlcv: pd.DataFrame):

        self.bars = self.plp.attach_levels_to_bars(ohlcv, t_samp='5Min')

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
    
    def find_retest_and_build_trades(self, ohlcv: pd.DataFrame, stop_buffer_pct=0.0025):
        trades = []

        events = self.get_trigger_events(ohlcv)

        trigger_bars = self.plp.attach_levels_to_bars(ohlcv, t_samp='5Min')

        # trigger_bars = self.bars

        for lvl_name, L, breakout_ts, direction in events:

            # print(f"Processing {direction} breakout at {breakout_ts} of level {lvl_name}={L}")

            try:
                idx0 = trigger_bars.index.get_loc(breakout_ts)
            except KeyError:
                continue
            look_slice = trigger_bars.iloc[idx0+1: idx0+1 + self.lookahead_bars]

            # is breakout within last lookahead_bars?
            if look_slice.empty or len(look_slice) == self.lookahead_bars:
                continue
            

            # ToDo: try to optimize this threshold
            risk = min(stop_buffer_pct * L, 10.0)

            threshold = min(self.threshold_pct * L, risk * 0.5)
            distances = (look_slice['close'] - L).abs()

            if np.any(distances > 10.0):
                continue

            hits = distances <= threshold

            # if not hits.any():
            #     continue
            # hit_idx = hits.idxmax()


            if not hits.iloc[-1]:
                continue
            hit_idx = hits.index[-1]
            entry_price = float(look_slice.loc[hit_idx, 'close'])

            trigger_row = trigger_bars.iloc[-1]

            # trigger_row['bear'] = 1 if direction == 'bear' else 0

            feature_cols = self.plp.feat_cols

            bear = 1 if direction == 'bear' else 0

            feature_vec = np.append(trigger_row[feature_cols].values, bear)

            pred = self.xgb_model.predict_proba(feature_vec.reshape(1, -1))

            if pred[0][1] < 0.4:
                continue

            trades.append({
                'trigger_ts': breakout_ts,
                'entry_ts': hit_idx,
                'entry_price': entry_price,
                'direction': direction,
                'level_name': lvl_name,
                'level_value': L,
                'pred' : float(pred[0][1])
            })

        # remove duplicates of the same level
        # can keep them later if time since trigger becomes feature
        unique_trades = {}
        for trade in trades:
            key = (trade['level_name'], trade['direction'])
            if key not in unique_trades:
                unique_trades[key] = trade
        trades = list(unique_trades.values())

        return trades

if __name__ == "__main__":

    print(get_prev_day_df())
    pass
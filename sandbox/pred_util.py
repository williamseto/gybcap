
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

def generate_second_data(start_date, num_days, seed_base=0):
    all_seconds = []
    for day_offset in range(num_days):
        # np.random.seed(seed_base + day_offset)
        current_date = pd.to_datetime(start_date) + pd.Timedelta(days=day_offset)
        day_start = current_date + pd.Timedelta(hours=9, minutes=30)
        n_seconds = int(6.5 * 3600)  # 6.5 trading hours
        date_rng = pd.date_range(start=day_start, periods=n_seconds, freq='S')
        returns = np.random.normal(loc=0.00, scale=0.01, size=n_seconds)
        price = 100 + np.cumsum(returns)
        df = pd.DataFrame({'datetime': date_rng, 'price_s': price})
        df['date'] = df['datetime'].dt.date
        all_seconds.append(df)
    sec_df = pd.concat(all_seconds).reset_index(drop=True)
    return sec_df

def compute_slope(prices, window=10):
    n = len(prices)

    start = max(0, n - window - 1)
    x = np.arange(start, n)
    y = prices[start:n]
    slope = linregress(x, y).slope if len(x) > 1 else 0.0
    return slope

def generate_strict_reversal_labels(group, delta=0.002, slope_window=20, min_slope=1e-4):
    """
    Label the first legitimate reversal based on strict criteria:
    - Only consider reversal if initial trend slope magnitude >= min_slope.
    - Find first opposite move >= delta.
    - After that point, ensure no bounce in original trend direction >= 0.5 * delta
      before market close; otherwise discard reversal.
    - Compute remaining-day range as usual.
    """
    prices = group['price'].values
    n = len(prices)
    y_rev = np.zeros(n, dtype=int)
    y_range = np.zeros(n, dtype=float)

    # Compute rolling slope for trend detection
    slopes = np.zeros(n)
    for i in range(n):
        start = max(0, i - slope_window)
        x = np.arange(start, i + 1)
        y = prices[start:i + 1]
        slopes[i] = linregress(x, y).slope if len(x) > 1 else 0.0

    for i in range(n):
        current = prices[i]
        y_range[i] = (prices[i:].max() - prices[i:].min()) / current

        # Skip if no clear trend
        if abs(slopes[i]) < min_slope:
            continue

        # Determine trend direction: +1 for up, -1 for down
        direction = 1 if slopes[i] > 0 else -1

        # Future movements from this point
        future = prices[i:]
        pct = (future - current) / current

        # Identify first reversal candidate
        if direction == 1:
            rev_idxs = np.where(pct <= -delta)[0]
        else:
            rev_idxs = np.where(pct >= delta)[0]

        if rev_idxs.size == 0:
            continue

        j = rev_idxs[0]  # index of first reversal
        # After reversal, check for bounce in original direction
        post_rev = future[j:]
        post_pct = (post_rev - future[j]) / future[j]
        bounce_threshold = 0.5 * delta * direction  # positive for uptrend, negative for downtrend

        # For an uptrend, bounce_threshold is positive: check if any post_pct >= bounce_threshold
        # For a downtrend, bounce_threshold is negative: check if any post_pct <= bounce_threshold
        if direction == 1 and np.any(post_pct >= bounce_threshold):
            continue
        if direction == -1 and np.any(post_pct <= bounce_threshold):
            continue

        # If no disqualifying bounce, mark reversal
        y_rev[i] = 1

    group['y_rev'] = y_rev
    group['y_range'] = y_range
    return group

def simulate_trade_utility(prices, entry_idx,
                           stop_loss_pct=0.002, rr=2.0, trailing=True):

    entry_price = prices[entry_idx]
    if entry_idx > 0:
        direction = -1 if prices[entry_idx - 1] < entry_price else 1
    else:
        direction = -1
    sl_price = entry_price * (1 - direction * stop_loss_pct)
    tp_price = entry_price * (1 + direction * stop_loss_pct * rr)
    in_profit = False
    trail_sl = sl_price

    for i in range(entry_idx + 1, len(prices)):
        price = prices[i]
        if not in_profit:
            if (direction == 1 and price <= sl_price) or (direction == -1 and price >= sl_price):
                pnl = (sl_price - entry_price) * direction
                return pnl / (entry_price * stop_loss_pct)
            if (direction == 1 and price >= tp_price) or (direction == -1 and price <= tp_price):
                in_profit = True
                trail_sl = entry_price
                continue
        else:
            if direction == 1:
                trail_sl = max(trail_sl, price * (1 - stop_loss_pct))
                if price <= trail_sl:
                    pnl = (trail_sl - entry_price) * direction
                    return pnl / (entry_price * stop_loss_pct)
            else:
                trail_sl = min(trail_sl, price * (1 + stop_loss_pct))
                if price >= trail_sl:
                    pnl = (trail_sl - entry_price) * direction
                    return pnl / (entry_price * stop_loss_pct)
    pnl = (prices[-1] - entry_price) * direction
    return pnl / (entry_price * stop_loss_pct)

def get_slot_utils(slot, T_candidate, T_base, df_sorted):
    slot_utils = []
    for day, group in df_sorted.groupby('trading_day'):
        slot_idx = 1
        for _, row in group.iterrows():
            s = row['score']
            if slot_idx < slot:
                if s >= T_base[slot_idx]:
                    slot_idx += 1
                continue
            elif slot_idx == slot:
                if s >= T_candidate:
                    slot_utils.append(row['utility'])
                    break
                else:
                    continue
            else:
                break
    return slot_utils

def calibrate_threshold(slot, df_sorted, T_base, target_util=-0.1):
    scores = df_sorted['score'].values
    lo, hi = np.min(scores), np.max(scores)
    tol = 1e-3
    best_T = lo
    for _ in range(30):
        mid = (lo + hi) / 2
        utils = get_slot_utils(slot, mid, T_base, df_sorted)
        if len(utils) == 0:
            hi = mid
            continue
        avg_util = np.mean(utils)
        if abs(avg_util - target_util) < tol:
            best_T = mid
            break
        if avg_util > target_util:
            lo = mid
        else:
            hi = mid
        best_T = mid
    return best_T

class ReversalModel:
    def __init__(self, data, dt_idx='datetime', day_idx='date', params=None):

        self.sec_df = data
        self.dt_idx = dt_idx
        self.day_idx = day_idx

        if params is None:
            self.params = {
                'objective': 'binary:logistic',
                'max_depth': 4,
                'eta': 0.1,
                'verbosity': 1
            }
        else:
            self.params = params

    def compute_minute_features(self, sec_df):
    
        min_df = sec_df.set_index(self.dt_idx).groupby(self.day_idx).resample('1Min').agg({'price_s':'mean'}).bfill().reset_index()

        min_df.rename(columns={'price_s': 'price'}, inplace=True)
        min_df = min_df.groupby(self.day_idx, group_keys=False).apply(generate_strict_reversal_labels)

        # Feature engineering: minute‐level
        min_df['ret_1m'] = min_df.groupby(self.day_idx)['price'].pct_change(1).fillna(0)
        min_df['ret_5m'] = min_df.groupby(self.day_idx)['price'].pct_change(5).fillna(0)
        min_df['vol_10m'] = (min_df.groupby(self.day_idx)['ret_1m']
                            .rolling(10).std()
                            .reset_index(level=0, drop=True)
                            .fillna(0))
        min_df['min_to_close'] = min_df.groupby(self.day_idx)[self.dt_idx]\
            .transform(lambda x: (x.max() - x).dt.total_seconds()/60)
        
        min_df['dist_to_max'] = min_df.groupby(self.day_idx)['price'].transform(lambda x: x.max() - x)
        min_df['dist_to_min'] = min_df.groupby(self.day_idx)['price'].transform(lambda x: x - x.min())
        
        return min_df

    def compute_features(self, sec_df=None):
        if sec_df is None:
            sec_df = self.sec_df
        min_df = self.compute_minute_features(sec_df)

        # Feature engineering: second‐level aggregated to minute
        # compute per‐minute window over last 60 seconds
        # self.sec_df['ret_s'] = self.sec_df.groupby('date')['price_s'].pct_change().fillna(0)
        # # volatility in last 60s and count of sign flips
        # self.sec_df['sign'] = np.sign(self.sec_df['ret_s'])
        # self.sec_df['sign_flip'] = self.sec_df['sign'] != self.sec_df.groupby('date')['sign'].shift(1)
        # # aggregate for each minute
        # sec_feat = self.sec_df.groupby([self.sec_df['date'], self.sec_df['datetime'].dt.floor('T')]).agg({
        #     'ret_s': ['std'],
        #     'sign_flip': 'sum'
        # })
        # sec_feat.columns = ['vol_60s', 'flips_60s']
        # sec_feat = sec_feat.reset_index().rename(columns={'datetime': 'minute'})

        # # merge second‐level features into minute DF
        # min_df = pd.merge(min_df,
        #                 sec_feat,
        #                 left_on=['date', 'datetime'],
        #                 right_on=['date', 'minute'],
        #                 how='left').drop(columns=['minute'])
        
        if sec_df is not None:
            self.min_df = min_df

        return min_df

    def train(self, min_df=None):
        if min_df is None:
            min_df = self.min_df

        dtrain = self._prepare_data(min_df)

        # self.bst = xgb.train(self.params, dtrain, num_boost_round=50, obj=self._multitask_obj)
        self.bst = xgb.train(self.params, dtrain, num_boost_round=50)


    def predict(self, min_df):
        dtrain = self._prepare_data(min_df)

        preds = self.bst.predict(dtrain)

        min_df['pred_prob'] = preds
        min_df['pred_rev'] = (preds>=0.3).astype(int)

        contribs = None
        if len(min_df) == 1:
            contribs = self.bst.predict(dtrain, pred_contribs=True)
        return preds, contribs

        N = len(min_df)
        prob_rev = 1/(1+np.exp(-preds[:N]))
        min_df['pred_rev'] = (prob_rev>=0.5).astype(int)

        accuracy = (min_df['pred_rev']==min_df['y_rev']).mean()
        # print(f"Reversal accuracy: {accuracy:.2%}")

        return prob_rev



    def _prepare_data(self, min_df):
        # features = ['ret_1m', 'ret_5m', 'vol_10m', 'min_to_close', 'vol_60s', 'flips_60s']
        features = ['ret_1m', 'ret_5m', 'vol_10m', 'min_to_close', 'dist_to_max', 'dist_to_min']
        X = min_df[features].values
        y_rev = min_df['y_rev'].values

        # y_range = min_df['y_range'].values
        # # stack for multi‐task
        # X_dup = np.vstack([X, X])
        # y_flat = np.concatenate([y_rev, y_range])
        # dtrain = xgb.DMatrix(X_dup, label=y_flat)


        dtrain = xgb.DMatrix(X, label=y_rev)

        return dtrain

    def _multitask_obj(self, preds, dtrain):
        N = len(preds)//2
        logit = preds[:N]
        rng = preds[N:]
        labels = dtrain.get_label()
        y1, y2 = labels[:N], labels[N:]
        p = 1/(1+np.exp(-logit))
        grad1 = p - y1
        hess1 = p*(1-p)
        grad2 = rng - y2
        hess2 = np.ones_like(grad2)
        grad = np.concatenate([grad1, grad2])
        hess = np.concatenate([hess1, hess2])
        return grad, hess


class UtilityModel:
    def __init__(self, data, pred_rev_labels, day_idx='date', params=None):

        self.day_idx = day_idx

        if params is None:
            self.params = {
                'objective': 'reg:squarederror',
                'max_depth': 4,
                'eta': 0.1,
                'verbosity': 1
            }
        else:
            self.params = params

        self.data_df = data
        self.data_df['pred_rev_labels'] = pred_rev_labels

    def train(self):

        def simulate_trade_utility_day(group):
            day_group = group.reset_index(drop=True)
            group['utility'] = 0.0
            pred_rev_entries = day_group[day_group['pred_rev_labels']==1]

            if len(pred_rev_entries) == 0:
                return group

            def apply_row(row):
                return simulate_trade_utility(group['price'].values, row.name)

            utils = pred_rev_entries.apply(apply_row, axis=1)

            group_mask = group['pred_rev_labels']==1

            group.loc[group_mask, 'utility'] = utils.to_numpy()
            
            return group


        util_df = self.data_df.groupby(self.day_idx, group_keys=False).apply(simulate_trade_utility_day)

        util_df = util_df[util_df['pred_rev_labels']==1]

        unique_dates = sorted(self.data_df[self.day_idx].unique())

        train_dates, calib_dates = train_test_split(unique_dates, test_size=0.3, shuffle=False)


        # ---------------------------------------------
        # Train Utility Model on Train Set
        # ---------------------------------------------

        train_df = util_df[util_df[self.day_idx].isin(train_dates)].reset_index(drop=True)

        features = ['pred_prob', 'ret_1m', 'ret_5m', 'vol_10m', 'min_to_close']
        self.features = features

        X_train = train_df[features].values
        y_train = train_df['utility'].values

        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {'objective': 'reg:squarederror', 'max_depth': 4, 'eta': 0.1, 'verbosity': 0}
        f_theta = xgb.train(params, dtrain, num_boost_round=100)

        # Score calibration set
        calib_df = util_df[util_df[self.day_idx].isin(calib_dates)].reset_index(drop=True)

        X_calib = calib_df[features].values
        calib_df['score'] = f_theta.predict(xgb.DMatrix(X_calib))


        N_SLOTS = 5
        T_base = {i: 0.0 for i in range(1, N_SLOTS+1)}
        for k in range(1, N_SLOTS+1):
            T_base[k] = calibrate_threshold(k, calib_df, T_base, target_util=-0.1)

        print("Calibrated Thresholds on Full Dataset (slot → utility score):")
        for k in range(1, N_SLOTS+1):
            print(f"  Slot {k}: {T_base[k]:.5f}")
    
        # ---------------------------------------------
        # Retrain Utility Model on Train+Calib for Backtest
        # ---------------------------------------------
        train_calib_df = pd.concat([train_df, calib_df], ignore_index=True)
        X_full = train_calib_df[features].values
        y_full = train_calib_df['utility'].values
        dtrain_full = xgb.DMatrix(X_full, label=y_full)
        f_theta_full = xgb.train(params, dtrain_full, num_boost_round=100)

        self.f_theta_full = f_theta_full
        self.T_base = T_base
    




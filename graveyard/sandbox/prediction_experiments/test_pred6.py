import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------
# 1. Synthetic Data Generation (Seconds → Minutes)
# ---------------------------------------------
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

# Generate 30 days of synthetic second‐level data
sec_all = generate_second_data('2025-05-01', num_days=30, seed_base=42)

# Aggregate to minute bars
min_df = sec_all.copy()
min_df['datetime'] = min_df['datetime'].dt.floor('T')
min_df = min_df.groupby(['date', 'datetime']).agg({'price_s':'mean'}).reset_index()
min_df.rename(columns={'price_s': 'price'}, inplace=True)

# Extract unique trading dates
unique_dates = sorted(min_df['date'].unique())

# ---------------------------------------------
# 2. Utility‐Labeling Functions
# ---------------------------------------------
def compute_slope(prices, window=10):
    n = len(prices)
    slopes = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        x = np.arange(start, i + 1)
        y = prices[start:i + 1]
        slopes[i] = linregress(x, y).slope if len(x) > 1 else 0.0
    return slopes

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

# ---------------------------------------------
# 3. Build (Features → Utility) Table for All Signals
# ---------------------------------------------
hist_signals = []

for day in unique_dates:
    group = min_df[min_df['date'] == day].reset_index(drop=True)
    prices = group['price'].values
    n = len(prices)
    slopes = compute_slope(prices, window=20)
    
    group['ret_1m'] = group['price'].pct_change().fillna(0)
    group['ret_5m'] = group['price'].pct_change(5).fillna(0)
    group['vol_10m'] = group['ret_1m'].rolling(10).std().fillna(0)
    group['min_to_close'] = (group['datetime'].max() - group['datetime']).dt.total_seconds() / 60
    
    sec_day = sec_all[sec_all['date'] == day].copy()
    sec_day['minute'] = sec_day['datetime'].dt.floor('T')
    sec_day['ret_s'] = sec_day['price_s'].pct_change().fillna(0)
    sec_day['sign'] = np.sign(sec_day['ret_s'])
    sec_day['sign_flip'] = sec_day['sign'] != sec_day['sign'].shift(1)
    sec_feat = sec_day.groupby(['date', 'minute']).agg({'ret_s': 'std', 'sign_flip': 'sum'})
    sec_feat.columns = ['vol_60s', 'flips_60s']
    sec_feat = sec_feat.reset_index()
    
    merged = pd.merge(group, sec_feat,
                      left_on=['date', 'datetime'],
                      right_on=['date', 'minute'],
                      how='left').fillna(0)
    merged.drop(columns=['minute'], inplace=True)

    min_slope = 1e-4
    delta = 0.002
    for i in range(n):
        current = prices[i]
        future = prices[i:]
        # direction = 1 if slopes[i] > 0 else -1
        pct = (future - current) / current

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

        # rev_idxs = np.where(pct <= -0.002 if direction == 1 else pct >= 0.002)[0]
        # if rev_idxs.size > 0 and rev_idxs[0] > 0:
        if rev_idxs.size > 0:
            features = {
                'p_rev': 1.0,
                'R_day': (future.max() - future.min()) / current,
                'ret_1m': merged.loc[i, 'ret_1m'],
                'ret_5m': merged.loc[i, 'ret_5m'],
                'vol_10m': merged.loc[i, 'vol_10m'],
                'min_to_close': merged.loc[i, 'min_to_close'],
                # 'vol_60s': merged.loc[i, 'vol_60s'],
                # 'flips_60s': merged.loc[i, 'flips_60s'],
                'time_of_day_norm': i / n
            }
            util = simulate_trade_utility(prices, i,
                                          stop_loss_pct=0.002, rr=2.0, trailing=True)
            hist_signals.append({
                'day': day,
                'time_idx': i,
                'features': features,
                'utility': util
            })

hist_df = pd.DataFrame(hist_signals)
feature_keys = list(hist_df['features'].iloc[0].keys())
X_all = np.vstack(hist_df['features'].apply(
    lambda d: np.array([d[k] for k in feature_keys])
))
y_all = hist_df['utility'].values

# ---------------------------------------------
# 4. Split into Train (Days 1-20) & Calibrate (Days 21-25)
# ---------------------------------------------
train_days = unique_dates[:20]   # days 0–19
calib_days = unique_dates[20:25] # days 20–24
test_days  = unique_dates        # days 0–29 (full backtest)

train_df = hist_df[hist_df['day'].isin(train_days)].reset_index(drop=True)
calib_df = hist_df[hist_df['day'].isin(calib_days)].reset_index(drop=True)

# ---------------------------------------------
# 5. Train Utility Model on Train Set
# ---------------------------------------------
X_train = np.vstack(train_df['features'].apply(lambda d: np.array([d[k] for k in feature_keys])))
y_train = train_df['utility'].values

dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'objective': 'reg:squarederror', 'max_depth': 4, 'eta': 0.1, 'verbosity': 0}
f_theta = xgb.train(params, dtrain, num_boost_round=100)

# Score calibration set
X_calib = np.vstack(calib_df['features'].apply(lambda d: np.array([d[k] for k in feature_keys])))
calib_df['score'] = f_theta.predict(xgb.DMatrix(X_calib))


# ---------------------------------------------
# 6. Calibrate Slot Thresholds on Calib Set
# ---------------------------------------------
calib_sorted = calib_df.sort_values(['day', 'time_idx']).reset_index(drop=True)

def get_slot_utils(slot, T_candidate, T_base, df_sorted):
    slot_utils = []
    for day, group in df_sorted.groupby('day'):
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

T_base = {i: 0.0 for i in range(1, 6)}
for k in range(1, 6):
    T_base[k] = calibrate_threshold(k, calib_sorted, T_base, target_util=-0.1)

print("Calibrated Thresholds on Full Dataset (slot → utility score):")
for k in range(1, 6):
    print(f"  Slot {k}: {T_base[k]:.5f}")

# ---------------------------------------------
# 7. Retrain Utility Model on Train+Calib for Backtest
# ---------------------------------------------
train_calib_df = pd.concat([train_df, calib_df], ignore_index=True)
X_full = np.vstack(train_calib_df['features'].apply(lambda d: np.array([d[k] for k in feature_keys])))
y_full = train_calib_df['utility'].values
dtrain_full = xgb.DMatrix(X_full, label=y_full)
f_theta_full = xgb.train(params, dtrain_full, num_boost_round=100)

# ---------------------------------------------
# 8. Backtest on ALL 30 Days (Days 0–29)
# ---------------------------------------------
STOP_LOSS_PCT = 0.001
RR = 2.0
DELTA_PERF = 0.005
KAPPA = 0.1
BUDGET = 5

# Precompute historical 30‐minute vol for market adjustment
min_df['ret_1m'] = min_df.groupby('date')['price'].pct_change().fillna(0)
min_df['vol_30m'] = min_df.groupby('date')['ret_1m'].rolling(30).std().reset_index(level=0, drop=True).fillna(0)
vol_hist_mean = min_df['vol_30m'].mean()

backtest_results = []

for day in unique_dates:
    test_group = min_df[min_df['date'] == day].reset_index(drop=True)
    prices = test_group['price'].values
    n = len(prices)
    test_group['slopes'] = compute_slope(prices, window=10)
    test_group['ret_1m'] = test_group['price'].pct_change().fillna(0)
    test_group['ret_5m'] = test_group['price'].pct_change(5).fillna(0)
    test_group['vol_10m'] = test_group['ret_1m'].rolling(10).std().fillna(0)
    test_group['min_to_close'] = (test_group['datetime'].max() - test_group['datetime']).dt.total_seconds() / 60
    
    sec_day = sec_all[sec_all['date'] == day].copy()
    sec_day['minute'] = sec_day['datetime'].dt.floor('T')
    sec_day['ret_s'] = sec_day['price_s'].pct_change().fillna(0)
    sec_day['sign'] = np.sign(sec_day['ret_s'])
    sec_day['sign_flip'] = sec_day['sign'] != sec_day['sign'].shift(1)
    sec_feat = sec_day.groupby(['date', 'minute']).agg({'ret_s': 'std', 'sign_flip': 'sum'})
    sec_feat.columns = ['vol_60s', 'flips_60s']
    sec_feat = sec_feat.reset_index()
    
    test_group = pd.merge(test_group, sec_feat,
                          left_on=['date', 'datetime'],
                          right_on=['date', 'minute'],
                          how='left').fillna(0).drop(columns=['minute'])
    
    in_trade = False
    slot_today = 0
    pnl_cum = 0.0
    trades = []
    
    for i, row in test_group.iterrows():
        # If currently in trade, check exits
        if in_trade:
            price = row['price']
            if (direction_exec == 1 and price <= sl_price) or (direction_exec == -1 and price >= sl_price):
                util = (sl_price - entry_price) * direction_exec / (entry_price * STOP_LOSS_PCT)
                in_trade = False; pnl_cum += util
                trades[-1].update({'exit_time': row['datetime'], 'exit_price': sl_price, 'util': util})
                continue
            if in_profit:
                if direction_exec == 1:
                    trail_sl = max(trail_sl, price * (1 - STOP_LOSS_PCT))
                    if price <= trail_sl:
                        util = (trail_sl - entry_price) * direction_exec / (entry_price * STOP_LOSS_PCT)
                        in_trade = False; pnl_cum += util
                        trades[-1].update({'exit_time': row['datetime'], 'exit_price': trail_sl, 'util': util})
                        continue
                else:
                    trail_sl = min(trail_sl, price * (1 + STOP_LOSS_PCT))
                    if price >= trail_sl:
                        util = (trail_sl - entry_price) * direction_exec / (entry_price * STOP_LOSS_PCT)
                        in_trade = False; pnl_cum += util
                        trades[-1].update({'exit_time': row['datetime'], 'exit_price': trail_sl, 'util': util})
                        continue
            future = prices[i:]
            direction = 1 if row['slopes'] > 0 else -1
            pct = (future - price) / price
            rev_idxs = np.where(pct <= -0.002 if direction == 1 else pct >= 0.002)[0]
            if rev_idxs.size > 0 and rev_idxs[0] == 0 and direction == -direction_exec:
                util = (price - entry_price) * direction_exec / (entry_price * STOP_LOSS_PCT)
                in_trade = False; pnl_cum += util
                trades[-1].update({'exit_time': row['datetime'], 'exit_price': price, 'util': util})
                continue
            continue
        
        if slot_today >= BUDGET:
            break
        
        current = row['price']
        future = prices[i:]
        direction = 1 if row['slopes'] > 0 else -1
        pct = (future - current) / current
        rev_idxs = np.where(pct <= -0.002 if direction == 1 else pct >= 0.002)[0]
        if rev_idxs.size == 0 or rev_idxs[0] == 0:
            continue
        
        features = np.array([
            1.0,
            ((future.max() - future.min()) / current),
            row['ret_1m'],
            row['ret_5m'],
            row['vol_10m'],
            row['min_to_close'],
            row['vol_60s'],
            row['flips_60s'],
            i / n
        ])
        score_t = f_theta_full.predict(xgb.DMatrix(features.reshape(1, -1)))[0]
        k = slot_today + 1
        
        delta_perf = -DELTA_PERF if pnl_cum >= 0 else DELTA_PERF
        if i >= 29:
            vol30 = test_group['ret_1m'].iloc[i-29:i+1].std()
        else:
            vol30 = vol_hist_mean
        delta_mkt = KAPPA * (vol30 - vol_hist_mean)
        
        T_today = T_base[k] + delta_perf + delta_mkt
        if score_t < T_today:
            continue
        
        entry_price = current
        direction_exec = -direction
        sl_price = entry_price * (1 - direction_exec * STOP_LOSS_PCT)
        tp_price = entry_price * (1 + direction_exec * STOP_LOSS_PCT * RR)
        in_trade = True
        in_profit = False
        trail_sl = sl_price
        
        slot_today += 1
        trades.append({
            'slot': k,
            'entry_time': row['datetime'],
            'entry_price': entry_price,
            'score': score_t,
            'threshold': T_today
        })
    
    # End-of-day forced exit if still in trade
    if in_trade:
        final_price = prices[-1]
        util = (final_price - entry_price) * direction_exec / (entry_price * STOP_LOSS_PCT)
        pnl_cum += util
        trades[-1].update({
            'exit_time': test_group['datetime'].iloc[-1],
            'exit_price': final_price,
            'util': util
        })
    
    # Record metrics for this day
    day_utils = [t['util'] for t in trades if 'util' in t]
    backtest_results.append({
        'day': day,
        'num_trades': len(trades),
        'total_util': sum(day_utils),
        'win_rate': np.mean([u > 0 for u in day_utils]) if day_utils else np.nan
    })

# ---------------------------------------------
# 7. Aggregate & Display Results
# ---------------------------------------------
results_df = pd.DataFrame(backtest_results)
print("Backtest Over Entire Dataset Summary:")
print(results_df.to_string(index=False))

print("\nOverall Metrics:")
print(f"  Avg Trades/Day   : {results_df['num_trades'].mean():.2f}")
print(f"  Avg Utility/Day  : {results_df['total_util'].mean():.2f} R")
print(f"  Avg Win Rate     : {results_df['win_rate'].mean() * 100:.2f}%")

# Plot total utility per day
plt.figure(figsize=(10, 4))
plt.bar(results_df['day'].astype(str), results_df['total_util'])
plt.xticks(rotation=45)
plt.ylabel('Total Utility (R multiples)')
plt.title('Daily Total Utility (Full Sample Backtest)')
plt.tight_layout()
plt.show()

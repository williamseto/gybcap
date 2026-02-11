import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------
# 1. Synthetic Data Generation (Seconds + Minutes)
# ---------------------------------------------
def generate_second_data(start_date, num_days, seed_base=0):
    all_seconds = []
    for day_offset in range(num_days):
        # np.random.seed(seed_base + day_offset)
        current_date = pd.to_datetime(start_date) + pd.Timedelta(days=day_offset)
        day_start = current_date + pd.Timedelta(hours=9, minutes=30)
        n_seconds = int(6.5 * 3600)
        date_rng = pd.date_range(start=day_start, periods=n_seconds, freq='S')
        returns = np.random.normal(loc=0.00, scale=0.01, size=n_seconds)
        price = 100 + np.cumsum(returns)
        df = pd.DataFrame({'datetime': date_rng, 'price_s': price})
        df['date'] = df['datetime'].dt.date
        all_seconds.append(df)
    sec_df = pd.concat(all_seconds).reset_index(drop=True)
    return sec_df

# Generate 10 days historical, 1 live
sec_hist = generate_second_data('2025-05-01', num_days=10, seed_base=42)
sec_live = generate_second_data('2025-05-11', num_days=1, seed_base=100)
sec_all = pd.concat([sec_hist, sec_live]).reset_index(drop=True)

# Aggregate to minute-level without duplicate 'date'
min_df = sec_all.copy()
min_df['datetime'] = min_df['datetime'].dt.floor('T')
min_df = min_df.groupby(['date', 'datetime']).agg({'price_s':'mean'}).reset_index()
min_df.rename(columns={'price_s': 'price'}, inplace=True)

# Split historical vs live
live_date = sec_live['date'].iloc[0]
min_hist = min_df[min_df['date'] < live_date].reset_index(drop=True)
min_live = min_df[min_df['date'] == live_date].reset_index(drop=True)

# ---------------------------------------------
# 2. Label Generation & Historical PnL Simulation
# ---------------------------------------------
def compute_slope(prices, window):
    n = len(prices)
    slopes = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        x = np.arange(start, i + 1)
        y = prices[start:i + 1]
        slopes[i] = linregress(x, y).slope if len(x) > 1 else 0.0
    return slopes

def simulate_trade_pnl(prices, entry_idx, stop_loss_pct=0.002, rr=2.0, trailing=True):
    entry_price = prices[entry_idx]
    if entry_idx > 0:
        direction = -1 if prices[entry_idx-1] < entry_price else 1
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
                return (sl_price - entry_price) * direction
            if (direction == 1 and price >= tp_price) or (direction == -1 and price <= tp_price):
                in_profit = True
                trail_sl = entry_price
                continue
        else:
            if direction == 1:
                trail_sl = max(trail_sl, price * (1 - stop_loss_pct))
                if price <= trail_sl:
                    return (trail_sl - entry_price) * direction
            else:
                trail_sl = min(trail_sl, price * (1 + stop_loss_pct))
                if price >= trail_sl:
                    return (trail_sl - entry_price) * direction
    return (prices[-1] - entry_price) * direction

# Build historical signals DataFrame
hist_signals = []

for day, group in min_hist.groupby('date'):
    prices = group['price'].values
    n = len(prices)
    slopes = compute_slope(prices, window=20)
    group = group.copy()
    group['ret_1m'] = group['price'].pct_change().fillna(0)
    group['ret_5m'] = group['price'].pct_change(5).fillna(0)
    group['vol_10m'] = group['ret_1m'].rolling(10).std().fillna(0)
    group['min_to_close'] = (group['datetime'].max() - group['datetime']).dt.total_seconds() / 60

    sec_day = sec_hist[sec_hist['date'] == day].copy()
    sec_day['minute'] = sec_day['datetime'].dt.floor('T')
    sec_day['ret_s'] = sec_day['price_s'].pct_change().fillna(0)
    sec_day['sign'] = np.sign(sec_day['ret_s'])
    sec_day['sign_flip'] = sec_day['sign'] != sec_day['sign'].shift(1)
    sec_feat = sec_day.groupby(['date', 'minute']).agg({'ret_s': 'std', 'sign_flip': 'sum'})
    sec_feat.columns = ['vol_60s', 'flips_60s']
    sec_feat = sec_feat.reset_index()

    merged = pd.merge(group, sec_feat, left_on=['date', 'datetime'], right_on=['date', 'minute'], how='left').fillna(0)
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
                'ret_1m': merged.iloc[i]['ret_1m'],
                'ret_5m': merged.iloc[i]['ret_5m'],
                'vol_10m': merged.iloc[i]['vol_10m'],
                'min_to_close': merged.iloc[i]['min_to_close'],
                'vol_60s': merged.iloc[i]['vol_60s'],
                'flips_60s': merged.iloc[i]['flips_60s'],
                'time_of_day_norm': i / n
            }
            pnl = simulate_trade_pnl(prices, i, stop_loss_pct=0.002, rr=2.0, trailing=True)
            hist_signals.append({
                'day': day,
                'time_idx': i,
                'features': features,
                'pnl': pnl,
                'dt': group.iloc[i]['datetime'],
                'price': prices[i]
            })

# plt.plot(min_hist['datetime'], min_hist['price'])
# for signal in hist_signals:
#     plt.scatter(signal['dt'], signal['price'], color='red', marker='x')
# plt.show()
# exit()

hist_df = pd.DataFrame(hist_signals)
feature_keys = list(hist_df['features'].iloc[0].keys())
X_all = np.vstack(hist_df['features'].apply(lambda d: np.array([d[k] for k in feature_keys])))
y_all = hist_df['pnl'].values

# ---------------------------------------------
# 3. Split Historical into Train (first 8 days) and Calibration (last 2 days)
# ---------------------------------------------
unique_days = sorted(hist_df['day'].unique())
train_days = unique_days[:8]
calib_days = unique_days[8:]

train_mask = hist_df['day'].isin(train_days)
calib_mask = hist_df['day'].isin(calib_days)

train_df = hist_df[train_mask].reset_index(drop=True)
calib_df = hist_df[calib_mask].reset_index(drop=True)

X_train = np.vstack(train_df['features'].apply(lambda d: np.array([d[k] for k in feature_keys])))
y_train = train_df['pnl'].values

# ---------------------------------------------
# 4. Train PnL Prediction Model on Train Set
# ---------------------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'objective': 'reg:squarederror', 'max_depth': 4, 'eta': 0.1, 'verbosity': 0}
f_theta = xgb.train(params, dtrain, num_boost_round=100)

# Check if calibration set has signals; if empty, use full hist for calibration
if calib_df.empty:
    print("Calibration set is empty; using full historical for calibration.")
    calib_df = hist_df.copy()
    calib_df['score'] = f_theta.predict(xgb.DMatrix(X_all))
else:
    X_calib = np.vstack(calib_df['features'].apply(lambda d: np.array([d[k] for k in feature_keys])))
    calib_df['score'] = f_theta.predict(xgb.DMatrix(X_calib))

# ---------------------------------------------
# 5. Baseline Slot-Threshold Calibration on Calibration Set
# ---------------------------------------------
calib_sorted = calib_df.sort_values(['day', 'time_idx']).reset_index(drop=True)

def get_slot_pnls(slot, T_candidate, T_baseline, df_sorted):
    slot_pnls = []
    for day, group in df_sorted.groupby('day'):
        slot_idx = 1
        for _, row in group.iterrows():
            s = row['score']
            if slot_idx < slot:
                if s >= T_baseline[slot_idx]:
                    slot_idx += 1
                continue
            elif slot_idx == slot:
                if s >= T_candidate:
                    slot_pnls.append(row['pnl'])
                    break
                else:
                    continue
            else:
                break
    return slot_pnls

def calibrate_threshold_for_slot(slot, df_sorted, T_baseline, target_pnl=-0.0005):
    all_scores = df_sorted['score'].values
    lo, hi = np.min(all_scores), np.max(all_scores)
    tol = 1e-4
    best_T = lo
    for _ in range(25):
        mid = (lo + hi) / 2
        slot_pnls = get_slot_pnls(slot, mid, T_baseline, df_sorted)
        if len(slot_pnls) == 0:
            hi = mid
            continue
        avg_pnl = np.mean(slot_pnls)
        if abs(avg_pnl - target_pnl) < tol:
            best_T = mid
            break
        if avg_pnl > target_pnl:
            lo = mid
        else:
            hi = mid
        best_T = mid
    return best_T

T_baseline = {i: 0.0 for i in range(1, 6)}
for k in range(1, 6):
    T_baseline[k] = calibrate_threshold_for_slot(k, calib_sorted, T_baseline, target_pnl=-0.0005)

print("Calibrated Baseline Thresholds (slot → score):")
for k in range(1, 6):
    print(f"  Slot {k}: {T_baseline[k]:.5f}")

# ---------------------------------------------
# 6. Retrain PnL Model on Full Historical for Live
# ---------------------------------------------
dtrain_full = xgb.DMatrix(X_all, label=y_all)
f_theta_full = xgb.train(params, dtrain_full, num_boost_round=100)

# ---------------------------------------------
# 7. Live Trading Simulation with Adjusted Thresholds
# ---------------------------------------------
STOP_LOSS_PCT = 0.002
RR = 2.0
DELTA_PERF = 0.005
KAPPA = 0.1
BUDGET = 5

min_hist['ret_1m'] = min_hist.groupby('date')['price'].pct_change().fillna(0)
min_hist['vol_30m'] = min_hist.groupby('date')['ret_1m'].rolling(window=30).std().reset_index(level=0, drop=True).fillna(0)
vol_hist_mean = min_hist['vol_30m'].mean()

prices_live = min_live['price'].values
n_live = len(prices_live)
min_live['slopes'] = compute_slope(prices_live, window=10)
min_live['ret_1m'] = min_live['price'].pct_change().fillna(0)
min_live['ret_5m'] = min_live['price'].pct_change(5).fillna(0)
min_live['vol_10m'] = min_live['ret_1m'].rolling(window=10).std().fillna(0)
min_live['min_to_close'] = (min_live['datetime'].max() - min_live['datetime']).dt.total_seconds() / 60

sec_live_day = sec_live[sec_live['date'] == live_date].copy()
sec_live_day['minute'] = sec_live_day['datetime'].dt.floor('T')
sec_live_day['ret_s'] = sec_live_day['price_s'].pct_change().fillna(0)
sec_live_day['sign'] = np.sign(sec_live_day['ret_s'])
sec_live_day['sign_flip'] = sec_live_day['sign'] != sec_live_day['sign'].shift(1)
sec_live_feat = sec_live_day.groupby(['date', 'minute']).agg({'ret_s': 'std', 'sign_flip': 'sum'})
sec_live_feat.columns = ['vol_60s', 'flips_60s']
sec_live_feat = sec_live_feat.reset_index()

min_live = pd.merge(min_live, sec_live_feat, left_on=['date', 'datetime'], right_on=['date', 'minute'], how='left').fillna(0).drop(columns=['minute'])

slot_today = 0
pnl_cum = 0.0
trade_logs = []

for i, row in min_live.iterrows():
    if slot_today >= BUDGET:
        break
    
    current = row['price']
    future = prices_live[i:]
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
        i / n_live
    ])
    
    score_t = f_theta_full.predict(xgb.DMatrix(features.reshape(1, -1)))[0]
    k = slot_today + 1
    if k > BUDGET:
        break
    
    if pnl_cum >= 0:
        delta_perf = -DELTA_PERF
    else:
        delta_perf = DELTA_PERF
    
    if i >= 29:
        vol30 = min_live['ret_1m'].iloc[i-29:i+1].std()
    else:
        vol30 = vol_hist_mean
    delta_mkt = KAPPA * (vol30 - vol_hist_mean)
    
    T_today = T_baseline[k] + delta_perf + delta_mkt
    
    if score_t < T_today:
        continue
    
    entry_price = current
    direction_exec = -1 if (i > 0 and prices_live[i-1] < entry_price) else 1
    pnl = simulate_trade_pnl(prices_live, i, stop_loss_pct=STOP_LOSS_PCT, rr=RR, trailing=True)
    
    slot_today += 1
    pnl_cum += pnl
    trade_logs.append({
        'time_idx': i,
        'entry_time': row['datetime'],
        'entry_price': entry_price,
        'slot': k,
        'score': score_t,
        'threshold': T_today,
        'realized_pnl': pnl
    })

trades_df = pd.DataFrame(trade_logs)

# ---------------------------------------------
# 8. Results and Diagnostics
# ---------------------------------------------
print("\nLive Trading Simulation Results:")
print(f"Number of Trades: {len(trades_df)}")
if not trades_df.empty:
    print(f"Total PnL: {trades_df['realized_pnl'].sum():.5f}")
    print(f"Win Rate: {(trades_df['realized_pnl'] > 0).mean() * 100:.2f}%\n")
    print(trades_df.to_string(index=False))

plt.figure(figsize=(12, 5))
plt.plot(min_live['datetime'], min_live['price'], label='Price (Live Day)')
for idx, t in trades_df.iterrows():
    plt.scatter(min_live['datetime'].iloc[t['time_idx']], 
                min_live['price'].iloc[t['time_idx']],
                marker='o', edgecolors='r', facecolors='none',
                label='Executed Trade' if idx == 0 else "")
plt.title("Live Day Price with Executed Trades")
plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 1. Generate synthetic second‐level data for two days (market open 09:30 to 16:00)
def generate_second_data(start_date, seed=None):
    # np.random.seed(seed)
    # define market open/close times
    day_start = pd.Timestamp(start_date + ' 09:30:00')
    n_seconds = 6.5 * 3600  # 6.5 hours in seconds
    date_rng = pd.date_range(start=day_start, periods=int(n_seconds), freq='S')
    # simulate second‐level returns
    returns = np.random.normal(loc=0.0000, scale=0.01, size=len(date_rng))
    price = 100 + np.cumsum(returns)
    df = pd.DataFrame({'datetime': date_rng, 'price_s': price})
    df['date'] = df['datetime'].dt.date
    return df

sec1 = generate_second_data('2025-05-19', seed=1)
sec2 = generate_second_data('2025-05-20', seed=2)
sec = pd.concat([sec1, sec2]).reset_index(drop=True)

# 2. Aggregate to minute‐level for labeling and minute‐level features
min_df = sec.set_index('datetime').groupby('date').resample('1Min').agg({'price_s':'mean'}).reset_index()
min_df.rename(columns={'price_s': 'price'}, inplace=True)

# 3. Generate robust first‐reversal labels on minute data
def generate_labels(group, delta=0.002, slope_window=10):
    prices = group['price'].values
    n = len(prices)
    y_rev = np.zeros(n, dtype=int)
    y_range = np.zeros(n, dtype=float)
    # rolling slope
    slopes = np.zeros(n)
    for i in range(n):
        start = max(0, i - slope_window)
        x = np.arange(start, i+1)
        y = prices[start:i+1]
        slopes[i] = linregress(x, y).slope if len(x)>1 else 0.0
    for i in range(n):
        current = prices[i]
        future = prices[i:]
        direction = 1 if slopes[i]>0 else -1
        pct = (future - current) / current
        # first reversal
        rev_idxs = np.where(pct <= -delta if direction==1 else pct >= delta)[0]
        if rev_idxs.size>0 and rev_idxs[0]>0:
            y_rev[i] = 1
        # full remaining range
        y_range[i] = (future.max() - future.min()) / current
    group['y_rev'] = y_rev
    group['y_range'] = y_range
    return group

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
    group['slope'] = slopes
    return group

min_df = min_df.groupby('date', group_keys=False).apply(generate_strict_reversal_labels)

# 4. Feature engineering: minute‐level
min_df['ret_1m'] = min_df.groupby('date')['price'].pct_change(1).fillna(0)
min_df['ret_5m'] = min_df.groupby('date')['price'].pct_change(5).fillna(0)
min_df['vol_10m'] = (min_df.groupby('date')['ret_1m']
                    .rolling(10).std()
                    .reset_index(level=0, drop=True)
                    .fillna(0))
min_df['min_to_close'] = min_df.groupby('date')['datetime']\
    .transform(lambda x: (x.max() - x).dt.total_seconds()/60)

# 5. Feature engineering: second‐level aggregated to minute
# compute per‐minute window over last 60 seconds
sec['ret_s'] = sec.groupby('date')['price_s'].pct_change().fillna(0)
# volatility in last 60s and count of sign flips
sec['sign'] = np.sign(sec['ret_s'])
sec['sign_flip'] = sec['sign'] != sec.groupby('date')['sign'].shift(1)
# aggregate for each minute
sec_feat = sec.groupby([sec['date'], sec['datetime'].dt.floor('T')]).agg({
    'ret_s': ['std'],
    'sign_flip': 'sum'
})
sec_feat.columns = ['vol_60s', 'flips_60s']
sec_feat = sec_feat.reset_index().rename(columns={'datetime': 'minute'})

# merge second‐level features into minute DF
min_df = pd.merge(min_df,
                  sec_feat,
                  left_on=['date', 'datetime'],
                  right_on=['date', 'minute'],
                  how='left').drop(columns=['minute'])

# 6. Prepare training data
features = ['ret_1m', 'ret_5m', 'vol_10m', 'min_to_close', 'vol_60s', 'flips_60s']
X = min_df[features].values
y_rev = min_df['y_rev'].values
y_range = min_df['y_range'].values
# stack for multi‐task
X_dup = np.vstack([X, X])
y_flat = np.concatenate([y_rev, y_range])
dtrain = xgb.DMatrix(X_dup, label=y_flat)

# 7. Multi‐task custom objective
def multitask_obj(preds, dtrain):
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

# 8. Train model
params = {'max_depth':4, 'eta':0.1, 'verbosity':0}
bst = xgb.train(params, dtrain, num_boost_round=50, obj=multitask_obj)

# 9. Predictions & evaluation
preds = bst.predict(dtrain)
N = len(min_df)
prob_rev = 1/(1+np.exp(-preds[:N]))
min_df['pred_rev'] = (prob_rev>=0.5).astype(int)
accuracy = (min_df['pred_rev']==min_df['y_rev']).mean()
print(f"Reversal accuracy: {accuracy:.2%}")

# 10. Plot for first day
fd = min_df
plt.figure(figsize=(10,4))
plt.plot(fd['datetime'], fd['price'], label='Price')
plt.scatter(fd['datetime'][fd['y_rev']==1], fd['price'][fd['y_rev']==1], 
            marker='x', label='True Rev')
plt.scatter(fd['datetime'][fd['pred_rev']==1], fd['price'][fd['pred_rev']==1], 
            facecolors='none', edgecolors='r', label='Pred Rev')
plt.title(f"{fd['date'].iloc[0]} (Accuracy {accuracy:.2%})")
plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); 

# Trading parameters
delta = 0.002  # stop-loss percent
rr = 2.0       # reward-to-risk ratio
min_df['position'] = 0  # 1 for long, -1 for short, 0 flat

# Simulation variables
trades = []
in_trade = False

for idx, row in min_df.iterrows():
    if not in_trade and row['pred_rev'] == 1:
        # Open trade
        direction = -1 if row['slope'] > 0 else 1
        entry_price = row['price']
        sl_price = entry_price * (1 - direction * delta)
        tp_price = entry_price + direction * (delta * rr * entry_price)
        in_trade = True
        trade = {
            'entry_idx': idx,
            'entry_price': entry_price,
            'direction': direction,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'exit_idx': None,
            'exit_price': None,
            'outcome': None
        }
    elif in_trade:
        price = row['price']
        # Check stop loss
        if direction == 1 and price <= sl_price:
            trade.update({'exit_idx': idx, 'exit_price': price, 'outcome': 'SL'})
            trades.append(trade)
            in_trade = False
        elif direction == -1 and price >= sl_price:
            trade.update({'exit_idx': idx, 'exit_price': price, 'outcome': 'SL'})
            trades.append(trade)
            in_trade = False
        # Check take profit
        elif direction == 1 and price >= tp_price:
            trade.update({'exit_idx': idx, 'exit_price': tp_price, 'outcome': 'TP'})
            trades.append(trade)
            in_trade = False
        elif direction == -1 and price <= tp_price:
            trade.update({'exit_idx': idx, 'exit_price': tp_price, 'outcome': 'TP'})
            trades.append(trade)
            in_trade = False
    # record position
    min_df.at[idx, 'position'] = trade['direction'] if in_trade else 0

# Build DataFrame of trades
trades_df = pd.DataFrame(trades)
# Compute P&L
trades_df['pnl'] = trades_df['direction'] * (trades_df['exit_price'] - trades_df['entry_price'])

# Summary metrics
win_rate = (trades_df['outcome'] == 'TP').mean()
avg_pnl = trades_df['pnl'].mean()
print(f"Number of trades: {len(trades_df)}")
print(f"Win rate: {win_rate:.2%}")
print(f"Average P&L per trade: {avg_pnl:.4f}")

# Plot cumulative P&L over time
trades_df['exit_time'] = trades_df['exit_idx'].apply(lambda i: min_df.loc[i, 'datetime'])
trades_df = trades_df.sort_values('exit_time')
trades_df['cum_pnl'] = trades_df['pnl'].cumsum()

plt.figure(figsize=(10,4))
plt.plot(trades_df['exit_time'], trades_df['cum_pnl'], marker='o')
plt.title("Cumulative P&L from Execution Logic")
plt.xlabel("Time")
plt.ylabel("Cumulative P&L")
plt.grid(True)
plt.tight_layout()
plt.show()




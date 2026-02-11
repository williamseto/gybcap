import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime


data_filename = '../test_seconds_td0.csv'
sec = pd.read_csv(data_filename)

dt_format_str = "%m/%d/%Y %H:%M:%S"
sec['dt'] = sec.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", dt_format_str), axis=1)

# 2. Aggregate to minute‐level for labeling and minute‐level features
min_df = sec.set_index('dt').groupby('trading_day').resample('1Min').agg({'Close':'mean'}).bfill().reset_index()
min_df.rename(columns={'Close': 'price'}, inplace=True)


# 3. Generate robust first‐reversal labels on minute data
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

min_df = min_df.groupby('trading_day', group_keys=False).apply(generate_strict_reversal_labels)

# 4. Feature engineering: minute‐level
min_df['ret_1m'] = min_df.groupby('trading_day')['price'].pct_change(1).fillna(0)
min_df['ret_5m'] = min_df.groupby('trading_day')['price'].pct_change(5).fillna(0)
min_df['vol_10m'] = (min_df.groupby('trading_day')['ret_1m']
                    .rolling(10).std()
                    .reset_index(level=0, drop=True)
                    .fillna(0))
min_df['min_to_close'] = min_df.groupby('trading_day')['dt']\
    .transform(lambda x: (x.max() - x).dt.total_seconds()/60)


# 5. Feature engineering: second‐level aggregated to minute
# compute per‐minute window over last 60 seconds
sec['ret_s'] = sec.groupby('trading_day')['Close'].pct_change().fillna(0)
# volatility in last 60s and count of sign flips
sec['sign'] = np.sign(sec['ret_s'])
sec['sign_flip'] = sec['sign'] != sec.groupby('trading_day')['sign'].shift(1)
# aggregate for each minute

sec_feat = sec.set_index('dt').groupby('trading_day').resample('1Min').agg({
     'ret_s': ['std'],
    'sign_flip': 'sum'
})
sec_feat.columns = ['vol_60s', 'flips_60s']
sec_feat = sec_feat.reset_index().rename(columns={'dt': 'minute'})

# merge second‐level features into minute DF
min_df = pd.merge(min_df,
                  sec_feat,
                  left_on=['trading_day', 'dt'],
                  right_on=['trading_day', 'minute'],
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
fd = min_df[min_df['trading_day']==min_df['trading_day'].unique()[0]]
plt.figure(figsize=(10,4))
plt.plot(fd['dt'], fd['price'], label='Price')
plt.scatter(fd['dt'][fd['y_rev']==1], fd['price'][fd['y_rev']==1], 
            marker='x', label='True Rev')
plt.scatter(fd['dt'][fd['pred_rev']==1], fd['price'][fd['pred_rev']==1], 
            facecolors='none', edgecolors='r', label='Pred Rev')
plt.title(f"{fd['trading_day'].iloc[0]} (Accuracy {accuracy:.2%})")
plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

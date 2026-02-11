import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 1. Generate synthetic minute-price data for two days
def generate_price_data(start_date, n_minutes=500, seed=None):
    # np.random.seed(seed)
    date_rng = pd.date_range(start=start_date, periods=n_minutes, freq='T')
    returns = np.random.normal(loc=0.000, scale=0.01, size=n_minutes)
    price = 100 + np.cumsum(returns)
    df = pd.DataFrame({'datetime': date_rng, 'price': price})
    df['date'] = df['datetime'].dt.date
    return df

df1 = generate_price_data('2025-05-19', seed=42)
df2 = generate_price_data('2025-05-20', seed=24)
df = pd.concat([df1, df2]).reset_index(drop=True)

# 2. Improved label generation: first reversal only, based on smoothed slope
def generate_first_reversal_labels(group, delta=0.002, slope_window=10):
    prices = group['price'].values
    n = len(prices)
    y_rev = np.zeros(n, dtype=int)
    y_range = np.zeros(n, dtype=float)

    # Compute rolling slope (trend) for each minute
    slopes = np.zeros(n)
    for i in range(n):
        start = max(0, i - slope_window)
        x = np.arange(start, i+1)
        y = prices[start:i+1]
        slopes[i] = linregress(x, y).slope if len(x)>1 else 0.0

    for i in range(n):
        current = prices[i]
        future = prices[i:]
        direction = 1 if slopes[i] > 0 else -1
        # percent changes
        pct = (future - current) / current
        if direction == 1:
            # look for first drop >= delta
            idxs = np.where(pct <= -delta)[0]
        else:
            # look for first rise >= delta
            idxs = np.where(pct >= delta)[0]
        if idxs.size > 0:
            # first reversal event
            y_rev[i] = 1
        # full-day remaining range
        y_range[i] = (future.max() - future.min()) / current

    group['y_rev'] = y_rev
    group['y_range'] = y_range
    return group

def generate_filtered_reversal_labels(group, delta=0.002, slope_window=10, min_slope=0.001):
    """
    Label the first reversal only if prior trend (slope) magnitude exceeds min_slope.
    - group: DataFrame for a single trading day with 'price' column.
    - delta: reversal threshold (fractional).
    - slope_window: look-back window for slope estimation.
    - min_slope: minimum absolute slope to consider a valid trend.
    """
    prices = group['price'].values
    n = len(prices)
    y_rev = np.zeros(n, dtype=int)
    y_range = np.zeros(n, dtype=float)

    # Compute rolling slope for trend
    slopes = np.zeros(n)
    for i in range(n):
        start = max(0, i - slope_window)
        x = np.arange(start, i + 1)
        y = prices[start:i + 1]
        slopes[i] = linregress(x, y).slope if len(x) > 1 else 0.0

    for i in range(n):
        current_price = prices[i]
        # skip labeling if trend too weak
        if abs(slopes[i]) < min_slope:
            y_range[i] = (prices[i:].max() - prices[i:].min()) / current_price
            continue

        # determine direction from slope
        direction = 1 if slopes[i] > 0 else -1
        # full future series
        future = prices[i:]
        pct_changes = (future - current_price) / current_price

        # find first reversal index
        if direction == 1:
            rev_idxs = np.where(pct_changes <= -delta)[0]
        else:
            rev_idxs = np.where(pct_changes >= delta)[0]

        if rev_idxs.size > 0 and rev_idxs[0] > 0:
            y_rev[i] = 1

        y_range[i] = (future.max() - future.min()) / current_price

    group['y_rev'] = y_rev
    group['y_range'] = y_range
    return group

def generate_strict_reversal_labels(group, delta=0.002, slope_window=30, min_slope=1e-4):
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

df = df.groupby('date', group_keys=False).apply(generate_strict_reversal_labels)

# 3. Feature engineering
df['return_1'] = df.groupby('date')['price'].pct_change(1).fillna(0)
df['return_5'] = df.groupby('date')['price'].pct_change(5).fillna(0)
df['rolling_std_10'] = (
    df.groupby('date')['return_1']
      .rolling(window=10).std()
      .reset_index(level=0, drop=True)
      .fillna(0)
)
df['minutes_to_close'] = df.groupby('date')['datetime'] \
    .transform(lambda x: (x.max() - x).dt.total_seconds() / 60)

# Drop last minute of day
df['idx'] = df.groupby('date').cumcount()
df = df[df['idx'] < df.groupby('date')['idx'].transform('max')].copy()

# 4. Prepare training data
feature_cols = ['return_1', 'return_5', 'rolling_std_10', 'minutes_to_close']
X = df[feature_cols].values
y_rev = df['y_rev'].values
y_range = df['y_range'].values

# 5. Duplicate for multi-task
X_dup = np.vstack([X, X])
labels_flat = np.concatenate([y_rev, y_range])
dtrain = xgb.DMatrix(X_dup, label=labels_flat)

# 6. Custom objective
def multi_task_obj(preds, dtrain):
    N = len(preds) // 2
    logit_rev = preds[:N]
    pred_range = preds[N:]
    labels = dtrain.get_label()
    y_rev = labels[:N]
    y_range = labels[N:]

    p = 1 / (1 + np.exp(-logit_rev))
    grad_rev = p - y_rev
    hess_rev = p * (1 - p)

    grad_range = pred_range - y_range
    hess_range = np.ones_like(grad_range)

    alpha, beta = 1.0, 1.0
    grad = np.concatenate([alpha * grad_rev, beta * grad_range])
    hess = np.concatenate([alpha * hess_rev, beta * hess_range])
    return grad, hess

# 7. Train model
params = {'max_depth': 4, 'eta': 0.1, 'verbosity': 0}
bst = xgb.train(params, dtrain, num_boost_round=50, obj=multi_task_obj)

# 8. Predict & evaluate
preds = bst.predict(dtrain)
N = len(df)
pred_rev_prob = 1 / (1 + np.exp(-preds[:N]))
df['pred_rev'] = (pred_rev_prob >= 0.6).astype(int)
accuracy = (df['pred_rev'] == df['y_rev']).mean()

# 9. Plot for first day
first_date = df['date'].unique()[0]
sub = df[df['date'] == first_date]
plt.figure(figsize=(10,4))
plt.plot(sub['datetime'], sub['price'], label='Price')
plt.scatter(sub['datetime'][sub['y_rev']==1], sub['price'][sub['y_rev']==1], marker='x', label='True Reversal', color='green')
plt.scatter(sub['datetime'][sub['pred_rev']==1], sub['price'][sub['pred_rev']==1], marker='o', label='Predicted', color='red')
plt.title(f"{first_date} Price & Reversals (Accuracy: {accuracy:.2%})")
plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
plt.show()

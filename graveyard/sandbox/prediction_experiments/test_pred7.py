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
        np.random.seed(seed_base + day_offset)
        current_date = pd.to_datetime(start_date) + pd.Timedelta(days=day_offset)
        day_start = current_date + pd.Timedelta(hours=9, minutes=30)
        n_seconds = int(6.5 * 3600)
        date_rng = pd.date_range(start=day_start, periods=n_seconds, freq='S')
        returns = np.random.normal(loc=0.0, scale=0.01, size=n_seconds)
        price = 100 + np.cumsum(returns)
        df = pd.DataFrame({'datetime': date_rng, 'price_s': price})
        df['date'] = df['datetime'].dt.date
        all_seconds.append(df)
    return pd.concat(all_seconds).reset_index(drop=True)

# Generate 10 days of data for demonstration
sec_all = generate_second_data('2025-05-01', num_days=10, seed_base=42)

# ---------------------------------------------
# 2. Aggregate to Minute Bars and Compute Minute Features & Labels
# ---------------------------------------------
# Aggregate second data to minute level
min_df = sec_all.copy()
min_df['datetime'] = min_df['datetime'].dt.floor('T')
min_df = min_df.groupby(['date', 'datetime']).last().reset_index()
min_df.rename(columns={'price_s': 'price'}, inplace=True)


# Compute slopes and minute-level features
def compute_slope(arr, window=10):
    n = len(arr)
    slopes = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        x = np.arange(start, i + 1)
        y = arr[start:i + 1]
        slopes[i] = linregress(x, y).slope if len(x) > 1 else 0.0
    return slopes

min_df[['minute', 'slopes', 'ret_1m', 'ret_5m', 'vol_10m', 'time_of_day_norm']] = 0.0
minute_rows = []
for day in sorted(min_df['date'].unique()):

    df_day = min_df[min_df['date'] == day].copy()
    prices = df_day['price'].values
    n = len(prices)
    slopes = compute_slope(prices, window=10)
    df_day['slopes'] = slopes
    df_day['ret_1m'] = df_day['price'].pct_change().fillna(0)
    df_day['ret_5m'] = df_day['price'].pct_change(5).fillna(0)
    df_day['vol_10m'] = df_day['ret_1m'].rolling(10).std().fillna(0)
    df_day['time_of_day_norm'] = np.arange(n) / n

    min_df[min_df['date'] == day] = df_day
    df_day.reset_index(drop=True, inplace=True)


    for t in range(n):
        direction = 1 if slopes[t] > 0 else -1
        future = prices[t:]
        pct = (future - prices[t]) / prices[t]
        rev_idxs = np.where(pct <= -0.002 if direction == 1 else pct >= 0.002)[0]
        label = 1 if (rev_idxs.size > 0 and rev_idxs[0] > 0) else 0

        features = [
            df_day.loc[t, 'slopes'],
            df_day.loc[t, 'ret_1m'],
            df_day.loc[t, 'ret_5m'],
            df_day.loc[t, 'vol_10m'],
            df_day.loc[t, 'time_of_day_norm']
        ]
        minute_rows.append({
            'day': day,
            'time_idx': t,
            'datetime': df_day.loc[t, 'datetime'],
            'features_min': features,
            'label_rev': label
        })

minute_df = pd.DataFrame(minute_rows)


# ---------------------------------------------
# 3. Train Minute‐Level Reversal Classifier (Stage 1)
# ---------------------------------------------
X_min = np.vstack(minute_df['features_min'].values)
y_min = minute_df['label_rev'].values
dmin = xgb.DMatrix(X_min, label=y_min)
clf_params = {'objective': 'binary:logistic', 'max_depth': 4, 'eta': 0.1}
bst_min = xgb.train(clf_params, dmin, num_boost_round=100)


# Add predicted p_rev_min into minute_df
minute_df['p_rev_min'] = bst_min.predict(xgb.DMatrix(X_min))

# ---------------------------------------------
# 4. Build Second‐Level Dataset (Stage 2 Input)
# ---------------------------------------------
# We will define second‐level reversal label as any ±0.2% move within next 60 seconds.
sec_rows = []

# Precompute second‐level features: rolling vol_10s & flip count_10s
for day in sorted(sec_all['date'].unique()):
    sec_day = sec_all[sec_all['date'] == day].reset_index(drop=True)
    sec_day['ret_s'] = sec_day['price_s'].pct_change().fillna(0)
    sec_day['sign'] = np.sign(sec_day['ret_s'])
    sec_day['sign_flip'] = sec_day['sign'] != sec_day['sign'].shift(1)
    sec_day['vol_10s'] = sec_day['ret_s'].rolling(10).std().fillna(0)
    sec_day['flips_10s'] = sec_day['sign_flip'].rolling(10).sum().fillna(0)
    sec_day['sec_idx'] = np.arange(len(sec_day))


    # Merge minute‐level p_rev_min & features to each second
    # Determine minute index for each second
    sec_day['minute'] = sec_day['datetime'].dt.floor('T')
    df_day = min_df[min_df['date'] == day].reset_index(drop=True)
    df_day['minute'] = df_day['datetime']


    merged_min = pd.merge(sec_day, df_day[['minute', 'slopes', 'ret_1m', 'ret_5m', 'vol_10m', 'time_of_day_norm']],
                          on='minute', how='left').fillna(method='ffill')
    # Merge on p_rev_min
    merged_min = pd.merge(merged_min, minute_df[['datetime', 'p_rev_min']],
                          left_on='minute', right_on='datetime', how='left').fillna(method='ffill')
    merged_min.drop(columns=['datetime_y'], inplace=True)
    merged_min.rename(columns={'datetime_x': 'datetime'}, inplace=True)


    prices_sec = merged_min['price_s'].values
    n_sec = len(prices_sec)

    for i in range(n_sec):
        current_price = prices_sec[i]
        future = prices_sec[i:i + 60]  # next 60 seconds
        if len(future) < 2:
            label_sec = 0
        else:
            # use slope at corresponding minute
            direction = 1 if merged_min.loc[i, 'slopes'] > 0 else -1
            pct = (future - current_price) / current_price
            rev_idxs = np.where(pct <= -0.002 if direction == 1 else pct >= 0.002)[0]
            label_sec = 1 if (rev_idxs.size > 0 and rev_idxs[0] > 0) else 0

        features_sec = [
            merged_min.loc[i, 'p_rev_min'],       # minute‐model probability
            merged_min.loc[i, 'slopes'],          # minute slope
            merged_min.loc[i, 'ret_1m'],          # minute 1m return
            merged_min.loc[i, 'ret_5m'],          # minute 5m return
            merged_min.loc[i, 'vol_10m'],         # minute 10m vol
            merged_min.loc[i, 'time_of_day_norm'],# minute time norm
            merged_min.loc[i, 'vol_10s'],         # second‐level vol (10s)
            merged_min.loc[i, 'flips_10s']        # second‐level flips (10s)
        ]
        sec_rows.append({
            'day': day,
            'sec_idx': i,
            'datetime': merged_min.loc[i, 'datetime'],
            'features_sec': features_sec,
            'label_rev_sec': label_sec,
            'price': current_price
        })


sec_df = pd.DataFrame(sec_rows)


# ---------------------------------------------
# 5. Compute Base Margin from Minute Model
# ---------------------------------------------
X_sec_min = np.vstack(sec_df['features_sec'].values)[:, 1:6]

dsec_min = xgb.DMatrix(X_sec_min)
base_margin = bst_min.predict(dsec_min, output_margin=True)

# ---------------------------------------------
# 5. Train Second‐Level Classifier on Full Features Using Base Margin
# ---------------------------------------------
X_sec_full = np.vstack(sec_df['features_sec'].values)[:, 1:]


y_sec = sec_df['label_rev_sec'].values
dsec_full = xgb.DMatrix(X_sec_full, label=y_sec)
dsec_full.set_base_margin(base_margin)

sec_clf_params = {
    'objective':   'binary:logistic',
    'max_depth':   4,
    'eta':         0.1,
    'base_score':  0.5   # <-- must be in (0,1)
}

bst_sec = xgb.train(sec_clf_params, dsec_full, num_boost_round=50)

# Predictions to compare
sec_df['p_rev_sec'] = bst_sec.predict(xgb.DMatrix(X_sec_full))


# # ---------------------------------------------
# # 5. Train / Fine‐Tune Second‐Level Reversal Classifier
# # ---------------------------------------------
# # Start by reusing minute‐model's booster as initial model
# X_sec = np.vstack(sec_df['features_sec'].values)
# y_sec = sec_df['label_rev_sec'].values
# dsec = xgb.DMatrix(X_sec, label=y_sec)

# # Continue training from minute‐level booster (transfer learning)
# # For XGBoost, specify the previous booster as 'xgb_model'
# bst_sec = xgb.train(clf_params, dsec, num_boost_round=50, xgb_model=bst_min)

# # Add predicted p_rev_sec to sec_df
# sec_df['p_rev_sec'] = bst_sec.predict(xgb.DMatrix(X_sec))


# ---------------------------------------------
# 6. Sample Evaluation: Compare Minute vs Second Predictions
# ---------------------------------------------
# Take a slice of one day to visualize
day0 = sorted(sec_all['date'].unique())[0]
sec_day0 = sec_df[sec_df['day'] == day0].reset_index(drop=True)

preds = sec_df['p_rev_sec']
N = len(preds)
prob_rev = 1/(1+np.exp(-preds[:N]))
pred_rev = (prob_rev>=0.5).astype(int)

fd = sec_day0
fd['pred_rev'] = pred_rev[sec_df['day'] == day0]

plt.plot(fd['datetime'], fd['price'], label='Price')
plt.scatter(fd['datetime'][fd['label_rev_sec']==1], fd['price'][fd['label_rev_sec']==1], 
            marker='x', label='True Rev', color='red')
# plt.scatter(fd['datetime'][fd['pred_rev']==1], fd['price'][fd['pred_rev']==1], 
#             facecolors='none', edgecolors='r', label='Pred Rev')
plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

exit()

# Plot p_rev_min (mapped from minute to seconds), p_rev_sec, and true label on first 1000 seconds
# Map minute predictions to seconds
map_min_to_sec = minute_df[minute_df['day'] == day0][['time_idx', 'p_rev_min']]
map_min_to_sec['start_sec'] = map_min_to_sec['time_idx'] * 60
map_min_to_sec['end_sec'] = map_min_to_sec['start_sec'] + 59
sec_day0['p_rev_min_mapped'] = 0.0

for _, row in map_min_to_sec.iterrows():
    mask = (sec_day0['sec_idx'] >= row['start_sec']) & (sec_day0['sec_idx'] <= row['end_sec'])
    sec_day0.loc[mask, 'p_rev_min_mapped'] = row['p_rev_min']

plt.figure(figsize=(12, 5))
window = 1000
plt.plot(sec_day0['sec_idx'][:window], sec_day0['p_rev_min_mapped'][:window], label='Minute Model p_rev')
plt.plot(sec_day0['sec_idx'][:window], sec_day0['p_rev_sec'][:window], label='Second Model p_rev')
plt.scatter(sec_day0['sec_idx'][:window][sec_day0['label_rev_sec'][:window] == 1],
            sec_day0['p_rev_sec'][:window][sec_day0['label_rev_sec'][:window] == 1],
            color='red', s=10, label='True Reversal (sec)')
plt.xlabel('Second Index (Day 0)')
plt.ylabel('Predicted p_rev')
plt.title('Minute vs Second Model Predictions (First 1000 Seconds)')
plt.legend()
plt.tight_layout()
plt.show()

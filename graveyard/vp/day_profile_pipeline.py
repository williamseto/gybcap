#!/usr/bin/env python3
"""
Minimal end-to-end pipeline:
- label historical days with deterministic Dalton-style rules
- extract per-minute cumulative features (online-safe)
- train a day-wise classifier (sample-weight option to favor early minutes)
- evaluate per-minute accuracy and earliest-minute-to-target
- save model + scaler, and provide incremental scorer function
"""

import argparse, os, math, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

### ---------------------------
### Utilities: VbP & profile ops
### ---------------------------
def build_minute_vbp_matrix(prices, volumes, bin_size=0.5, kernel=(0.2,0.6,0.2)):
    """
    Fast vectorized build of per-minute VbP rows.
    - prices, volumes: arrays length T
    - returns bin_centers, per_minute_vbp (T x n_bins)
    """
    minp, maxp = prices.min() - 5, prices.max() + 5
    bins = np.arange(math.floor(minp), math.ceil(maxp) + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    n_bins = len(bin_centers)
    T = len(prices)
    per_minute_vbp = np.zeros((T, n_bins), dtype=float)

    # nearest-bin index per minute
    idxs = np.searchsorted(bins, prices, side='right') - 1
    idxs = np.clip(idxs, 0, n_bins-1)

    # distribute volume into center +/-1 using kernel
    offsets = [-1, 0, +1]
    for offset, weight in zip(offsets, kernel):
        idxs_off = np.clip(idxs + offset, 0, n_bins-1)
        per_minute_vbp[np.arange(T), idxs_off] += volumes * weight

    return bin_centers, per_minute_vbp

def compute_va70_from_v(vbp, bin_centers):
    """Return (va_low, va_high) for given vbp vector."""
    total = vbp.sum()
    if total <= 0:
        return float(bin_centers[0]), float(bin_centers[-1])
    poc_idx = int(np.argmax(vbp))
    cum = vbp[poc_idx]
    low = poc_idx; high = poc_idx
    target = 0.7 * total
    while cum < target:
        left = vbp[low-1] if low-1 >= 0 else -1
        right = vbp[high+1] if high+1 < len(vbp) else -1
        if left >= right:
            low -= 1
            cum += vbp[low]
        else:
            high += 1
            cum += vbp[high]
        if low == 0 and high == len(vbp)-1:
            break
    return float(bin_centers[low]), float(bin_centers[high])

def entropy_of_v(v):
    s = v.sum()
    if s <= 0:
        return 0.0
    p = v / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def simple_peak_count(v):
    # simple local maxima count (fast)
    if len(v) < 3:
        return 0
    left = v[:-2]; center = v[1:-1]; right = v[2:]
    return int(((center > left) & (center > right)).sum())

### ---------------------------
### Deterministic labeler
### ---------------------------
def deterministic_day_label(prices, volumes, bin_size=0.5):
    """
    Dalton-style rules (tunable):
    - Double if VbP has >=2 peaks separated by valley
    - Trend if VA70 narrow and POC near edge
    - Balance otherwise
    """
    bc, per_minute_vbp = build_minute_vbp_matrix(prices, volumes, bin_size=bin_size)
    total_vbp = per_minute_vbp.sum(axis=0)
    poc_price = bc[int(np.argmax(total_vbp))]
    va_low, va_high = compute_va70_from_v(total_vbp, bc)
    va_width = va_high - va_low
    day_range = max(1.0, prices.max() - prices.min())
    poc_rel = (poc_price - prices.min()) / day_range
    # crude peak detection
    peaks = simple_peak_count(total_vbp)
    if peaks >= 2:
        return 'Double'
    # Trend heuristics (tunable thresholds)
    if va_width < 0.35 * day_range and (poc_rel < 0.2 or poc_rel > 0.8):
        return 'Trend'
    return 'Balance'

### ---------------------------
### Feature extractor (per-minute)
### ---------------------------
def extract_per_minute_features(prices, volumes, bin_size=0.5):
    """
    Return DataFrame with per-minute features computed only using data up to that minute.
    Columns include:
      minute, last_price, price_vs_vwap, ret1, ret5, cum_vol, vol_share,
      poc_price, poc_rel, va_width, va_width_rel, entropy, bimodal_count
    """
    T = len(prices)
    bc, per_minute_vbp = build_minute_vbp_matrix(prices, volumes, bin_size=bin_size)
    cum_vbp = np.cumsum(per_minute_vbp, axis=0)
    cum_vol = np.cumsum(volumes)
    cum_pv = np.cumsum(prices * volumes)
    vwap = cum_pv / np.maximum(1, cum_vol)
    day_range = max(1.0, prices.max() - prices.min())

    rows = []
    for t in range(T):
        vbp_t = cum_vbp[t]
        poc_idx = int(np.argmax(vbp_t))
        poc_price = float(bc[poc_idx])
        va_low, va_high = compute_va70_from_v(vbp_t, bc)
        va_width = va_high - va_low
        poc_rel = (poc_price - prices.min()) / day_range
        ent = entropy_of_v(vbp_t)
        last_price = float(prices[t])
        ret1 = float((prices[t] - prices[t-1]) / prices[t-1]) if t >= 1 else 0.0
        ret5 = float((prices[t] - prices[t-6]) / prices[t-6]) if t >= 6 else 0.0
        vol_share = float(cum_vol[t] / max(1, volumes.sum()))
        bcount = simple_peak_count(vbp_t)
        rows.append({
            'minute': t,
            'last_price': last_price,
            'vwap': float(vwap[t]),
            'price_vs_vwap': last_price - float(vwap[t]),
            'ret1': ret1, 'ret5': ret5,
            'cum_vol': float(cum_vol[t]), 'vol_share': vol_share,
            'poc_price': poc_price, 'poc_rel': poc_rel,
            'va_width': va_width, 'va_width_rel': va_width / day_range,
            'entropy': ent, 'bimodal_count': bcount
        })
    return pd.DataFrame(rows)

### ---------------------------
### Pipeline runner
### ---------------------------
def make_dataset_from_grouped_minute_bars(df_bars, bin_size=0.5):
    """
    Input:
        df_bars with columns ['date','minute','close','volume'] (minute runs 0..T-1)
    Output:
        df_samples: rows for each minute with features + day label + date
    """
    samples = []
    dates = sorted(df_bars['date'].unique())
    for date in dates:
        day = df_bars[df_bars['date'] == date].sort_values('minute')
        prices = day['close'].values.astype(float)
        vols = day['volume'].values.astype(float)
        label = deterministic_day_label(prices, vols, bin_size=bin_size)
        feat_df = extract_per_minute_features(prices, vols, bin_size=bin_size)
        feat_df['date'] = date
        feat_df['label'] = label
        samples.append(feat_df)
    return pd.concat(samples, ignore_index=True)

def train_and_evaluate(df_samples, feature_cols=None, output_dir='artifacts', early_weight=1.0, seed=42):
    """
    - feature_cols: list of columns to feed the model; if None we use a default.
    - early_weight: float >=1. 1.0 means uniform weighting. >1 increases weight on early minutes (see below).
      Implementation: weight = 1 + (early_weight-1) * (1-minute_norm) ** 2
    """
    os.makedirs(output_dir, exist_ok=True)
    if feature_cols is None:
        feature_cols = ['last_price','price_vs_vwap','ret1','ret5','cum_vol','vol_share',
                        'poc_price','poc_rel','va_width','va_width_rel','entropy','bimodal_count','minute']
    X = df_samples[feature_cols].copy()
    # normalize minute into [0,1]
    X['minute'] = X['minute'] / X['minute'].max()
    y = df_samples['label']
    groups = df_samples['date']

    # day-wise split (no leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed+1)
    train_idx2, val_idx = next(split2.split(X.iloc[train_idx], y.iloc[train_idx], groups=groups.iloc[train_idx]))
    train_idx_final = train_idx[train_idx2]
    val_idx_final = train_idx[val_idx]

    X_train = X.iloc[train_idx_final].reset_index(drop=True)
    y_train = y.iloc[train_idx_final].reset_index(drop=True)
    X_val = X.iloc[val_idx_final].reset_index(drop=True)
    y_val = y.iloc[val_idx_final].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_test = groups.iloc[test_idx].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # sample weighting to favor early minutes:
    # minute_norm in [0,1] where 0==open -> weight high if early_weight>1
    minute_norm = X_train['minute'].values
    sample_weight = 1.0 + (early_weight - 1.0) * (1.0 - minute_norm) ** 2

    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    # Save artifacts
    joblib.dump({'model': clf, 'scaler': scaler, 'features': feature_cols}, os.path.join(output_dir, 'day_classifier.joblib'))

    # Evaluate: per-minute accuracy on test days
    X_test_scaled = scaler.transform(X_test)
    preds = clf.predict(X_test_scaled)
    df_test = X_test.copy()
    df_test['label'] = y_test.values
    df_test['pred'] = preds
    df_test['date'] = groups_test.values

    acc_by_minute = df_test.groupby('minute').apply(lambda g: accuracy_score(g['label'], g['pred']))
    acc_df = acc_by_minute.reset_index()
    acc_df.columns = ['minute', 'accuracy']
    acc_df.to_csv(os.path.join(output_dir, 'acc_by_minute.csv'), index=False)

    # end-of-day performance and confusion
    eod = df_test[df_test['minute'] == df_test['minute'].max()]
    report = classification_report(eod['label'], eod['pred'])
    cm = confusion_matrix(eod['label'], eod['pred'])
    print("End-of-day classification report:\n", report)
    print("Confusion matrix (rows=true):\n", cm)

    # earliest minute where accuracy >= thresholds
    for thr in [0.6, 0.7, 0.75, 0.8]:
        found = acc_df[acc_df['accuracy'] >= thr]
        if not found.empty:
            print(f"earliest minute where accuracy >= {thr:.2f}: {int(found['minute'].min())}")
        else:
            print(f"no minute reached accuracy >= {thr:.2f}")

    # plot and save
    plt.figure(figsize=(10,4))
    plt.plot(acc_df['minute'], acc_df['accuracy'])
    plt.xlabel('Minute of day (0..T-1)')
    plt.ylabel('Accuracy (test set) at minute')
    plt.title('Per-minute accuracy on test set')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'acc_by_minute.png'), dpi=150)
    plt.close()

    return {'clf': clf, 'scaler': scaler, 'feature_cols': feature_cols, 'acc_df': acc_df, 'test_df': df_test}

### ---------------------------
### Incremental scorer (realtime)
### ---------------------------
def make_feature_row_from_incremental_state(bin_centers, cum_vbp, cum_vol, cum_pv, last_price, minute_index, total_day_volume_estimate=None):
    """
    Given cumulative VbP (array), cumulative volume scalar, cumulative PV scalar, produce the feature row
    that matches train-time features (note: same names/order must be maintained).
    Returns dict of features.
    """
    poc_idx = int(np.argmax(cum_vbp))
    poc_price = float(bin_centers[poc_idx])
    va_low, va_high = compute_va70_from_v(cum_vbp, bin_centers)
    va_width = va_high - va_low
    day_min_estimate = bin_centers.min()  # crude; in production track day min precisely
    day_range_estimate = max(1.0, bin_centers.max() - day_min_estimate)
    poc_rel = (poc_price - day_min_estimate) / day_range_estimate
    entropy = entropy_of_v(cum_vbp)
    # compute vwap from cum_pv and cum_vol
    vwap = float(cum_pv / max(1.0, cum_vol))
    # minute must be normalized outside or match training - here return raw minute (normalize later)
    feature_row = {
        'last_price': float(last_price),
        'price_vs_vwap': float(last_price - vwap),
        'ret1': 0.0, # caller can fill with a rolling price buffer if available
        'ret5': 0.0,
        'cum_vol': float(cum_vol),
        'vol_share': float(cum_vol / max(1.0, total_day_volume_estimate)) if total_day_volume_estimate else 0.0,
        'poc_price': poc_price,
        'poc_rel': poc_rel,
        'va_width': va_width,
        'va_width_rel': va_width / day_range_estimate,
        'entropy': entropy,
        'bimodal_count': simple_peak_count(cum_vbp),
        'minute': minute_index
    }
    return feature_row

### ---------------------------
### Demo / CLI
### ---------------------------
def simulate_days_to_minute_bars(n_days=200):
    """Small simulator used for testing - returns dataframe with columns date, minute, close, volume"""
    rng = np.random.RandomState(123)
    rows = []
    for d in range(n_days):
        # choose a label probabilistically (for variety)
        r = rng.rand()
        if r < 0.45: lab = 'Balance'
        elif r < 0.80: lab = 'Trend'
        else: lab = 'Double'
        base = 4000 + (d % 50) * 0.1
        T = 390
        if lab == 'Trend':
            drift = rng.choice([0.02, -0.02])
            prices = base + np.cumsum(drift + rng.normal(scale=0.3, size=T))
        elif lab == 'Balance':
            prices = np.zeros(T)
            x = base
            for t in range(T):
                x += 0.2*(base-x) + rng.normal(scale=0.5)
                prices[t] = x
        else:
            split = T//2 + rng.randint(-20,20)
            low_center = base - 10 - rng.rand()*4
            high_center = base + 10 + rng.rand()*4
            prices = np.concatenate([rng.normal(low_center, 1.0, split), rng.normal(high_center, 1.0, T-split)])
        vols = rng.poisson(150, size=T) + 20
        for t in range(T):
            rows.append({'date': f'd{d}', 'minute': t, 'close': float(prices[t]), 'volume': int(vols[t])})
    return pd.DataFrame(rows)

def main(args):
    if args.simulate:
        print("Simulating minute bars (demo)...")
        df_bars = simulate_days_to_minute_bars(n_days=args.days)
    else:
        df_bars = pd.read_csv(args.input, parse_dates=False)
        # expect date, minute, close, volume
        assert {'date','minute','close','volume'}.issubset(set(df_bars.columns)), "Input CSV must contain date,minute,close,volume"

    print("Building dataset (per-minute features, deterministic labels)...")
    df_samples = make_dataset_from_grouped_minute_bars(df_bars, bin_size=args.bin_size)

    print("Samples:", len(df_samples), "unique days:", df_samples['date'].nunique())
    artifacts = train_and_evaluate(df_samples, output_dir=args.output_dir, early_weight=args.early_weight)
    print("Saved artifacts to:", args.output_dir)
    print("Model: ", os.path.join(args.output_dir, 'day_classifier.joblib'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, help='CSV path with columns date,minute,close,volume (minute 0..T-1)')
    p.add_argument('--output-dir', type=str, default='artifacts', help='Where to save model + reports')
    p.add_argument('--simulate', action='store_true', help='Run simulated demo data instead of reading CSV')
    p.add_argument('--days', type=int, default=120, help='Number of simulated days (demo mode)')
    p.add_argument('--bin-size', type=float, default=0.5)
    p.add_argument('--early-weight', type=float, default=1.0,
                   help='>=1.0. 1.0=uniform. >1 upweights early-minute samples to teach early-day accuracy.')
    args = p.parse_args()
    main(args)

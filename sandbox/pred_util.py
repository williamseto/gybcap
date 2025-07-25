
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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
    def __init__(self, data, dt_idx='datetime', day_idx='date', range_df=None, params=None):

        self.sec_df = data
        self.dt_idx = dt_idx
        self.day_idx = day_idx
        self.range_df = range_df

        if params is None:
            self.params = {
                'objective': 'binary:logistic',
                'max_depth': 12,
                'eta': 0.1,
                'verbosity': 1
            }
        else:
            self.params = params

    def compute_minute_features(self, sec_df):

        agg_dict = {
            'price_s': 'mean',
            'TickDataSaver:Buys': 'sum',
            'TickDataSaver:Sells': 'sum',
            'Volume': 'sum'
        }
        min_df = sec_df.set_index(self.dt_idx).groupby(self.day_idx).resample('1Min').agg(agg_dict).bfill().reset_index()
        min_df.rename(columns={'TickDataSaver:Buys': 'buys', 'TickDataSaver:Sells': 'sells'}, inplace=True)

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

        min_df['delta_m'] = (min_df['buys'] - min_df['sells'])
        min_df['cum_delta_m'] = min_df.groupby(self.day_idx)['delta_m'].cumsum()

        min_df['vol_20m_avg'] = min_df.groupby(self.day_idx)['Volume'].transform(lambda x: x.rolling(20).mean()).fillna(1)

        min_df['buy_vol_pct'] = min_df['buys'] / min_df['vol_20m_avg']
        min_df['sell_vol_pct'] = min_df['sells'] / min_df['vol_20m_avg']

        min_df['curr_lod'] = min_df.groupby(self.day_idx)['price'].cummin()
        min_df['curr_hod'] = min_df.groupby(self.day_idx)['price'].cummax()

        min_df['dist_to_lod'] = min_df['price'] - min_df['curr_lod']
        min_df['dist_to_hod'] = min_df['curr_hod'] - min_df['price']

        def get_range_dists(group):
            day = group['dt'].dt.date.iloc[-1]

            try:
                pred_range_pct = self.range_df[self.range_df['Date'].dt.date==day]['Predicted_RangePct'].iloc[0] / 100.0
            except:
                # If no range data available for this day, use a default value
                pred_range_pct = 0.005

            # pred_range_pct = (group['price'].max() - group['price'].min()) / group['price'].iloc[0]

            pred_range_high = group['price'].iloc[0] * (1 + pred_range_pct * 0.5)
            pred_range_low = group['price'].iloc[0] * (1 - pred_range_pct * 0.5)

            group['dist_to_max'] = (pred_range_high - group['price']) / group['price']
            group['dist_to_min'] = (group['price'] - pred_range_low) / group['price']

            return group

        if self.range_df is not None:
            min_df = min_df.groupby(self.day_idx, group_keys=False).apply(get_range_dists)
        
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
        features = ['ret_1m', 'ret_5m', 'vol_10m', 'min_to_close', 'dist_to_max', 'dist_to_min', 'delta_m', 'cum_delta_m', 'Volume', 'buy_vol_pct', 'sell_vol_pct', 'dist_to_lod', 'dist_to_hod', 'ml_pred_rev']
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
    

class ReversalDataset(Dataset):

    def __init__(self, df, back_bars=60, front_bars=15):
        self.segments = []


        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'TickDataSaver:Buys', 'TickDataSaver:Sells']

        N = len(df)
        W = back_bars + front_bars + 1


        def grab_window(row):
            i = df.index.get_loc(row.name)
            start, end = i-back_bars, i+front_bars+1

            if start < 0 or end > len(df):
                return None

            if df.iloc[start]['trading_day'] != df.iloc[end]['trading_day']:
                return None

            segment = df.iloc[start:end][feature_cols].to_numpy().T

            price_open = segment[0, 0]
            segment[:4, :] = (segment[:4, :] - price_open) / price_open    # normalize all price channels by open
            segment[4, :]  = np.log1p(segment[4, :])                       # log‑volume

            return segment

        windows = (
            df.loc[df['y_rev']==1]
            .apply(grab_window, axis=1)
            .dropna()
            .tolist()
        )
        self.X = np.stack(windows, axis=0)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32)


# --- TCN Autoencoder definition -----------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=pad, dilation=dilation)

        self.conv = nn.Sequential(self.conv, nn.BatchNorm1d(out_ch))

        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.relu)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.net(x)
        out = out[:, :, :x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNAutoencoder(nn.Module):
    def __init__(self, in_ch=5, num_levels=4, kernel_size=3, hidden_ch=64):
        super().__init__()
        # encoder
        enc_layers = []
        chs = [in_ch] + [hidden_ch] * num_levels
        for i in range(num_levels):
            enc_layers.append(TCNBlock(chs[i], chs[i+1], kernel_size, dilation=2**i))
        self.encoder = nn.Sequential(*enc_layers)
        # bottleneck pooling
        self.pool = nn.AdaptiveMaxPool1d(1)
        # decoder: mirror of encoder
        dec_layers = []
        rev_chs = [hidden_ch] + list(reversed(chs[:-1]))
        for i in range(num_levels):
            dec_layers.append(
                nn.ConvTranspose1d(rev_chs[i], rev_chs[i+1], kernel_size,
                                   dilation=2**(num_levels-1-i),
                                   padding=(kernel_size-1)*(2**(num_levels-1-i)))
            )
            dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        # x: [B, C, L]
        h = self.encoder(x)
        z = self.pool(h)  # [B, hidden_ch, 1]
        z_rep = z.repeat(1, 1, x.size(2))
        x_rec = self.decoder(z_rep)
        return x_rec, z.squeeze(-1)  # return reconstruction and latent
    
# --- Training & introspection -------------------------------------------

def train_ae(model, loader, opt, loss_fn, device):
    model.train()
    total = 0
    for X in loader:
        X = X.to(device)
        opt.zero_grad()
        X_rec, _ = model(X)
        loss = loss_fn(X_rec, X)
        loss.backward()
        opt.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def compute_latent_and_error(model, loader, device):
    model.eval()
    latents = []
    errors = []
    for X in loader:
        X = X.to(device)
        X_rec, z = model(X)
        err = ((X_rec - X)**2).mean(dim=[1,2])  # MSE per sample
        latents.append(z.cpu().numpy())
        errors.append(err.cpu().numpy())
    return np.vstack(latents), np.concatenate(errors)

# ——— 0. Helpers: Local & Global Encoders —————————————
class LocalTCNEncoder(nn.Module):
    def __init__(self, in_ch=5, hidden_ch=64, levels=4):
        super().__init__()
        # same as your TCNClassifier’s encoder up to pooling
        layers = []
        chs = [in_ch] + [hidden_ch]*levels
        for i in range(levels):
            layers.append(TCNBlock(chs[i], chs[i+1], kernel_size=3, dilation=2**i))
        self.net  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        h = self.net(x)             # [B, hidden_ch, L]
        z = self.pool(h).squeeze(-1)  # [B, hidden_ch]
        return z

class GlobalCNNEncoder(nn.Module):
    def __init__(self, in_ch=5, hidden_ch=32):
        super().__init__()
        # assume global input is heavily downsampled (e.g. 30 bars of 1‑min OHLCV)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,  16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16,     32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
    def forward(self, x):
        h = self.net(x)            # [B, 32, 1]
        return h.squeeze(-1)       # [B, 32]

# ——— 1. Fusion Heads —————————————————————————
class MLPFusion(nn.Module):
    def __init__(self, d1, d2, hidden=64, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d1+d2, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, z1, z2):
        return self.fc(torch.cat([z1, z2], dim=1))

class AttentionFusion(nn.Module):
    def __init__(self, d_query, d_kv, d_model=64, num_classes=2):
        super().__init__()
        self.q_proj = nn.Linear(d_query, d_model, bias=False)
        self.k_proj = nn.Linear(d_kv,    d_model, bias=False)
        self.v_proj = nn.Linear(d_kv,    d_model, bias=False)
        self.out    = nn.Linear(d_model + d_query, num_classes)
    def forward(self, z_local, z_global):
        # [B, d_model]
        q = self.q_proj(z_local)         # query
        k = self.k_proj(z_global)        # key
        v = self.v_proj(z_global)        # value
        # scalar attention score for each sample
        attn = torch.softmax((q * k).sum(dim=1, keepdim=True) / (q.size(1)**0.5), dim=1)
        # fuse: weighted global info
        z_attn = attn * v                # [B, d_model]
        # final logits
        return self.out(torch.cat([z_local, z_attn], dim=1))

# ——— 2. Two‑Stream Wrapper ——————————————————————
class TwoStreamModel(nn.Module):
    def __init__(self, local_enc, global_enc, fusion_head):
        super().__init__()
        self.local_enc  = local_enc
        self.global_enc = global_enc
        self.fusion     = fusion_head
    def forward(self, x_local, x_global):
        z1 = self.local_enc(x_local)
        z2 = self.global_enc(x_global)
        return self.fusion(z1, z2)


from sklearn.metrics import accuracy_score

def train_twostream_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds, all_y = [], []
    for x_loc, x_glo, y in loader:
        x_loc, x_glo, y = x_loc.to(device), x_glo.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x_loc, x_glo)
        loss   = loss_fn(logits, y)
        loss.backward(); opt.step()
        total_loss += loss.item() * y.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_y.append(y.cpu())
    preds = torch.cat(all_preds)
    truth = torch.cat(all_y)
    return total_loss/len(truth), accuracy_score(truth, preds)

@torch.no_grad()
def eval_twostream_model(model, loader, device):
    model.eval()
    all_preds, all_y = [], []
    for x_loc, x_glo, y in loader:
        x_loc, x_glo, y = x_loc.to(device), x_glo.to(device), y.to(device)
        logits = model(x_loc, x_glo)
        all_preds.append(logits.argmax(1).cpu())
        all_y.append(y.cpu())
    preds = torch.cat(all_preds)
    truth = torch.cat(all_y)
    return accuracy_score(truth, preds)

class TwoStreamReversalDataset(Dataset):

    def __init__(self, sec_df, min_df):
        self.segments = []

        self.feature_cols = ['Close', 'Volume', 'TickDataSaver:Buys', 'TickDataSaver:Sells']

        self.min_feature_cols = ['price', 'Volume', 'buys', 'sells']

        self.sec_df = sec_df
        self.min_df = min_df

        def get_indices(row):
            i = sec_df.index.get_loc(row.name)
            start = i - 60
            end = i + 15 + 1

            if start < 0 or end > len(sec_df) - 1:
                return None

            if sec_df.iloc[start]['trading_day'] != sec_df.iloc[end]['trading_day']:
                return None

            dt_sec = sec_df.iloc[i]['dt'].floor('min')

            row_min = min_df.loc[min_df['dt'] == dt_sec]
            if row_min.empty:
                return None

            min_idx = row_min.index[0]
            start_min = min_idx - 30

            if start_min < 0:
                return None
            
            if min_df.iloc[start_min]['trading_day'] != min_df.iloc[min_idx]['trading_day']:
                return None

            return row.name

        self.indices = sec_df.apply(get_indices, axis=1).dropna()


    def normalize_price(self, segment):
        mean_price = np.mean(segment[0, :])
        std_price = np.maximum(np.std(segment[0, :]), 1e-6)
        segment[0, :] = (segment[0, :] - mean_price) / std_price  # z‑score price

        segment[1:, :]  = np.log1p(segment[1:, :])                       # log‑volume

        segment[1, :] = (segment[1, :] - np.mean(segment[1, :])) / (np.std(segment[1, :]) + 1e-6)  # z‑score volume
        segment[2, :] = (segment[2, :] - np.mean(segment[2, :])) / (np.std(segment[2, :]) + 1e-6)  # z‑score buys
        segment[3, :] = (segment[3, :] - np.mean(segment[3, :])) / (np.std(segment[3, :]) + 1e-6)  # z‑score sells

        return segment


    def grab_window(self, row_idx):
        i = int(row_idx)
        start = i - 60
        end = i + 15 + 1

        if start < 0 or end > len(self.sec_df) - 1:
            return None

        if self.sec_df.iloc[start]['trading_day'] != self.sec_df.iloc[end]['trading_day']:
            return None

        dt_sec = self.sec_df.iloc[i]['dt'].floor('min')

        row_min = self.min_df.loc[self.min_df['dt'] == dt_sec]
        if row_min.empty:
            return None

        min_idx = row_min.index[0]
        start_min = min_idx - 30

        if start_min < 0:
            return None

        global_segment = self.min_df.iloc[start_min:min_idx][self.min_feature_cols].to_numpy().T

        global_segment = self.normalize_price(global_segment)


        local_segment = self.sec_df.iloc[start:end][self.feature_cols].to_numpy().T

        local_segment = self.normalize_price(local_segment)

        label = self.sec_df.iloc[i]['y_rev']


        return local_segment, global_segment, label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        local_window, global_window, label = self.grab_window(self.indices.iloc[i])

        return torch.tensor(local_window, dtype=torch.float32), torch.tensor(global_window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


from numpy.lib.stride_tricks import sliding_window_view

class TwoStreamReversalDatasetVec(Dataset):

    def __init__(self, sec_df, min_df):

        self.sec_df = sec_df
        self.min_df = min_df
        
        self.sec_df['price'] = self.sec_df['Close']
        self.sec_df['Volume'] = self.sec_df['Volume']
        self.sec_df['buys'] = self.sec_df['TickDataSaver:Buys']
        self.sec_df['sells'] = self.sec_df['TickDataSaver:Sells']

        self.L_sec = 60 + 15 + 1
        self.L_min = 30

        self.sec_df = self.precompute_norms(self.sec_df, self.L_sec)
        self.min_df = self.precompute_norms(self.min_df, self.L_min)

        self.norm_feature_cols = ['price_z', 'vol_z', 'buys_z', 'sells_z']

        self.feature_cols = ['Close', 'Volume', 'TickDataSaver:Buys', 'TickDataSaver:Sells']

        self.min_feature_cols = ['price', 'Volume', 'buys', 'sells']

        self.precompute_windows()

    def precompute_norms(self, df, L_window):

        # 1a) price: rolling mean/std over the same length & centered
        df['price_mean'] = df['price'].rolling(window=L_window, center=True, min_periods=1).mean()
        df['price_std']  = df['price'].rolling(window=L_window, center=True, min_periods=1).std().clip(lower=1e-6)
        
        # 1b) log-1p volume & then its rolling mean/std
        df['log_vol'] = np.log1p(df['Volume'])
        df['vol_mean'] = df['log_vol'].rolling(window=L_window, center=True, min_periods=1).mean()
        df['vol_std']  = df['log_vol'].rolling(window=L_window, center=True, min_periods=1).std().add(1e-6)

        # 1c) same for buys & sells
        for col in ['buys', 'sells']:
            df[f'{col}_mean'] = df[col].rolling(window=L_window, center=True, min_periods=1).mean()
            df[f'{col}_std'] = df[col].rolling(window=L_window, center=True, min_periods=1).std().add(1e-6)

        # price z-score
        df['price_z'] = (df['price'] - df['price_mean']) / df['price_std']

        # volume z-score of log1p(vol)
        df['vol_z']   = (df['log_vol'] - df['vol_mean']) / df['vol_std']

        # buys & sells z-scores
        for col in ['buys','sells']:
            df[f'{col}_z'] = (df[col] - df[f'{col}_mean']) / df[f'{col}_std']

        return df
            

    def precompute_windows(self):
        # 1) Pull out raw arrays once
        sec_arr   = self.sec_df[self.norm_feature_cols].to_numpy()            # (N_sec, D_sec)
        min_arr   = self.min_df[self.norm_feature_cols].to_numpy()        # (N_min, D_min)
        td_sec    = self.sec_df['trading_day'].to_numpy()                # (N_sec,)
        td_min    = self.min_df['trading_day'].to_numpy()                # (N_min,)
        dt_floor  = self.sec_df['dt'].dt.floor('min')
        y_rev_all = self.sec_df['y_rev'].to_numpy()

        # 2) Minute‐index lookup
        minute_indexer = pd.Series(np.arange(len(self.min_df)), index=self.min_df['dt'])
        sec_min_idx = dt_floor.map(minute_indexer).to_numpy()  # (N_sec,)

        # 3) Build sliding‐window **views**
        L_local  = self.L_sec

        sec_w    = sliding_window_view(sec_arr, window_shape=L_local, axis=0)
        #   sec_w[i] covers sec_arr[i : i+L_local];
        #   we want windows centered on original row i+60, so shift:
        sec_w    = sec_w[60 : sec_arr.shape[0] - 15]           # shape (N_centered, L_local, D_sec)
        # aligns center→second-row

        L_global = self.L_min
        min_w    = sliding_window_view(min_arr, window_shape=L_global, axis=0)  # (N_min - 29, 30, D_min)

        # 4) Compute for each centered‐window its minute‐window start index
        # idx      = np.arange(0, sec_arr.shape[0] - 16)        # maps sec_w → original row i
        idx = np.arange(0, sec_w.shape[0] - 61)
        starts   = sec_min_idx[idx] - L_global                # desired min-window start
        ends     = sec_min_idx[idx]                          # end_min (exclusive) index


        # Mask 1: must lie in bounds of minute‐windows
        valid_gl = (starts >= 0) & (starts < min_w.shape[0])  # valid minute‐windows

        # Mask 2: seconds‐window day‐boundary check (same as before)
        same_day_sec = (td_sec[idx - 60] == td_sec[idx + 16])

        # Mask 3: **new** minute‐window day‐boundary check
        #          trading_day at start_min must equal at end_min
        same_day_min = (td_min[starts] == td_min[ends])

        # Combine all masks
        valid = valid_gl & same_day_sec & same_day_min

        self.valid_idx = np.nonzero(valid)[0]                       # indices into sec_w/min_w/idx arrays

        # tst_dbg = self.sec_df.iloc[self.valid_idx]
        # tst_dbg.to_csv('tst_dbg.csv')

        # 7) Precompute raw labels aligned to sec_w
        self.labels_base = y_rev_all[idx]                          # length N_centered

        self.starts = starts
        self.sec_w = sec_w
        self.min_w = min_w


    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, i):
        idx = self.valid_idx[i]
        local_chunk = self.sec_w[idx]
        global_chunk = self.min_w[self.starts[idx]]
        label = self.labels_base[idx]
        
        return torch.tensor(local_chunk, dtype=torch.float32), torch.tensor(global_chunk, dtype=torch.float32), torch.tensor(label, dtype=torch.long), idx

class TwoStreamReversalModel:
    def __init__(self, device):

        # architect parameters
        d1, d2 = 64, 32    # local hidden_ch, global hidden_ch

        # instantiate two models
        self.mlp_model = TwoStreamModel(
            LocalTCNEncoder(in_ch=4, hidden_ch=d1),
            GlobalCNNEncoder(in_ch=4, hidden_ch=d2),
            MLPFusion(d1, d2)
        ).to(device)

        # attn_model = TwoStreamModel(
        #     LocalTCNEncoder(in_ch=5, hidden_ch=d1),
        #     GlobalCNNEncoder(in_ch=5, hidden_ch=d2),
        #     AttentionFusion(d1, d2)
        # ).to(device)

        # training setup
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim_mlp  = torch.optim.Adam(self.mlp_model.parameters(),  lr=1e-3)
        # optim_attn = torch.optim.Adam(attn_model.parameters(), lr=1e-3)

        self.device = device

    def load_model(self, path):
        self.mlp_model.load_state_dict(torch.load(path, map_location=self.device))


    def train(self, loader):
        for epoch in range(1, 31):
            mlp_loss, mlp_acc = train_twostream_epoch(self.mlp_model, loader, self.optim_mlp, self.loss_fn, self.device)
            # val_acc_mlp      = eval_model(mlp_model, val_loader, device)

            print(f"Epoch {epoch:02d} | MLP ▶ train_loss={mlp_loss:.3f}, acc={mlp_acc:.3f}")

            # att_loss, att_acc = train_epoch(attn_model, train_loader, optim_attn, loss_fn, device)
            # val_acc_attn      = eval_model(attn_model, val_loader, device)

            # print(f"Epoch {epoch:02d} | "
            #       f"MLP ▶ train_loss={mlp_loss:.3f}, val_acc={val_acc_mlp:.3f} | "
            #       f"ATT ▶ train_loss={att_loss:.3f}, val_acc={val_acc_attn:.3f}")
        
        torch.save(self.mlp_model.state_dict(), 'twostream_mlp_model.pth')

    def predict(self, loader, pred_len):
        all_probs = torch.zeros(pred_len)

        self.mlp_model.eval()
        with torch.no_grad():
            for x_loc, x_glo, y, idx_batch in loader:
                print(idx_batch[0].item())
                x_loc, x_glo, y = x_loc.to(self.device), x_glo.to(self.device), y.to(self.device)
                logits  = self.mlp_model(x_loc, x_glo)
                probs  = F.softmax(logits, dim=1)[:,1].cpu()
                all_probs[idx_batch] = probs

        return all_probs.numpy()


from pred_util import *
from datetime import datetime, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cpu')

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

range_filename = 'range_predictions.csv'
range_df = pd.read_csv(range_filename)
range_df['Date'] = pd.to_datetime(range_df['Date'])


data_filename = '../test_seconds_td0.csv'
sec_df = pd.read_csv(data_filename)

dt_format_str = "%m/%d/%Y %H:%M:%S"
# sec_df['dt'] = sec_df.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", dt_format_str), axis=1)
sec_df['dt'] = pd.to_datetime(sec_df['Date'] + ' ' + sec_df['Time'], format=dt_format_str)

sec_df['price_s'] = sec_df['Close']

dt_idx = 'dt'
day_idx = 'trading_day'

rev_model = ReversalModel(sec_df, dt_idx=dt_idx, day_idx=day_idx, range_df=range_df)

rev_model_features = rev_model.compute_features()

seconds_labels_df = pd.merge_asof(
    sec_df,
    rev_model.min_df[['dt', 'y_rev']],
    left_on='dt',
    right_on='dt',
    direction='backward'
)

full_ds = TwoStreamReversalDatasetVec(seconds_labels_df, rev_model.min_df)

loader = DataLoader(full_ds, batch_size=32)

ml_rev_model = TwoStreamReversalModel(device)

seconds_labels_df['ml_pred_rev'] = ml_rev_model.predict(loader, len(seconds_labels_df))

seconds_labels_df[['Date','Time','Open','High','Low','Close','Volume','TickDataSaver:Buys','TickDataSaver:Sells','ovn','trading_day','ml_pred_rev']].to_csv('test_sec.csv', index=False)
exit()

# rev_model.min_df['ml_pred_rev'] = seconds_labels_df.set_index('dt').groupby('trading_day').resample('1Min').agg({'ml_pred_rev':'last'}).reset_index()['ml_pred_rev']

rev_model.train()

prob_rev, _ = rev_model.predict(rev_model.min_df)

rev_model.bst.save_model('rev_model.json')

exit()

# min_df = rev_model.min_df
# fd = min_df[min_df[day_idx]==min_df[day_idx].unique()[0]]
# plt.figure(figsize=(10,4))
# plt.plot(fd[dt_idx], fd['price'], label='Price')
# plt.scatter(fd[dt_idx][fd['y_rev']==1], fd['price'][fd['y_rev']==1], 
#             marker='x', label='True Rev')
# plt.scatter(fd[dt_idx][fd['pred_rev']==1], fd['price'][fd['pred_rev']==1], 
#             facecolors='none', edgecolors='r', label='Pred Rev')
# plt.title(f"{fd[day_idx].iloc[0]} ")
# plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.tight_layout(); plt.show()

# exit()


utility_model = UtilityModel(rev_model.min_df, rev_model.min_df['pred_rev'], day_idx=day_idx)

utility_model.train()

utility_model.f_theta_full.save_model('utility_model.json')


from backtesting import Strategy
from backtesting import Backtest
from backtesting.test import SMA

STOP_LOSS_PCT = 0.001
RR = 2.0
DELTA_PERF = 0.005
KAPPA = 0.1
BUDGET = 5

class UtilityStrategy(Strategy):

    utility_model = None
    
    def init(self):

        self.sma5 = self.I(SMA, self.data.Close, 5)

        self.unique_dates = sorted(self.data.df[day_idx].unique())

        self.curr_day_idx = 0

        self.curr_day = self.unique_dates[self.curr_day_idx]


        self.in_trade = False
        self.slot_today = 0

    
    def next(self):

        row_date = self.data.df[self.utility_model.day_idx].iloc[-1]

        if self.data.dt[-1].astype('M8[s]').item().time() >= time(13, 00):
            if self.position:
                self.position.close()
            return


        if row_date != self.curr_day:
            self.curr_day_idx += 1
            self.curr_day = self.unique_dates[self.curr_day_idx]
            self.slot_today = 0

        
        if self.position:

            price = self.data.Close[-1]

            if abs(self.position.size) == 1:

                if self.position.is_long:
                    self.trades[0].sl = max(self.trades[0].entry_price, price * (1 - STOP_LOSS_PCT * RR))
                elif self.position.is_short:
                    self.trades[0].sl = min(self.trades[0].entry_price, price * (1 + STOP_LOSS_PCT * RR))

            # maybe exit trade if got reversal signal in other direction
            # future = prices[i:]
            # direction = 1 if row['slopes'] > 0 else -1
            # pct = (future - price) / price
            # rev_idxs = np.where(pct <= -0.002 if direction == 1 else pct >= 0.002)[0]
            # if rev_idxs.size > 0 and rev_idxs[0] == 0 and direction == -direction_exec:
            #     util = (price - entry_price) * direction_exec / (entry_price * STOP_LOSS_PCT)
            #     in_trade = False; pnl_cum += util
            #     trades[-1].update({'exit_time': row['datetime'], 'exit_price': price, 'util': util})
            #     continue
            
            return
        
        if self.slot_today >= BUDGET:
            # goto next day
            return
        
        # check if we can enter a new trade
        row = self.data.df.iloc[[-1]]
    
        # features = ['ret_1m', 'ret_5m', 'vol_10m', 'min_to_close', 'vol_60s', 'flips_60s']

        prob_rev, contribs = rev_model.predict(row)
        pred_rev = (prob_rev>=0.3).astype(int)

        if pred_rev[0] == 0:
            return
        
        features = row[utility_model.features].values

        score_t = self.utility_model.f_theta_full.predict(xgb.DMatrix(features.reshape(1, -1)))[0]
        k = self.slot_today + 1
        
        delta_perf = -DELTA_PERF if self.equity >= 10000 else DELTA_PERF
        # if i >= 29:
        #     vol30 = test_group['ret_1m'].iloc[i-29:i+1].std()
        # else:
        #     vol30 = vol_hist_mean
        # delta_mkt = KAPPA * (vol30 - vol_hist_mean)
        delta_mkt = 0
        
        T_today = self.utility_model.T_base[k] + delta_perf + delta_mkt

        T_today = np.clip(T_today, a_min=None, a_max=0.0)

        if score_t < T_today:
            return
        
        # enter trade
        sma_slope = compute_slope(self.sma5, window=5)

        direction_exec = -1 if sma_slope > 0 else 1

        self.entry_price = self.data.Close[-1]

        sl_price = self.entry_price * (1 - direction_exec * STOP_LOSS_PCT)
        tp_price = self.entry_price * (1 + direction_exec * STOP_LOSS_PCT * RR)


        if direction_exec == 1:
            self.buy(size=1.0, sl=sl_price)
            self.buy(size=1.0, sl=sl_price, tp=tp_price)

        else:
            self.sell(size=1.0, sl=sl_price)
            self.sell(size=1.0, sl=sl_price, tp=tp_price)



        self.trail_sl = sl_price
        self.tp_price = tp_price
        self.sl_price = sl_price
        
        self.slot_today += 1

    

    


bt_df = utility_model.data_df.copy()
bt_df['Open'] = bt_df['price']
bt_df['High'] = bt_df['price']
bt_df['Low'] = bt_df['price']
bt_df['Close'] = bt_df['price']

bt = Backtest(bt_df, UtilityStrategy, cash=10000, commission=0.0)
stats = bt.run(utility_model=utility_model)
print(stats)
bt.plot()





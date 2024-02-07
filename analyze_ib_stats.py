
import pandas as pd
import numpy as np
import pdb
from scipy import stats, signal
import matplotlib.pyplot as plt

stats_df = pd.read_csv('test_time_stats_full.csv')

trading_day_groups = stats_df.groupby(['trading_day'])

real_num_days = len(trading_day_groups)

td_offset = 1
# expansion_days = []
balance_days = []
bal_tst = 0

lo_ext = []
hi_ext = []
for i in range(len(trading_day_groups)):

    td = i + td_offset

    curr_td_df = trading_day_groups.get_group(td)

    try:
        ib_stats = eval(stats_df.loc[curr_td_df.index[0], 'ib_stats'])[0]
    except:
        # print("error", stats_df.loc[curr_td_df.index[0], 'ib_stats'])
        real_num_days = real_num_days - 1
        continue

    if not ib_stats:
        continue

    curr_atr = stats_df.loc[curr_td_df.index[0], 'ATR']

    rth_range = ib_stats['rth_hi'] - ib_stats['rth_lo']

    if rth_range < (curr_atr * 1):
        bal_tst = bal_tst + 1

    # curr_close = curr_td_df[curr_td_df['Time'] == " 15:59:00.0"]['Close']
    # if curr_close.empty:
    #     curr_close = curr_td_df['Close'].tail(1)
    # curr_close = curr_close.values[0]

    # if ib_stats['ib_hi_ext'] < 0.5 and ib_stats['ib_lo_ext'] < 0.5:
    #     balance_days.append(i)

    # if curr_close > ib_stats['ib_lo'] and curr_close < ib_stats['ib_hi']:
    #     bal_tst = bal_tst + 1

        # print(i, ib_stats)

    # lo_ext.append(ib_stats['ib_lo_ext'])
    # hi_ext.append(ib_stats['ib_hi_ext'])



# print(len(full_balance_days), len(expansion_days), real_num_days)

print(bal_tst)

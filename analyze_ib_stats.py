
import pandas as pd
import numpy as np
import pdb
from scipy import stats, signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

stats_df = pd.read_csv('test_time_stats_full.csv')

stats_df = stats_df.assign(DT='NAN')
stats_df["DT"] = stats_df["Date"] + stats_df["Time"]
stats_df['DT'] = pd.to_datetime(stats_df['DT'])

trading_day_groups = stats_df.groupby(['trading_day'])

real_num_days = len(trading_day_groups)

td_offset = 1
# expansion_days = []
balance_days = []
bal_tst = 0

lo_ext = []
hi_ext = []
pchg = []
other_lo_ext = []
other_hi_ext = []
up_pchg = []
down_pchg = []
ud_pchg = []
up_tst = 0
down_tst = 0
ud_tst = 0
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

    curr_open = curr_td_df[curr_td_df['Time'] == " 09:30:00.0"]['Open']
    curr_close = curr_td_df[curr_td_df['Time'] == " 15:59:00.0"]['Close']

    if curr_open.empty or curr_close.empty:
        continue

    curr_open = curr_open.values[0]
    curr_close = curr_close.values[0]
    curr_pchg = (curr_close - curr_open) / curr_open * 100.0

    time_df = curr_td_df.set_index("DT")
    rth_df = time_df.between_time('9:30', '15:59')

    total_volume = rth_df['Volume'].sum()

    ib_ext = 0.25
    ib_hi_ext = (ib_stats['ib_range'] * ib_ext) + ib_stats['ib_hi']
    ib_lo_ext = ib_stats['ib_lo'] - (ib_stats['ib_range'] * ib_ext)

    ib_price_df = rth_df[(rth_df['Close'] > ib_lo_ext) & (rth_df['Close'] < ib_hi_ext)]

    ib_volume = ib_price_df['Volume'].sum()

    if (ib_volume / total_volume) > 0.9:
        bal_tst = bal_tst + 1
        lo_ext.append(ib_stats['ib_lo_ext'])
        hi_ext.append(ib_stats['ib_hi_ext'])
    else:
        if ib_stats['ib_lo_ext'] < 0.25 and ib_stats['ib_hi_ext'] > 0.25:
            up_tst = up_tst + 1
            up_pchg.append(curr_pchg)

        elif ib_stats['ib_hi_ext'] < 0.25 and ib_stats['ib_lo_ext'] > 0.25:
            down_tst = down_tst + 1
            down_pchg.append(curr_pchg)

        elif ib_stats['ib_hi_ext'] > 0.25 and ib_stats['ib_lo_ext'] > 0.25:
            ud_tst = ud_tst + 1
            ud_pchg.append(curr_pchg)


    # curr_atr = stats_df.loc[curr_td_df.index[0], 'ATR']

    # rth_range = ib_stats['rth_hi'] - ib_stats['rth_lo']

    # if rth_range < (curr_atr * 0.5):
    #     bal_tst = bal_tst + 1
    #     lo_ext.append(ib_stats['ib_lo_ext'])
    #     hi_ext.append(ib_stats['ib_hi_ext'])
    #     pchg.append(curr_pchg)

    # curr_close = curr_td_df[curr_td_df['Time'] == " 15:59:00.0"]['Close']
    # if curr_close.empty:
    #     curr_close = curr_td_df['Close'].tail(1)
    # curr_close = curr_close.values[0]

    # if ib_stats['ib_hi_ext'] < 0.5 and ib_stats['ib_lo_ext'] < 0.5:
    #     bal_tst = bal_tst + 1


    # if curr_close > ib_stats['ib_lo'] and curr_close < ib_stats['ib_hi']:
    #     bal_tst = bal_tst + 1


print(bal_tst, up_tst, down_tst, ud_tst)

# plt.scatter(lo_ext, hi_ext)
# plt.scatter(other_lo_ext, other_hi_ext)
# h = plt.hist2d(other_lo_ext, other_hi_ext, norm=LogNorm())
# plt.colorbar(h[3])
# plt.hist(pchg, bins=100)

# plt.hist(up_pchg, bins=100)
# plt.title('up')

# plt.figure()
plt.hist(ud_pchg, bins=100)


plt.show()

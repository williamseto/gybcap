
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, signal


def test_calc(df):

    print(df)

def calc_atr(prev_td_df, curr_td_df, period, day):

    curr_hi = curr_td_df['High'].max()
    curr_lo = curr_td_df['Low'].min()

    prev_close = prev_td_df[prev_td_df['Time'] == " 15:59:00.0"]['Close']
    if prev_close.empty:
        prev_close = prev_td_df['Close'].tail(1)
    prev_close = prev_close.values[0]

    curr_tr = np.max([curr_hi - curr_lo, np.abs(curr_hi - prev_close), np.abs(curr_lo - prev_close)])

    if day == 0:
        return curr_tr
    
    curr_period = period
    if day < period:
        curr_period = day + 1

    prev_atr = prev_td_df.iloc[0]['ATR']

    curr_atr = (prev_atr * (curr_period - 1) + curr_tr) / curr_period

    return curr_atr

def calc_pchg(prev_td_df, curr_td_df, day):

    if day == 0:
        return 0.0

    prev_close = prev_td_df[prev_td_df['Time'] == " 15:59:00.0"]['Close']
    if prev_close.empty:
        prev_close = prev_td_df['Close'].tail(1)
    prev_close = prev_close.values[0]

    curr_close = curr_td_df[curr_td_df['Time'] == " 15:59:00.0"]['Close']
    if curr_close.empty:
        curr_close = curr_td_df['Close'].tail(1)
    curr_close = curr_close.values[0]

    pchg = (curr_close - prev_close) / prev_close * 100.0

    # print(curr_close, prev_close, pchg)
    return pchg

def calc_ib_stats(curr_td_df):

    ib_stats = {}

    time_df = curr_td_df.set_index("DT")

    rth_df = time_df.between_time('9:30', '15:59')

    if rth_df.empty:
        return ib_stats

    ib_df = time_df.between_time('9:30', '10:30')

    ib_stats['ib_lo'] = ib_df['Low'].min()
    ib_stats['ib_hi'] = ib_df['High'].max()
    ib_stats['ib_range'] = ib_stats['ib_hi'] - ib_stats['ib_lo']

    ib_stats['rth_lo'] = rth_df['Low'].min()
    ib_stats['rth_hi'] = rth_df['High'].max()

    ib_stats['ib_hi_ext'] = (ib_stats['rth_hi'] - ib_stats['ib_hi']) / ib_stats['ib_range']
    ib_stats['ib_lo_ext'] = (ib_stats['ib_lo'] - ib_stats['rth_lo']) / ib_stats['ib_range']

    # curr_td_df.reset_index(level=1, inplace=True)

    return ib_stats


price_df = pd.read_csv('test_time_full.csv')

price_df = price_df.assign(DT='NAN')
price_df["DT"] = price_df["Date"] + price_df["Time"]
price_df['DT'] = pd.to_datetime(price_df['DT'])
# price_df = price_df.set_index("DT", append=True)

price_df = price_df.assign(ATR=np.nan)
price_df = price_df.assign(PCHG=np.nan)
price_df = price_df.assign(ib_stats=np.nan)

trading_day_groups = price_df.groupby(['trading_day'])

atr_period = 14

td_offset = 1
tst = []
for i in range(len(trading_day_groups)):

    td = i + td_offset

    if i == 0:
        prev_td_df = trading_day_groups.get_group(td)
    else:
        prev_td_df = trading_day_groups.get_group(td - 1)

    curr_td_df = trading_day_groups.get_group(td)

    curr_atr = calc_atr(prev_td_df, curr_td_df, atr_period, i)
    price_df.iloc[curr_td_df.index, price_df.columns.get_loc('ATR')] = curr_atr

    curr_pchg = calc_pchg(prev_td_df, curr_td_df, i)
    price_df.iloc[curr_td_df.index, price_df.columns.get_loc('PCHG')] = curr_pchg

    ib_stats = calc_ib_stats(curr_td_df)

    # tst.append(curr_td_df.index[0])
    price_df.loc[curr_td_df.index[0], 'ib_stats'] = [ib_stats]



# print(price_df)

price_df = price_df.drop('DT', axis=1)

price_df.to_csv('test_time_stats_full.csv', index=False, float_format='%.2f')


# stats_df = pd.read_csv('test_time_atr.csv')
# for idx in tst:
#     tst_obj = eval(stats_df.loc[idx, 'ib_stats'])
#     print(tst_obj[0])
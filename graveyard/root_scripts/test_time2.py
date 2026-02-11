
import pandas as pd

from datetime import datetime, timedelta

# price_data = pd.read_csv('es_test_data.csv')
price_data = pd.read_csv('es_historical_public222.txt')

new_price_df = price_data

max_timedelta = timedelta(hours=2)

rth_open_hr = 9
rth_close_hr = 16
ovn_open_hr = 18
ovn_open_min = 0

# dt_format_str = "%m/%d/%Y %H:%M:%S"
dt_format_str = "%Y/%m/%d %H:%M:%S.%f"

def check_ovn(row):
    if row['dt'].hour < rth_open_hr or row['dt'].hour > rth_close_hr or \
       (row['dt'].hour == rth_open_hr and row['dt'].minute < 30):
            return 1
    return 0

new_price_df['dt'] = new_price_df.apply(lambda row: datetime.strptime(f"{row['Date']}{row['Time']}", dt_format_str), axis=1)
new_price_df['ovn'] = new_price_df.apply(check_ovn, axis=1)

new_price_df['dt_delta'] = new_price_df['dt'] - new_price_df['dt'].shift(1)
new_price_df['close_time'] = new_price_df.apply(lambda row: row['dt'].hour == ovn_open_hr and row['dt'].minute == ovn_open_min, axis=1)
new_price_df['trading_day'] = (new_price_df['close_time'] | (new_price_df['dt_delta'] > max_timedelta)).cumsum()


new_price_df.set_index(pd.DatetimeIndex(new_price_df["dt"]), inplace=True)

import pandas_ta as ta
import numpy as np
def compute_intraday_trends(td_df):

    def set_trend_info(trends_list, trend_str, start_idx, end_idx):
        # print(trend_str, td_df["Time"].iloc[start_idx], td_df["Time"].iloc[end_idx], td_df["Low"].iloc[start_idx:end_idx].min(), td_df["High"].iloc[start_idx:end_idx].max())

        trend_val = 0
        if trend_str == "up":
            trend_val = 1
        elif trend_str == "down":
            trend_val = 2

        trends_list[start_idx:end_idx] = [trend_val] * (end_idx - start_idx)

    # compute VWAP and BBands around VWAP to determine threshold

    avg_price = (td_df["Open"] + td_df["High"] + td_df["Low"] + td_df["Close"]).div(4)
    vwap = avg_price.mul(td_df["Volume"]).cumsum().div(td_df["Volume"].cumsum())
    
    volume_sums = td_df["Volume"].cumsum()
    volume_vwap_sums = td_df["Volume"].mul(vwap).cumsum()
    volume_vwap_sqr_sums = td_df["Volume"].mul(vwap.pow(2)).cumsum()

    vwap_diff_sq = ((td_df["High"] + td_df["Low"] + td_df["Close"]).div(3) - volume_vwap_sums.div(volume_sums)).pow(2)
    
    # vwap_std = vwap_diff_sq.mul(td_df["Volume"]).expanding().sum().div(volume_sums).apply(np.sqrt)
    # vwap_std = (volume_vwap_sqr_sums.div(volume_sums) - ((volume_vwap_sums.div(volume_sums)).pow(2))).apply(np.sqrt)
    # vwap_std = vwap.sub(vwap.expanding().apply(lambda x : x.mean())).pow(2).cumsum().div(n_vals).apply(np.sqrt)
    vwap_std = avg_price.sub(vwap).pow(2).cumsum().div(np.arange(1,len(vwap)+1)).apply(np.sqrt)



    trends = [None] * len(td_df)
    previous_trend = "flat"
    current_trend = "flat"
    current_high = td_df["High"].iloc[0]
    current_low = td_df["Low"].iloc[0]
    trend_start_idx = 0

    current_high_idx = 0
    current_low_idx = 0

    flat_period = 30

    for i in range(1, len(td_df)):
        high = td_df["High"].iloc[i]
        low = td_df["Low"].iloc[i]

        threshold = vwap_std.iloc[i]

        if threshold < 2:
            threshold = 2

        high_change = (high - current_low)
        low_change = (low - current_high)

        previous_trend = current_trend

        if current_trend == "up":
            if low_change <= -threshold:
                current_trend = "down"

            if i > flat_period:
                range_low = td_df["Low"].iloc[i-flat_period:i].min()
                range_high = td_df["High"].iloc[i-flat_period:i].max()
                
                if (range_high - range_low) < threshold / 2:
                    current_trend = "flat"

        elif current_trend == "down":
            if high_change >= threshold:
                current_trend = "up"

            if i > flat_period:
                range_low = td_df["Low"].iloc[i-flat_period:i].min()
                range_high = td_df["High"].iloc[i-flat_period:i].max()
                
                if abs(range_low - range_high) < threshold / 2:
                    current_trend = "flat"

        else:  # Initialize trend
            if high_change >= threshold:
                current_trend = "up"
            elif low_change <= -threshold:
                current_trend = "down"

        if high > current_high:
            current_high = high
            current_high_idx = i
        
        if low < current_low:
            current_low = low
            current_low_idx = i

        if previous_trend != current_trend:

            # print(f"updating from {previous_trend} to {current_trend}, {low_change}, {high_change}, {threshold}. {trend_start_idx} {current_low_idx} {current_high_idx} {i}")

            if current_trend == "up" and current_low_idx > trend_start_idx:

                set_trend_info(trends, previous_trend, trend_start_idx, current_low_idx)
                # print(previous_trend, td_df["Time"].iloc[trend_start_idx], td_df["Time"].iloc[current_low_idx], td_df["Low"].iloc[trend_start_idx:current_low_idx].min(), td_df["High"].iloc[trend_start_idx:current_low_idx].max())
                # trends[trend_start_idx:current_low_idx] = [previous_trend] * (current_low_idx - trend_start_idx)
                trend_start_idx = current_low_idx

            if current_trend == "down" and current_high_idx > trend_start_idx:

                set_trend_info(trends, previous_trend, trend_start_idx, current_high_idx)
                # print(previous_trend, td_df["Time"].iloc[trend_start_idx], td_df["Time"].iloc[current_high_idx], td_df["Low"].iloc[trend_start_idx:current_high_idx].min(), td_df["High"].iloc[trend_start_idx:current_high_idx].max())
                # trends[trend_start_idx:current_high_idx] = [previous_trend] * (current_high_idx - trend_start_idx)
                trend_start_idx = current_high_idx

            if current_trend == "flat" and (i - trend_start_idx > flat_period):

                set_trend_info(trends, previous_trend, trend_start_idx, i-flat_period)
                # print(previous_trend, td_df["Time"].iloc[trend_start_idx], td_df["Time"].iloc[i-flat_period], td_df["Low"].iloc[trend_start_idx:i-flat_period].min(), td_df["High"].iloc[trend_start_idx:i-flat_period].max())
                # trends[trend_start_idx:i-flat_period] = [previous_trend] * (i - flat_period - trend_start_idx)
                trend_start_idx = i - flat_period


            if trend_start_idx < i:

                set_trend_info(trends, current_trend, trend_start_idx, i)
                # print(current_trend, td_df["Time"].iloc[trend_start_idx], td_df["Time"].iloc[i], td_df["Low"].iloc[trend_start_idx:i].min(), td_df["High"].iloc[trend_start_idx:i].max())
                # trends[trend_start_idx:i] = [current_trend] * (i - trend_start_idx)

            current_high = high
            current_low = low
            trend_start_idx = i

            previous_trend = current_trend

            current_high_idx = i
            current_low_idx = i

    
    # trends[trend_start_idx:] = [current_trend] * (len(td_df) - trend_start_idx)
    set_trend_info(trends, current_trend, trend_start_idx, len(td_df))

    return pd.Series(trends, index=td_df.index)


new_price_df["intraday_trend"] = new_price_df.groupby('trading_day').apply(compute_intraday_trends, include_groups=False).droplevel(0)

new_price_df.drop(['dt', 'dt_delta', 'close_time'], axis=1, inplace=True)
new_price_df.to_csv('test_time_trend.csv', index=False)

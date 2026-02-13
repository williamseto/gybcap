
import pandas as pd

from datetime import datetime, timedelta

# price_data = pd.read_csv('test2.csv')
price_data = pd.read_csv('es_historical_public222.txt')

new_price_df = price_data

trading_day_idx = 0

first_row = new_price_df.iloc[0]
last_time = datetime.strptime(f"{first_row['Date']}{first_row['Time']}", "%Y/%m/%d %H:%M:%S.%f")
max_timedelta = timedelta(hours=2)

# for index, row in new_price_df.iterrows():

#     curr_time = datetime.strptime(f"{row['Date']}{row['Time']}", "%Y/%m/%d %H:%M:%S.%f")

#     row['ovn'] = 0

#     if curr_time.hour < 9 or \
#        curr_time.hour > 16 or \
#        (curr_time.hour == 9 and curr_time.minute < 30):
#         row['ovn'] = 1

#     if (curr_time.hour == 18 and curr_time.minute == 0) or \
#        ((curr_time - last_time) > max_timedelta):
#         trading_day_idx += 1


#     last_time = curr_time

#     row['trading_day'] = trading_day_idx

#     new_price_df.iloc[index] = row

def check_ovn(row):
    if row['dt'].hour < 9 or row['dt'].hour > 16 or \
       (row['dt'].hour == 9 and row['dt'].minute < 30):
            return 1
    return 0

new_price_df['dt'] = new_price_df.apply(lambda row: datetime.strptime(f"{row['Date']}{row['Time']}", "%Y/%m/%d %H:%M:%S.%f"), axis=1)
new_price_df['ovn'] = new_price_df.apply(check_ovn, axis=1)

new_price_df['dt_delta'] = new_price_df['dt'] - new_price_df['dt'].shift(1)
new_price_df['close_time'] = new_price_df.apply(lambda row: row['dt'].hour == 18 and row['dt'].minute == 0, axis=1)
new_price_df['trading_day'] = (new_price_df['close_time'] | (new_price_df['dt_delta'] > max_timedelta)).cumsum()


new_price_df.set_index(pd.DatetimeIndex(new_price_df["dt"]), inplace=True)

import pandas_ta as ta
import numpy as np
def compute_intraday_trends(td_df):

    # compute VWAP and BBands around VWAP to determine threshold
    vwap = ta.vwap(td_df["High"], td_df["Low"], td_df["Close"], td_df["Volume"])
    
    volume_sums = td_df["Volume"].expanding().sum()
    volume_vwap_sums = td_df["Volume"].mul(vwap).expanding().sum()
    volume_vwap_sqr_sums = td_df["Volume"].mul(vwap.pow(2)).expanding().sum()

    vwap_diff_sq = ((td_df["High"] + td_df["Low"] + td_df["Close"]).div(3) - volume_vwap_sums.div(volume_sums)).pow(2)
    
    vwap_std = vwap_diff_sq.mul(td_df["Volume"]).expanding().sum().div(volume_sums).apply(np.sqrt)

    # vwap_std = (volume_vwap_sqr_sums.div(volume_sums) - (volume_vwap_sums.div(volume_sums).pow(2))).apply(np.sqrt)

    vwap_range = vwap_std.mul(4)
    # vwap_range.to_csv("test_vwap2.csv")

    # exit()

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

        # Calculate percentage changes
        # high_change = (high - current_low) / current_low
        # low_change = (low - current_high) / current_high

        if td_df["ovn"].iloc[i]:
            threshold = vwap_range.iloc[i] / 2
        else:
            threshold = vwap_range.iloc[i] / 4

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

            print(f"updating from {previous_trend} to {current_trend}, {low_change}, {high_change}, {threshold}. {trend_start_idx} {current_low_idx} {current_high_idx} {i}")

            if current_trend == "up" and current_low_idx > trend_start_idx:

                print(previous_trend, td_df["Time"].iloc[trend_start_idx], td_df["Time"].iloc[current_low_idx], td_df["Low"].iloc[trend_start_idx:current_low_idx].min(), td_df["High"].iloc[trend_start_idx:current_low_idx].max())
                trends[trend_start_idx:current_low_idx] = [previous_trend] * (current_low_idx - trend_start_idx)
                trend_start_idx = current_low_idx

            if current_trend == "down" and current_high_idx > trend_start_idx:

                print(previous_trend, td_df["Time"].iloc[trend_start_idx], td_df["Time"].iloc[current_high_idx], td_df["Low"].iloc[trend_start_idx:current_high_idx].min(), td_df["High"].iloc[trend_start_idx:current_high_idx].max())
                trends[trend_start_idx:current_high_idx] = [previous_trend] * (current_high_idx - trend_start_idx)
                trend_start_idx = current_high_idx

            if current_trend == "flat":
                print(previous_trend, td_df["Time"].iloc[trend_start_idx], td_df["Time"].iloc[i-flat_period], td_df["Low"].iloc[trend_start_idx:i-flat_period].min(), td_df["High"].iloc[trend_start_idx:i-flat_period].max())
                trends[trend_start_idx:i-flat_period] = [previous_trend] * (i - flat_period - trend_start_idx)
                trend_start_idx = i - flat_period


            print(current_trend, td_df["Time"].iloc[trend_start_idx], td_df["Time"].iloc[i], td_df["Low"].iloc[trend_start_idx:i].min(), td_df["High"].iloc[trend_start_idx:i].max())
            trends[trend_start_idx:i] = [current_trend] * (i - trend_start_idx)

            current_high = high
            current_low = low
            trend_start_idx = i

            previous_trend = current_trend

            current_high_idx = i
            current_low_idx = i

    
    trends[trend_start_idx:] = [current_trend] * (len(td_df) - trend_start_idx)

    exit()
    return trends


new_price_df.groupby('trading_day').apply(compute_intraday_trends, include_groups=False)

# new_price_df.drop(['dt', 'dt_delta', 'close_time'], axis=1, inplace=True)
# new_price_df.to_csv('test_time_full.csv', index=False)

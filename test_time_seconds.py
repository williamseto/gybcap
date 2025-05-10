
import pandas as pd

from datetime import datetime, timedelta

# price_data = pd.read_csv('es_seconds_20250312_1y.csv')
price_data = pd.read_csv('test_seconds.csv')

max_timedelta = timedelta(minutes=30)

rth_open_hr = 6
rth_close_hr = 13
# ovn_open_hr = 15
# ovn_open_min = 0

dt_format_str = "%m/%d/%Y %H:%M:%S"


def check_ovn(row):

    if row['dt'].hour < rth_open_hr or row['dt'].hour > rth_close_hr or \
        (row['dt'].hour == rth_open_hr and row['dt'].minute < 30) or \
        (row['dt'].hour == rth_close_hr and row['dt'].second > 0):
            return 1
    return 0

price_data['dt'] = price_data.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", dt_format_str), axis=1)
price_data['ovn'] = price_data.apply(check_ovn, axis=1)

price_data['dt_delta'] = price_data['dt'] - price_data['dt'].shift(1)
# new_price_df['close_time'] = new_price_df.apply(lambda row: row['dt'].hour == ovn_open_hr and row['dt'].minute == ovn_open_min, axis=1)
# new_price_df['trading_day'] = (new_price_df['close_time'] | (new_price_df['dt_delta'] > max_timedelta)).cumsum()

price_data['trading_day'] = (price_data['dt_delta'] > max_timedelta).cumsum()

price_data.drop(['Unnamed', 'dt', 'dt_delta'], axis=1, inplace=True)
price_data.to_csv('test_seconds_td0.csv', index=False)
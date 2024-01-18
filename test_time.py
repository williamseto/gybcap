
import pandas as pd

from datetime import datetime, timedelta

# price_data = pd.read_csv('test2.csv')
price_data = pd.read_csv('es_historical_public222.txt')

new_price_df = price_data.assign(ovn='NAN')
new_price_df = new_price_df.assign(trading_day='NAN')

trading_day_idx = 0

first_row = new_price_df.iloc[0]
last_time = datetime.strptime(f"{first_row['Date']}{first_row['Time']}", "%Y/%m/%d %H:%M:%S.%f")
max_timedelta = timedelta(hours=2)

for index, row in new_price_df.iterrows():

    curr_time = datetime.strptime(f"{row['Date']}{row['Time']}", "%Y/%m/%d %H:%M:%S.%f")

    row['ovn'] = 0

    if curr_time.hour < 9 or \
       curr_time.hour > 16 or \
       (curr_time.hour == 9 and curr_time.minute < 30):
        row['ovn'] = 1

    if (curr_time.hour == 18 and curr_time.minute == 0) or \
       ((curr_time - last_time) > max_timedelta):
        trading_day_idx += 1


    last_time = curr_time

    row['trading_day'] = trading_day_idx

    new_price_df.iloc[index] = row

new_price_df.to_csv('test_time_full.csv', index=False)
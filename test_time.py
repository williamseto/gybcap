
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


new_price_df.drop(['dt', 'dt_delta', 'close_time'], axis=1, inplace=True)
new_price_df.to_csv('test_time_full.csv', index=False)

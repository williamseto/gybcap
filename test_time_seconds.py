
import pandas as pd

from datetime import datetime, timedelta
import argparse

def check_ovn(row):

    if row['dt'].hour < rth_open_hr or row['dt'].hour > rth_close_hr or \
        (row['dt'].hour == rth_open_hr and row['dt'].minute < 30) or \
        (row['dt'].hour == rth_close_hr and row['dt'].second > 0):
            return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="add ovn and trading day to pricedata csv"
    )
    parser.add_argument(
        "--input",
        help="input csv",
        required=True
    )
    parser.add_argument(
        "--output",
        help="output csv",
        required=True
    )

    args = parser.parse_args()


    # price_data = pd.read_csv('test_minutes0831.csv')

    price_data = pd.read_csv(args.input)

    price_data = price_data[price_data.columns.drop(list(price_data.filter(regex='Unnamed')))]

    max_timedelta = timedelta(minutes=30)

    rth_open_hr = 6
    rth_close_hr = 14
    dt_format_str = "%m/%d/%Y %H:%M:%S"


    price_data['dt'] = price_data.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", dt_format_str), axis=1)
    price_data['ovn'] = price_data.apply(check_ovn, axis=1)

    price_data['dt_delta'] = price_data['dt'] - price_data['dt'].shift(1)
    # new_price_df['close_time'] = new_price_df.apply(lambda row: row['dt'].hour == ovn_open_hr and row['dt'].minute == ovn_open_min, axis=1)
    # new_price_df['trading_day'] = (new_price_df['close_time'] | (new_price_df['dt_delta'] > max_timedelta)).cumsum()

    price_data['trading_day'] = (price_data['dt_delta'] > max_timedelta).cumsum()

    price_data.drop(['dt', 'dt_delta'], axis=1, inplace=True)
    # price_data.to_csv('test_minutes0831_td0.csv', index=False)

    price_data.to_csv(args.output, index=False)

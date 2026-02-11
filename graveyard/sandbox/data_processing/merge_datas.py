import argparse
import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="combine 2 price data csv"
    )
    parser.add_argument(
        "--old_csv",
        help="csv with older timestamps",
        required=True
    )
    parser.add_argument(
        "--new_csv",
        help="csv with newer timestamps",
        required=True
    )

    parser.add_argument(
        "--out",
        help="output",
        required=True
    )

    args = parser.parse_args()

    old_csv = args.old_csv
    new_csv = args.new_csv

    old_df = pd.read_csv(old_csv)
    new_df = pd.read_csv(new_csv)

    dt_format_str = "%m/%d/%Y %H:%M:%S"

    old_df['dt'] = old_df.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", dt_format_str), axis=1)
    new_df['dt'] = new_df.apply(lambda row: datetime.strptime(f"{row['Date']} {row['Time']}", dt_format_str), axis=1)


    last_old_ts = old_df['dt'].iloc[-1]

    new_df_after = new_df.loc[new_df['dt'] > last_old_ts]

    print(new_df_after.head())

    merged_df = pd.concat([old_df, new_df_after])

    merged_df.drop(['dt'], axis=1, inplace=True)

    merged_df.to_csv(args.out, index=False)

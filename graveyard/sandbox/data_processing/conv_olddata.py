
import pandas as pd

from datetime import datetime, timedelta

# price_data = pd.read_csv('../test.csv')
price_data = pd.read_csv('../es_historical_public222.txt')

new_price_df = price_data



dt_format_str = "%Y/%m/%d %H:%M:%S.%f"


new_price_df['dt'] = new_price_df.apply(lambda row: datetime.strptime(f"{row['Date']}{row['Time']}", dt_format_str), axis=1)

new_price_df["dt"] = new_price_df["dt"] - timedelta(hours=3)


new_price_df['Date'] = new_price_df["dt"].dt.strftime("%-m/%d/%Y")
new_price_df['Time'] = new_price_df["dt"].dt.strftime("%H:%M:%S")



new_price_df.rename(columns={' Bid Volume': 'BidVolume', ' Ask Volume': 'AskVolume'}, inplace=True)


new_price_df.to_csv('../es_historical_public222_conv.csv', columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'BidVolume', 'AskVolume'], index=False)



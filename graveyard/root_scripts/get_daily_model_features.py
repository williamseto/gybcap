
import pandas as pd
import numpy as np
import talib


stats_df = pd.read_csv('test_time_stats_full.csv')

stats_df = stats_df.assign(DT='NAN')
stats_df["DT"] = stats_df["Date"] + stats_df["Time"]
stats_df['DT'] = pd.to_datetime(stats_df['DT'])

def reduce_daily_stats(group):

    daily_high = group['High'].max()
    daily_low = group['Low'].min()
    total_volume = group['Volume'].sum()
    daily_open = group['Open'].values[0]
    daily_close = group['Close'].values[-1]
    daily_range = daily_high - daily_low

    try:
        ib_stats = eval(group['ib_stats'].values[0])

        if isinstance(ib_stats, list):
            ib_stats = ib_stats[0]
    except:
        return None

    if not ib_stats:
        return None

    curr_close = group[group['Time'] == " 15:59:00.0"]['Close']
    if curr_close.empty:
        curr_close = group[group['Time'] == group['Time'].values[-1]]['Close']

    daily_close = curr_close.values[0]

    rth_open = group[group['Time'] == " 09:30:00.0"]['Open']
    if rth_open.empty:
        rth_open = group[group['Time'] == group['Time'].values[0]]['Open']
    
    rth_pchg = (curr_close.item() - rth_open.item()) / rth_open.item() * 100.0

    time_df = group.set_index("DT")
    rth_df = time_df.between_time('9:30', '15:59')

    ib_hi_df = rth_df[rth_df['Close'] > ib_stats['ib_hi']]
    ib_lo_df = rth_df[rth_df['Close'] < ib_stats['ib_lo']]

    if abs(rth_pchg) < 0.5:
        trend_label = 0.0
    elif rth_pchg > 0.5:
        trend_label = 1.0
    else:
        trend_label = 2.0

    return pd.Series({
        'date': group['Date'].values[0],
        'open': daily_open,
        'high': daily_high,
        'low': daily_low,
        'close': daily_close,
        'range': daily_range,
        'volume': total_volume,
        'rth_pchg': rth_pchg,
        'ib_lo_ext': ib_stats['ib_lo_ext'],
        'ib_hi_ext': ib_stats['ib_hi_ext'],
        'ib_lo_vol': ib_lo_df['Volume'].sum() / total_volume,
        'ib_hi_vol': ib_hi_df['Volume'].sum() / total_volume,
        'ib_range_ratio': ib_stats['ib_range'] / (ib_stats['rth_hi'] - ib_stats['rth_lo']),
        'trend': trend_label
    })

daily_stats_df = stats_df.groupby('trading_day').apply(reduce_daily_stats, include_groups=False).dropna().reset_index()


daily_stats_df["return"] = daily_stats_df["close"].pct_change()

daily_stats_df["sma5"] = talib.SMA(daily_stats_df["close"], timeperiod=5)
daily_stats_df["sma20"] = talib.SMA(daily_stats_df["close"], timeperiod=20)

daily_stats_df["rsi"] = talib.RSI(daily_stats_df["close"], timeperiod=14)

daily_stats_df["sma5_rel"] = daily_stats_df["sma5"] / daily_stats_df["close"]
daily_stats_df["sma20_rel"] = daily_stats_df["sma20"] / daily_stats_df["close"]

daily_stats_df["volatility"] = daily_stats_df["return"].rolling(window=20).std()

daily_stats_df["return"] *= 100.0
daily_stats_df["volatility"] *= 100.0

import pandas_ta as ta
daily_stats_df["atr"] = ta.atr(daily_stats_df["high"], daily_stats_df["low"], daily_stats_df["close"])

daily_stats_df.to_csv('test_daily_stats.csv', index=False)

# training_data = daily_stats_df.dropna()
# daily_features = training_data[["return", "sma5_rel", "sma20_rel", "volatility", "rth_pchg", "ib_lo_ext", "ib_hi_ext", "ib_lo_vol", "ib_hi_vol", "ib_range_ratio"]]
# daily_trends = training_data[["trend"]]

# daily_trends.to_csv('test_daily_labels.csv', index=False)

# from sklearn.ensemble import GradientBoostingClassifier
# # prior_model = GradientBoostingClassifier(random_state=42)
# # prior_model.fit(daily_features.to_numpy(), daily_trends.to_numpy().ravel())

# # train_preds = prior_model.predict_proba(daily_features)
# # print(train_preds)

# lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

# for learning_rate in lr_list:
#     prior_model = GradientBoostingClassifier(n_estimators=5, learning_rate=learning_rate, max_depth=2, max_features=5, random_state=42)
#     prior_model.fit(daily_features.to_numpy(), daily_trends.to_numpy().ravel())

#     print("Learning rate: ", learning_rate)
#     print("Accuracy score (training): {0:.3f}".format(prior_model.score(daily_features.to_numpy(), daily_trends.to_numpy().ravel())))
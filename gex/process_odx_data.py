import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm

# ---- Black-Scholes helpers (per-share) ----
def bs_d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_gamma(S, K, r, sigma, T):
    if sigma <= 0 or T <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, r, sigma, T)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_delta(S, K, r, sigma, T, option_type='C'):
    if sigma <= 0 or T <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, r, sigma, T)
    if option_type == 'C':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def compute_gamma_shares(row: pd.Series) -> float:
    r = 0.05
    contract_mult = 100

    iv = row['iv']
    if np.isnan(iv) or iv <= 0:
        return 0.0
    K = float(row['strike'])
    S_level = float(row['underlying_price'])
    oi = float(row['open_interest'] or 0)
    if oi <= 0:
        return 0.0
    T = float(row['years_to_expiration'])
    optype = row['instrument_class']
    gamma = bs_gamma(S_level, K, r, iv, T)
    delta = bs_delta(S_level, K, r, iv, T, option_type=optype)

    gamma_shares = gamma * contract_mult * oi

    return np.abs(gamma_shares)



def process_data(data_path):

    df = pd.read_csv(data_path)

    df.rename(
        columns={
            " [QUOTE_DATE]": "trade_date",
            " [EXPIRE_DATE]": "expiration",
            " [STRIKE]": "strike",
            " [C_IV]": "c_iv",
            " [P_IV]": "p_iv",
        },
        inplace=True
    )

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    df["c_iv"] = pd.to_numeric(df["c_iv"], errors='coerce')
    df["p_iv"] = pd.to_numeric(df["p_iv"], errors='coerce')

    keep_cols = [
        c for c in [
            "trade_date",
            "expiration",
            "strike",
            "c_iv",
            "p_iv",
        ] if c in df.columns
    ]
    df = df[keep_cols]

    shared_cols = [
        "trade_date",
        "expiration",
        "strike",
    ]

    put_cols = [
        "p_iv",
    ]
    call_cols = [
        "c_iv",
    ]

    shared_df = df[shared_cols]

    put_prefix = "p_"
    call_prefix = "c_"

    put_df = df[put_cols]
    put_df.columns = [c[len(put_prefix):] for c in put_df.columns]
    put_df_combined = pd.concat([shared_df, put_df], axis=1)
    put_df_combined["instrument_class"] = "P"

    call_df = df[call_cols]
    call_df.columns = [c[len(call_prefix):] for c in call_df.columns]
    call_df_combined = pd.concat([shared_df, call_df], axis=1)
    call_df_combined["instrument_class"] = "C"

    pd.set_option('display.max_rows', None)
    # example_date = pd.to_datetime('2022-10-31')
    # # example_expiration = pd.to_datetime('2022-10-19')

    # call_df_combined = call_df_combined[(call_df_combined["trade_date"] == example_date)]
    # print(call_df_combined)
    # exit()

    pc_df = pd.concat([put_df_combined, call_df_combined], ignore_index=True).sort_values(by=['expiration', 'strike']).dropna(subset=['iv'])

    # only keep rows where the expiration is after the trade date
    pc_df = pc_df[pc_df["expiration"] > pc_df["trade_date"]]


    spx_data = yf.download("^GSPC", start=pc_df["trade_date"].min(), end=pc_df["trade_date"].max() + pd.Timedelta(days=1))
    spx_data.columns = spx_data.columns.droplevel(1)
    spx_data = spx_data["Close"]


    merged_df = pd.merge(pc_df, spx_data, left_on="trade_date", right_index=True, how="left")
    merged_df.rename(columns={"Close": "underlying_price"}, inplace=True)

    merged_df = merged_df[abs(merged_df["strike"] - merged_df["underlying_price"]) <= merged_df["underlying_price"] * 0.1]

    merged_df["years_to_expiration"] = (merged_df["expiration"] - merged_df["trade_date"]).dt.days / 365
    

    # example_date = pd.to_datetime('2023-01-04')
    # merged_df = merged_df[merged_df["trade_date"] == example_date]

    # merged_df.to_csv('odx_df.csv', index=False)

    return merged_df


def merge_with_oi_data():

    odx_df = pd.read_csv('all_merged_df.csv')

    oi_df = pd.read_csv('oi_table.csv')


    merged_df = pd.merge(odx_df, oi_df, on=['trade_date', 'instrument_class', 'expiration', 'strike'], how='inner')


    return merged_df


if __name__ == "__main__":

    # data_path = '~/Downloads/spx_eod_2023q1-cfph7w/spx_eod_202301.txt'

    data_paths = [
        '~/Downloads/spx_eod_2022q4-dmme3k/spx_eod_202210.txt',
        '~/Downloads/spx_eod_2022q4-dmme3k/spx_eod_202211.txt',
        '~/Downloads/spx_eod_2022q4-dmme3k/spx_eod_202212.txt',
        '~/Downloads/spx_eod_2023q1-cfph7w/spx_eod_202301.txt',
        '~/Downloads/spx_eod_2023q1-cfph7w/spx_eod_202302.txt',
        '~/Downloads/spx_eod_2023q1-cfph7w/spx_eod_202303.txt',
    ]

    all_dfs = []
    for data_path in data_paths:
        df = process_data(data_path)
        all_dfs.append(df)

    all_merged_df = pd.concat(all_dfs)

    all_merged_df.to_csv('all_merged_df.csv', index=False)

    oi_merged_df = merge_with_oi_data()

    # oi_merged_df["gamma_shares"] = oi_merged_df.apply(compute_gamma_shares, axis=1)

    oi_merged_df["trade_date"] = pd.to_datetime(oi_merged_df["trade_date"])
    oi_merged_df["expiration"] = pd.to_datetime(oi_merged_df["expiration"])
    oi_merged_df.sort_values(by=['trade_date'], inplace=True)

    save_df = oi_merged_df[["trade_date", "instrument_class", "expiration", "strike", "iv", "open_interest"]]

    save_df.to_csv('oi_merged_df.csv', index=False)

    save_df.to_parquet('gamma_shares_2023q1.parquet')

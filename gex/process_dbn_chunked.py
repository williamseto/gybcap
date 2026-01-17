
import databento as db
import argparse
from datetime import date, timedelta, datetime, timezone
from dateutil.relativedelta import relativedelta
import glob
import os
import sys
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import yfinance as yf
import math
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


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _norm_cdf(x):
    # stable CDF via erf
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, r, q, sigma, option_type='call'):
    """
    Black-Scholes price for European call or put with continuous dividend yield q.
    S: spot
    K: strike
    T: time to expiry (years)
    r: risk-free rate (annual, continuous)
    q: continuous dividend yield (annual)
    sigma: volatility (annual, stddev)
    option_type: 'call' or 'put'
    """
    if T <= 0 or sigma <= 0:
        # treat as immediate payoff
        if option_type.lower().startswith('c'):
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type.lower().startswith('c'):
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)

def bs_vega(S, K, T, r, q, sigma):
    """Analytic vega (derivative of price w.r.t. sigma)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * math.exp(-q * T) * _norm_pdf(d1) * sqrtT


def extract_opra_symbol_info(opra_symbol):

    if len(opra_symbol) != 21:
        raise ValueError("Invalid OPRA symbol length")

    # Extract components by string slicing
    root_symbol_padded = opra_symbol[0:6]
    expiration_date_str = opra_symbol[6:12]
    option_type = opra_symbol[12]
    strike_price_str_padded = opra_symbol[13:21]

    # Process and format the extracted data
    root_symbol = root_symbol_padded.strip()
    try:
        # Convert YYMMDD string to a datetime object
        expiration_date = datetime.strptime(expiration_date_str, '%y%m%d').date()
    except ValueError:
        expiration_date = "Invalid Date"

    # Convert the padded strike price integer representation to a float/Decimal
    # The last three digits represent the decimal part (price * 1000)
    strike_price_int = int(strike_price_str_padded)
    strike_price = float(strike_price_int) / 1000.0

    return [strike_price, option_type, expiration_date, root_symbol]


def implied_volatility(market_price, S, K, T, r=0.0, q=0.0, option_type='call',
                       initial_guess=0.2, tol=1e-10, maxiter=100):
    """
    Solve for implied volatility sigma such that Black-Scholes price ~= market_price.
    Uses Newton-Raphson with vega, falling back to robust bisection if needed.
    Returns sigma (annualized). Raises ValueError for invalid inputs.
    """
    # basic sanity bounds for European options with continuous div yield
    if market_price < 0:
        raise ValueError("Market price must be non-negative.")
    # intrinsic lower bound (present-value adjusted)
    if option_type.lower().startswith('c'):
        lower_bound = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
        upper_bound = S * math.exp(-q * T)  # trivial upper bound for call
    else:
        lower_bound = max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))
        upper_bound = K * math.exp(-r * T)  # trivial upper bound for put

    eps = 1e-12
    # if market_price < lower_bound - 1e-12 or market_price > upper_bound + 1e-12:
    #     raise ValueError(f"Market price {market_price:.6g} outside no-arbitrage bounds "
    #                      f"[{lower_bound:.6g}, {upper_bound:.6g}].")

    if market_price < lower_bound - eps:
        market_price = lower_bound + eps*2.0

    if market_price > upper_bound + eps:
        market_price = upper_bound - eps*2.0


    # If price is equal to intrinsic (within tol), implied vol -> 0
    if abs(market_price - lower_bound) < tol:
        return 0.0

    # function whose root we want: f(sigma) = model_price(sigma) - market_price
    def f(sigma):
        return bs_price(S, K, T, r, q, sigma, option_type) - market_price

    # Try Newton-Raphson first
    sigma = max(1e-12, initial_guess)
    for i in range(maxiter):
        price = bs_price(S, K, T, r, q, sigma, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        vega = bs_vega(S, K, T, r, q, sigma)
        # if vega is small or NR step would go out of sensible bounds, break to bisection
        if vega < 1e-14:
            break
        step = diff / vega
        sigma_new = sigma - step
        if sigma_new <= 0 or sigma_new > 5.0:  # 5.0 = 500% annual vol cap for robustness
            break
        sigma = sigma_new

    # If NR didn't converge, do a robust bisection/bracketing method
    low = 1e-12
    high = max(0.5, sigma * 2.0)
    # expand high until f(high) > 0 (model_price > market_price) — pricing monotone increasing in sigma
    # we want f(low) <= 0 <= f(high) (or vice-versa depending on sign), so that a root is bracketed
    f_low = f(low)
    f_high = f(high)
    it = 0
    while f_high < 0 and high < 10.0 and it < 60:  # cap high at 1000% vol for safety
        high *= 2.0
        f_high = f(high)
        it += 1
    if f_low * f_high > 0:
        # As a last resort, try swapping low/high (shouldn't normally be needed)
        # or return the best guess we have
        # Return sigma (NR result or initial guess) as best-effort
        return max(sigma, initial_guess)

    for i in range(200):
        mid = 0.5 * (low + high)
        f_mid = f(mid)
        if abs(f_mid) < tol or (high - low) < 1e-12:
            return mid
        # decide which side to keep
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    # final fallback
    return 0.5 * (low + high)


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


def process_dbn_chunked(args):

    in_dir = args.directory

    # files = sorted(glob.glob(os.path.join(in_dir, "*.dbn")) + glob.glob(os.path.join(in_dir, "*.dbn.zst")))
    # if not files:
    #     print("No .dbn or .dbn.zst files found in", in_dir)
    #     sys.exit(1)

    # print(f"Loading {files[0]} ...")
    # store = db.DBNStore.from_file(files[0])

    # store.to_parquet('test.parquet')
    # exit()


    dataset = ds.dataset('spx_stats_combined.parquet', format="parquet")

    # today = date.today()
    # start_date = today - relativedelta(years=3, months=2)

    # start_date = datetime.strptime("10/01/2022", "%m/%d/%Y").date()
    # today = datetime.strptime("03/31/2023", "%m/%d/%Y").date()

    start_date = datetime.strptime("04/01/2023", "%m/%d/%Y").date()
    today = datetime.strptime("10/31/2025", "%m/%d/%Y").date()

    print(f"Filtering records from {start_date} through {today}")


    spx_data = yf.download("^GSPC", start=start_date, end=today)
    spx_data.columns = spx_data.columns.droplevel(1)
    spx_data = spx_data["Close"]

    date_col = "ts_event"
    idx = 0

    oi_day_tables = []
    day_tables = []
    current_date = start_date
    while current_date < today:
        next_date = current_date + timedelta(days=1)
        print(f"--- Processing data for date: {current_date.isoformat()} ---")


        expr = (ds.field(date_col) >= pa.scalar(current_date)) & (ds.field(date_col) < pa.scalar(next_date))

        scanner = dataset.scanner(filter=expr)

        df = scanner.to_table().to_pandas()

        current_date = next_date

        if len(df) == 0:
            continue


        df['trade_date'] = df.ts_event.dt.date

        stat_type_mask = (df["stat_type"] == db.StatType.CLEARED_VOLUME) | \
            (df["stat_type"] == db.StatType.OPEN_INTEREST) | \
            (df["stat_type"] == db.StatType.CLOSE_PRICE) | \
            (df["stat_type"] == db.StatType.HIGHEST_BID) | \
            (df["stat_type"] == db.StatType.LOWEST_OFFER)
        stats_df = df[stat_type_mask].copy()

        def stat_type_to_name(stat_type):
            if stat_type == db.StatType.CLEARED_VOLUME:
                return "cleared_volume"
            elif stat_type == db.StatType.OPEN_INTEREST:
                return "open_interest"
            elif stat_type == db.StatType.CLOSE_PRICE:
                return "close_price"
            elif stat_type == db.StatType.HIGHEST_BID:
                return "highest_bid"
            elif stat_type == db.StatType.LOWEST_OFFER:
                return "lowest_offer"

        stats_df["stat_name"] = stats_df["stat_type"].apply(stat_type_to_name)

        cv_mask = stats_df["stat_type"] == db.StatType.CLEARED_VOLUME
        oi_mask = stats_df["stat_type"] == db.StatType.OPEN_INTEREST
        cp_mask = stats_df["stat_type"] == db.StatType.CLOSE_PRICE
        hb_mask = stats_df["stat_type"] == db.StatType.HIGHEST_BID
        lo_mask = stats_df["stat_type"] == db.StatType.LOWEST_OFFER


        stats_df.loc[cv_mask, "stat_value"] = stats_df[cv_mask]["quantity"]
        stats_df.loc[oi_mask, "stat_value"] = stats_df[oi_mask]["quantity"]
        stats_df.loc[cp_mask, "stat_value"] = stats_df[cp_mask]["price"]
        stats_df.loc[hb_mask, "stat_value"] = stats_df[hb_mask]["price"]
        stats_df.loc[lo_mask, "stat_value"] = stats_df[lo_mask]["price"]


        # remove rows where open_interest is 0
        stats_df = stats_df[~((stats_df["stat_name"] == "open_interest") & (stats_df["stat_value"] == 0))]

        # Groupby and keep the last published record for each statistic
        stats_df = stats_df.groupby(["trade_date", "symbol", "stat_type"], as_index=False).agg('last').sort_values(["trade_date", "instrument_id"])
        
        keep_cols = [
            c for c in [
                "trade_date",
                "symbol",
                "instrument_id",
                "stat_type",
                "open_interest",
                "cleared_volume",
                "close_price",
                "ts_event_ts",
                "stat_value",
                "stat_name",
                "highest_bid",
                "lowest_offer",
            ] if c in stats_df.columns
        ]
        stats_df = stats_df[keep_cols]


        pivot_df = stats_df.pivot_table(index=["trade_date", "symbol"], columns="stat_name", values="stat_value", aggfunc="last").reset_index()

        pivot_df["open_interest"] = pivot_df["open_interest"].fillna(0)
   
        pivot_df[["strike", "instrument_class", "expiration", "root_symbol"]] = pivot_df.apply(lambda x: extract_opra_symbol_info(x['symbol']), axis=1, result_type='expand')

        # only keep rows where the expiration is after the trade date
        pivot_df = pivot_df[pivot_df["expiration"] > pivot_df["trade_date"]]


        # oi_df = pivot_df[["trade_date", "instrument_class", "expiration", "strike", "open_interest"]]

        # # agg for SPX and SPXW
        # oi_df = oi_df.groupby(["trade_date", "instrument_class", "expiration", "strike"], as_index=False).agg({'open_interest':'sum'})

        # all_duplicates = oi_df[oi_df.duplicated(subset=['trade_date', 'instrument_class', 'expiration', 'strike'], keep=False)]

        # if not all_duplicates.empty:
        #     all_duplicates.to_csv('all_duplicates.csv', index=False)

        #     agg_dict = {
        #         'open_interest':'sum'
        #     }
        #     oi_df_updated = oi_df.groupby(["trade_date", "instrument_class", "expiration", "strike"], as_index=False).agg(agg_dict)
        #     oi_df_updated.to_csv('oi_df_updated.csv', index=False)

        #     exit()

        # oi_day_tables.append(pa.Table.from_pandas(oi_df, preserve_index=False))

        # if len(oi_day_tables) == 1:
        #     oi_table = oi_day_tables[0]
        # else:
        #     oi_table = pa.concat_tables(oi_day_tables, promote_options='default')

        # # Write to parquet
        # pq.write_table(oi_table, "oi_table.parquet")
        # continue

        pivot_df["trade_date"] = pd.to_datetime(pivot_df["trade_date"])
        pivot_df["expiration"] = pd.to_datetime(pivot_df["expiration"])

        pivot_df["years_to_expiration"] = (pivot_df["expiration"] - pivot_df["trade_date"]).dt.days / 365

        pivot_df['midprice'] = (pivot_df['highest_bid'] + pivot_df['lowest_offer']) / 2 

        merged_df = pd.merge(pivot_df, spx_data, left_on="trade_date", right_index=True, how="left")
        merged_df.rename(columns={"Close": "underlying_price"}, inplace=True)
        merged_df = merged_df[abs(merged_df["strike"] - merged_df["underlying_price"]) <= merged_df["underlying_price"] * 0.1]


        merged_df["iv"] = merged_df.apply(lambda row: implied_volatility(row["midprice"], row["underlying_price"], row["strike"], row["years_to_expiration"], r=0.04, q=0.013, option_type=row["instrument_class"]), axis=1)

        # merged_df["gamma_shares"] = merged_df.apply(compute_gamma_shares, axis=1)

        
        save_df = merged_df[["trade_date", "instrument_class", "expiration", "strike", "root_symbol", "iv", "open_interest"]]


        # agg for SPX and SPXW
        save_df = save_df.groupby(["trade_date", "instrument_class", "expiration", "strike"], as_index=False).agg({'open_interest':'sum', 'iv':'mean'})

        save_df = save_df[["trade_date", "instrument_class", "expiration", "strike", "iv", "open_interest"]]

        day_tables.append(pa.Table.from_pandas(save_df, preserve_index=False))

        if len(day_tables) == 1:
            out_table = day_tables[0]
        else:
            out_table = pa.concat_tables(day_tables, promote_options='default')

        # Write to parquet
        pq.write_table(out_table, "gamma_shares_2025.parquet")



def inspect_parquet(path):
    table = pq.read_table(path)
    df = table.to_pandas()
    df.to_csv('output.csv', index=False)
    exit()


def dbn_to_parquet(path):
    files = sorted(glob.glob(os.path.join(path, "*.dbn")) + glob.glob(os.path.join(path, "*.dbn.zst")))
    if not files:
        raise FileNotFoundError("No DBN files found in the input directory.")

    parquet_dir = os.path.join(os.getcwd(), "parqs")

    start_date = datetime.strptime("10/01/2022", "%m/%d/%Y").date()
    end_date = datetime.strptime("03/31/2023", "%m/%d/%Y").date()

    current_month_date = start_date
    while current_month_date < end_date:
        break
        end_of_month_date = current_month_date + relativedelta(months=1) - timedelta(days=1)

        month_filename = f"opra-pillar-{current_month_date.strftime('%Y%m%d')}-{end_of_month_date.strftime('%Y%m%d')}.statistics.dbn.zst"
        print(f"Processing {month_filename} ...")

        file_path = os.path.join(path, month_filename)

        if not os.path.exists(file_path):
            print(f"File {month_filename} does not exist")
            continue

        store = db.DBNStore.from_file(file_path)

        store.to_parquet(os.path.join(parquet_dir, month_filename.replace(".statistics.dbn.zst", ".parquet")))

        current_month_date = current_month_date + relativedelta(months=1)


    all_parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))

    schema = pq.ParquetFile(all_parquet_files[0]).schema_arrow

    with pq.ParquetWriter("spxw_stats.parquet", schema=schema) as writer:
        for f in all_parquet_files:
            writer.write_table(pq.read_table(f, schema=schema))


def combine_monthly_weekly_parquets():
    input_files = ["test.parquet", "spxw_stats.parquet"]
    output_path = "spx_stats_combined.parquet"

    schema = pq.read_schema(input_files[0])

    with pq.ParquetWriter(output_path, schema=schema) as writer:
        for file in input_files:
            writer.write_table(pq.read_table(file, schema=schema))


def merge_gamma_shares_parquets():
    input_files = ["gamma_shares_2023q1.parquet", "gamma_shares_2025.parquet"]
    output_path = "gamma_shares_combined.parquet"

    schema = pq.read_schema(input_files[0])

    with pq.ParquetWriter(output_path, schema=schema) as writer:
        for file in input_files:
            writer.write_table(pq.read_table(file, schema=schema))

def merge_gamma_stats_with_es_data():
    gamma_shares_path = "gamma_shares_combined.parquet"
    es_data_path = "../raw_data/es_min_3y_clean_td.csv"


    gamma_df = pd.read_parquet(gamma_shares_path)
    gamma_df['trade_date'] = pd.to_datetime(gamma_df['trade_date'])


    es_data_df = pd.read_csv(es_data_path)
    es_data_df['Date'] = pd.to_datetime(es_data_df['Date'])


    spx_data = yf.download("^GSPC", start=gamma_df.iloc[0]['trade_date'].date(), end=gamma_df.iloc[-1]['trade_date'].date())
    spx_data.columns = spx_data.columns.droplevel(1)
    spx_data = spx_data["Open"]


    spx_price_adj = 0

    def compute_gamma_stats(group):
        trade_date = group.iloc[-1]['Date']

        prev_trade_date = trade_date - pd.tseries.offsets.BDay(1)

        print("processing trade date: ", prev_trade_date.strftime('%Y-%m-%d'))

        gamma_day_df = gamma_df[gamma_df["trade_date"] == prev_trade_date]

        global spx_price_adj
        try:
            spx_open_price = spx_data.loc[trade_date.strftime('%Y-%m-%d')]
            es_open_price = group[group["Time"] == "06:30:00"]["Open"].values[0]

            spx_price_adj = spx_open_price - es_open_price
        except Exception as e:
            pass


        strikes = gamma_day_df["strike"].to_numpy()

        es_closes_adj = group["Close"].to_numpy() + spx_price_adj

        strike_price_diffs = np.abs(strikes[:, np.newaxis] - es_closes_adj)

        gamma_shares = gamma_day_df["gamma_shares"].to_numpy()

        nearby_gamma_shares = np.sum(np.where(strike_price_diffs <= 5.0, gamma_shares[:, np.newaxis], 0.0), axis=0)

        relative_gamma_shares = np.sum(np.where(strike_price_diffs <= 15.0, gamma_shares[:, np.newaxis], 0.0), axis=0)

        nearby_gamma_score = nearby_gamma_shares / relative_gamma_shares
        nearby_gamma_score = np.nan_to_num(nearby_gamma_score, nan=0.0, posinf=0.0, neginf=0.0)
        
        group["nearby_gamma_score"] = nearby_gamma_score

        # pd.set_option('display.max_rows', None)
        # print(group)
        # exit()

        return group

    es_data_df = es_data_df.groupby('trading_day').apply(compute_gamma_stats, include_groups=False).reset_index()
    es_data_df.drop('level_1', axis=1, inplace=True)

    es_data_df["Date"] = es_data_df["Date"].dt.strftime('%m/%d/%Y')


    es_data_df.to_csv("es_min_3y_clean_td_gamma.csv", index=False)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", help="Path to folder containing DBN (.dbn / .dbn.zst) file(s)")

    args = p.parse_args()
    process_dbn_chunked(args)

    # dbn_to_parquet(args.directory)



    inspect_parquet("gamma_shares_2025.parquet")

    # inspect_parquet("oi_table.parquet")

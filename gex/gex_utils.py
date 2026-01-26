
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.stats import norm
import yfinance as yf
import warnings


def bs_gamma_vec(S_vec, K_vec, r, iv_mat, T_mat):
    S_mat = S_vec.to_numpy()[:, None]
    K_mat = K_vec.to_numpy()[None, :]

    # ensure minimum value for iv_mat
    iv_mat = np.maximum(iv_mat, 0.00001)

    sqrtT = np.sqrt(T_mat)

    d1 = (
        np.log(S_mat / K_mat)
        + (r + 0.5 * iv_mat**2) * T_mat
    ) / (iv_mat * sqrtT)

    gamma = norm.pdf(d1) / (S_mat * iv_mat * sqrtT)
    return gamma


class IVMapper:
    def __init__(self):

        gamma_shares_path = "../gex/gamma_shares_combined.parquet"
        self.gamma_df = pd.read_parquet(gamma_shares_path).reset_index(drop=True)
        self.gamma_df['trade_date'] = pd.to_datetime(self.gamma_df['trade_date'])
        self.gamma_df['expiration'] = pd.to_datetime(self.gamma_df['expiration'])


        # example_date = pd.to_datetime('2022-10-04')
        # prev_trade_date = example_date - pd.tseries.offsets.BDay(1)

        # chain_df = self.gamma_df[self.gamma_df["trade_date"] == prev_trade_date]


        spx_data = yf.download("^GSPC", start=self.gamma_df.iloc[0]['trade_date'].date(), end=self.gamma_df.iloc[-1]['trade_date'].date())
        spx_data.columns = spx_data.columns.droplevel(1)
        self.spx_data = spx_data[["Open","Close"]]

        self.spx_price_adj = 0
    
        return




    def compute_gamma_stats(self, es_df):

        if es_df.empty:
            return 0

        trade_date = es_df.index[-1].date()

        # trade_date = pd.to_datetime('2025-04-09').date()


        # prev_trade_date = trade_date - pd.tseries.offsets.BDay(1)
        # chain_df = self.gamma_df[self.gamma_df["trade_date"] == prev_trade_date]


        # try to go backwards up to 3 days
        for i in range(1,4):
            prev_trade_date = trade_date - pd.tseries.offsets.BDay(i)
            chain_df = self.gamma_df[self.gamma_df["trade_date"] == prev_trade_date]
            if not chain_df.empty:
                break
        
        print(f"processing trade date: {trade_date.strftime('%Y-%m-%d')} with prev trade date {prev_trade_date.strftime('%Y-%m-%d')}")


        # remove rows where the expiration is before the trade date
        chain_df = chain_df[chain_df["expiration"].dt.date > trade_date]

        chain_df = chain_df.sort_values(by=['expiration'])


        def compute_mapped_ivs(expiry_df):


            try:
                spx_open_price = self.spx_data.loc[trade_date.strftime('%Y-%m-%d')]["Open"]
                es_open_price = es_df[es_df["time"] == "06:30:00"]["open"].values[0]

                self.spx_price_adj = spx_open_price - es_open_price
            except Exception as e:
                pass

            # print(es_df)
            # print(spx_open_price, es_open_price)
            # exit()


            S_new = es_df["close"].to_numpy() + self.spx_price_adj

            S_old = self.spx_data.loc[prev_trade_date.strftime('%Y-%m-%d')]["Close"]


            moneyness_shift = np.log(S_new / S_old)

            put_mask = expiry_df["instrument_class"] == "P"
            call_mask = expiry_df["instrument_class"] == "C"

            puts = expiry_df[put_mask]
            calls = expiry_df[call_mask]


            m0_puts = np.log(puts["strike"] / S_old)
            m0_calls = np.log(calls["strike"] / S_old)

            m_new_puts = m0_puts.to_numpy()[:, None] - moneyness_shift[None, :]
            m_new_calls = m0_calls.to_numpy()[:, None] - moneyness_shift[None, :]

            m_old_calls = np.log(calls["strike"] / S_old)
            m_old_puts = np.log(puts["strike"] / S_old)

            call_interp = interp1d(m_old_calls, calls["iv"], kind='linear', fill_value='extrapolate', bounds_error=False)
            put_interp = interp1d(m_old_puts, puts["iv"], kind='linear', fill_value='extrapolate', bounds_error=False)

            new_call_ivs = call_interp(m_new_calls)
            new_put_ivs = put_interp(m_new_puts)


            # pd.set_option('display.max_rows', None)
            # puts["new_iv"] = new_put_ivs[:, -1]
            # calls["new_iv"] = new_call_ivs[:, -1]


            mapped_ivs = np.zeros((len(expiry_df), len(es_df)))

            mapped_ivs[put_mask, :] = new_put_ivs
            mapped_ivs[call_mask, :] = new_call_ivs


            return mapped_ivs


        T_vec = es_df.index.values.astype('datetime64[s]')

        expiry_dt = chain_df['expiration'].values.astype('datetime64[s]') + pd.Timedelta(hours=13)


        tte = (expiry_dt - T_vec[:, None]).astype('timedelta64[s]').astype(float) / (365.0 * 24 * 60 * 60)


        mapped_ivs = chain_df.groupby('expiration').apply(compute_mapped_ivs, include_groups=False)


        mapped_ivs = np.concatenate(mapped_ivs.values)

        gamma_mat = bs_gamma_vec(es_df["close"], chain_df['strike'], 0.04, mapped_ivs.T, tte)

        oi_vec = chain_df['open_interest'].to_numpy()
        gamma_shares_mat = gamma_mat * oi_vec[None, :] * 100

        S_mat = es_df["close"].to_numpy()[:, None]

        strike_price_diffs = np.abs(S_mat - chain_df['strike'].to_numpy())

        local_gex = 0.01 * np.sum(np.where(strike_price_diffs <= 5.0, gamma_shares_mat, 0.0) * S_mat**2, axis=1)
        relative_gex = 0.01 * np.sum(np.where(strike_price_diffs <= 15.0, gamma_shares_mat, 0.0) * S_mat**2, axis=1)

        nearby_gamma_score = local_gex / relative_gex
        nearby_gamma_score = np.nan_to_num(nearby_gamma_score, nan=0.0, posinf=0.0, neginf=0.0)


        return nearby_gamma_score


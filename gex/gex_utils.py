
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.stats import norm
import yfinance as yf
import warnings

import sys


def bs_gamma_vec(S_vec, K_vec, r, iv_mat, T_mat):
    """
    Vectorized Black-Scholes gamma computation.

    Args:
        S_vec: Series/array of spot prices (shape: n_spots)
        K_vec: Series/array of strikes (shape: n_strikes)
        r: Risk-free rate
        iv_mat: IV matrix (shape: n_spots x n_strikes) or (n_strikes,) for single spot
        T_mat: Time-to-expiry matrix (shape: n_spots x n_strikes) or (n_strikes,) for single spot

    Returns:
        Gamma matrix (shape: n_spots x n_strikes)
    """
    if hasattr(S_vec, 'to_numpy'):
        S_mat = S_vec.to_numpy()[:, None]
    else:
        S_mat = np.atleast_1d(S_vec)[:, None]

    if hasattr(K_vec, 'to_numpy'):
        K_mat = K_vec.to_numpy()[None, :]
    else:
        K_mat = np.atleast_1d(K_vec)[None, :]

    # ensure minimum value for iv_mat
    iv_mat = np.maximum(iv_mat, 0.00001)

    sqrtT = np.sqrt(T_mat)

    d1 = (
        np.log(S_mat / K_mat)
        + (r + 0.5 * iv_mat**2) * T_mat
    ) / (iv_mat * sqrtT)

    gamma = norm.pdf(d1) / (S_mat * iv_mat * sqrtT)
    return gamma



class IVSurfaceMapper:
    """Maps IV surface to new spot prices using moneyness interpolation."""

    def __init__(self, strikes: np.ndarray, ivs: np.ndarray, option_types: np.ndarray, reference_spot: float):
        """
        Args:
            strikes: Array of strike prices
            ivs: Array of implied volatilities
            option_types: Array of 'C'/'P' for each strike
            reference_spot: Spot price when IVs were observed (S_old)
        """
        self.strikes = np.asarray(strikes)
        self.ivs = np.asarray(ivs)
        self.option_types = np.asarray(option_types)
        self.reference_spot = reference_spot

        # Separate put/call data
        self.put_mask = self.option_types == 'P'
        self.call_mask = self.option_types == 'C'

        # Compute moneyness for puts and calls
        put_strikes = self.strikes[self.put_mask]
        call_strikes = self.strikes[self.call_mask]
        put_ivs = self.ivs[self.put_mask]
        call_ivs = self.ivs[self.call_mask]

        self.m_old_puts = np.log(put_strikes / reference_spot) if len(put_strikes) > 0 else np.array([])
        self.m_old_calls = np.log(call_strikes / reference_spot) if len(call_strikes) > 0 else np.array([])

        # Create interpolators
        if len(self.m_old_puts) > 1:
            self.put_interp = interp1d(self.m_old_puts, put_ivs, kind='linear', fill_value='extrapolate', bounds_error=False)
        else:
            self.put_interp = None

        if len(self.m_old_calls) > 1:
            self.call_interp = interp1d(self.m_old_calls, call_ivs, kind='linear', fill_value='extrapolate', bounds_error=False)
        else:
            self.call_interp = None

    def map_ivs(self, new_spots: np.ndarray) -> np.ndarray:
        """
        Map IVs to multiple new spot prices.

        Args:
            new_spots: Array of new spot prices (n_spots,)

        Returns:
            IV matrix (n_strikes x n_spots)
        """
        new_spots = np.atleast_1d(new_spots)
        n_strikes = len(self.strikes)
        n_spots = len(new_spots)

        # Compute moneyness shift
        moneyness_shift = np.log(new_spots / self.reference_spot)

        # Initialize output
        mapped_ivs = np.zeros((n_strikes, n_spots))

        # Map put IVs
        if self.put_interp is not None:
            m_new_puts = self.m_old_puts[:, None] - moneyness_shift[None, :]
            new_put_ivs = self.put_interp(m_new_puts)
            mapped_ivs[self.put_mask, :] = new_put_ivs


        # Map call IVs
        if self.call_interp is not None:
            m_new_calls = self.m_old_calls[:, None] - moneyness_shift[None, :]
            new_call_ivs = self.call_interp(m_new_calls)
            mapped_ivs[self.call_mask, :] = new_call_ivs

        # ensure max/min IV bounds
        mapped_ivs = np.clip(mapped_ivs, 0.00001, 10.0)

        return mapped_ivs

    def map_ivs_single(self, new_spot: float) -> np.ndarray:
        """
        Map IVs to a single new spot price.

        Args:
            new_spot: New spot price

        Returns:
            IV array (n_strikes,)
        """
        return self.map_ivs(np.array([new_spot]))[:, 0]


def compute_batch_gamma_scores(
    spot_prices: np.ndarray,
    strikes: np.ndarray,
    iv_matrix: np.ndarray,
    ois: np.ndarray,
    tte_matrix: np.ndarray,
    local_radius: float = 5.0,
    relative_radius: float = 15.0,
    r: float = 0.04
) -> np.ndarray:
    """
    Vectorized batch GEX score computation.

    Args:
        spot_prices: Array of spot prices (n_spots,)
        strikes: Array of strike prices (n_strikes,)
        iv_matrix: IV matrix (n_spots x n_strikes)
        ois: Array of open interest values (n_strikes,)
        tte_matrix: Time-to-expiry matrix (n_spots x n_strikes)
        local_radius: Radius for local GEX calculation (default 5.0)
        relative_radius: Radius for relative GEX calculation (default 15.0)
        r: Risk-free rate (default 0.04)

    Returns:
        GEX scores array (n_spots,)
    """
    spot_prices = np.atleast_1d(spot_prices)
    strikes = np.atleast_1d(strikes)
    ois = np.atleast_1d(ois)

    n_spots = len(spot_prices)

    if len(strikes) == 0 or n_spots == 0:
        return np.zeros(n_spots)

    # Compute gamma matrix using vectorized BS gamma
    gamma_mat = bs_gamma_vec(spot_prices, strikes, r, iv_matrix, tte_matrix)

    # Gamma shares = gamma * OI * 100 (contract multiplier)
    oi_vec = ois[None, :]
    gamma_shares_mat = gamma_mat * oi_vec * 100

    # Spot prices as column vector
    S_mat = spot_prices[:, None]
    S_squared = S_mat ** 2

    # Strike price differences
    strike_price_diffs = np.abs(S_mat - strikes[None, :])

    # Compute local and relative GEX
    local_gex = 0.01 * np.sum(np.where(strike_price_diffs <= local_radius, gamma_shares_mat, 0.0) * S_squared, axis=1)
    relative_gex = 0.01 * np.sum(np.where(strike_price_diffs <= relative_radius, gamma_shares_mat, 0.0) * S_squared, axis=1)

    # Compute score with safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = local_gex / relative_gex

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.clip(scores, 0.0, 1.0)

    return scores


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

        # try to go backwards up to 3 days
        for i in range(1, 4):
            prev_trade_date = trade_date - pd.tseries.offsets.BDay(i)
            chain_df = self.gamma_df[self.gamma_df["trade_date"] == prev_trade_date]
            if not chain_df.empty:
                break

        print(f"processing trade date: {trade_date.strftime('%Y-%m-%d')} with prev trade date {prev_trade_date.strftime('%Y-%m-%d')}")

        # remove rows where the expiration is before the trade date
        chain_df = chain_df[chain_df["expiration"].dt.date > trade_date]
        chain_df = chain_df.sort_values(by=['expiration'])

        # Calibrate SPX-ES price adjustment
        try:
            spx_open_price = self.spx_data.loc[trade_date.strftime('%Y-%m-%d')]["Open"]
            es_open_price = es_df[es_df["time"] == "06:30:00"]["open"].values[0]
            self.spx_price_adj = spx_open_price - es_open_price
        except Exception:
            pass

        # Compute new spot prices (SPX equivalent)
        S_new = es_df["close"].to_numpy() + self.spx_price_adj
        S_old = self.spx_data.loc[prev_trade_date.strftime('%Y-%m-%d')]["Close"]

        # Create mapper per expiry and get mapped IVs
        def get_mapped_ivs_for_expiry(expiry_df):
            mapper = IVSurfaceMapper(
                strikes=expiry_df['strike'].to_numpy(),
                ivs=expiry_df['iv'].to_numpy(),
                option_types=expiry_df['instrument_class'].to_numpy(),
                reference_spot=S_old
            )
            return mapper.map_ivs(S_new)

        mapped_ivs = chain_df.groupby('expiration').apply(get_mapped_ivs_for_expiry, include_groups=False)
        mapped_ivs = np.concatenate(mapped_ivs.values)

        # Compute time-to-expiry matrix
        T_vec = es_df.index.values.astype('datetime64[s]')
        expiry_dt = chain_df['expiration'].values.astype('datetime64[s]') + pd.Timedelta(hours=13)
        tte = (expiry_dt - T_vec[:, None]).astype('timedelta64[s]').astype(float) / (365.0 * 24 * 60 * 60)

        # Use shared batch computation
        nearby_gamma_score = compute_batch_gamma_scores(
            spot_prices=S_new,
            strikes=chain_df['strike'].to_numpy(),
            iv_matrix=mapped_ivs.T,
            ois=chain_df['open_interest'].to_numpy(),
            tte_matrix=tte
        )

        return nearby_gamma_score


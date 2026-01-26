
import time
import threading
import numpy as np
import pandas as pd
from typing import Optional, Tuple

import sys

def get_spx_oc():
    import yfinance as yf

    def get_dte_years(expiry, as_of_date):
        expiry_date = pd.to_datetime(expiry).date()
        as_of_date = pd.to_datetime(as_of_date).date()
        days = max((expiry_date - as_of_date).days, 0)
        return days/365.0

    print("Fetching SPX option chain...")

    ticker = '^SPX'
    min_oi = 1
    tk = yf.Ticker(ticker)

    spot_hist = tk.history(period="1d")  # just to get recent spot
    if spot_hist.empty:
        raise RuntimeError("No spot data")
    spot = float(spot_hist['Open'].iloc[-1])
    as_of = spot_hist.index[-1].date()


    # Fetch chains for all expiries
    expiries = tk.options
    if not expiries:
        print("No option expiries")
        return None, None


    # Collect all options into one DataFrame
    rows = []
    for exp in expiries:
        try:
            oc = tk.option_chain(exp)
        except Exception:
            continue

        calls = oc.calls.copy(); puts = oc.puts.copy()
        calls = calls.assign(contractType='call', expiry=exp)
        puts = puts.assign(contractType='put', expiry=exp)
        chain = pd.concat([calls, puts], ignore_index=True, sort=False)
        chain = chain.rename(columns={'impliedVolatility':'iv','openInterest':'oi','strike':'strike'})
        chain = chain[['strike','iv','oi','contractType','expiry', 'volume']]

        chain = chain.dropna(subset=['strike'])

        # chain['oi'] = chain['volume']

        chain.fillna({'oi':0}, inplace=True)

        # filter OI
        # chain = chain[chain['oi'].fillna(0) >= min_oi]

        rows.append(chain)


        # if len(rows) > 3:
        #     break
    if not rows:
        print("No usable options found")
        return None, None
    
    chain = pd.concat(rows, ignore_index=True, sort=False)

    # chain['dte_years'] = chain['expiry'].map(lambda e: get_dte_years(e, as_of))

    print("Fetched SPX option chain.")

    return chain, spot


class RealtimeGEXProvider:
    """
    Computes GEX scores from live options chain data.

    Uses simplified computation with direct IV (no moneyness interpolation),
    or IV surface mapping for batch computation with price changes.
    """

    def __init__(self):
        """
        Args:
            cache: OptionsChainCache instance (creates one if not provided)
            spx_es_spread: Default SPX - ES spread in points (calibratable)
        """
        self.spx_es_spread = 0
        self.chain_df = None
        self.spot_price = None
        self._iv_mapper = None
        self._last_refresh_ts = None


    def fetch_chain(self):

        try:
            chain_df, spot = get_spx_oc()
            if chain_df is not None and not chain_df.empty:
                self.chain_df = chain_df
                self.spot_price = spot
                self._last_refresh_ts = time.time()

                # Create IV mapper for batch computation with IV surface mapping
                from gex_utils import IVSurfaceMapper
                self._iv_mapper = IVSurfaceMapper(
                    strikes=chain_df['strike'].to_numpy(),
                    ivs=chain_df['iv'].to_numpy(),
                    option_types=chain_df['contractType'].apply(
                        lambda x: 'C' if x == 'call' else 'P'
                    ).to_numpy(),
                    reference_spot=spot
                )

                print(f"[RealtimeGEXProvider] Refreshed chain: {len(chain_df)} contracts, spot={spot:.2f}")
                return
        except Exception as e:
            print(f"[RealtimeGEXProvider] Fetch attempt failed: {e}")



    def compute_gex_score_batch_mapped(
        self,
        es_prices: pd.DataFrame,
        local_radius: float = 5.0,
        relative_radius: float = 15.0
    ) -> np.ndarray:
        

        from gex_utils import compute_batch_gamma_scores

        if self.chain_df is None or self.chain_df.empty or self._iv_mapper is None:
            return np.zeros(len(es_prices))


        # Convert ES prices to SPX equivalent
        spx_prices = es_prices['close'] + self.spx_es_spread

        # Map IVs using the IV surface mapper
        mapped_ivs = self._iv_mapper.map_ivs(spx_prices)  # (n_strikes, n_spots)

        # Get strikes and OI
        strikes = self.chain_df['strike'].to_numpy()
        ois = self.chain_df['oi'].fillna(0).to_numpy()


        # Build TTE matrix (n_spots x n_strikes), time is in GMT
        T_vec = es_prices.index.values.astype('datetime64[s]')
        expiry_dt = self.chain_df['expiry'].values.astype('datetime64[s]') + pd.Timedelta(hours=21)
        tte_matrix = (expiry_dt - T_vec[:, None]).astype('timedelta64[s]').astype(float) / (365.0 * 24 * 60 * 60)


        # Use shared batch computation
        scores = compute_batch_gamma_scores(
            spot_prices=spx_prices,
            strikes=strikes,
            iv_matrix=mapped_ivs.T,  # (n_spots, n_strikes)
            ois=ois,
            tte_matrix=tte_matrix,
            local_radius=local_radius,
            relative_radius=relative_radius
        )

        return scores


if __name__ == "__main__":
    pass

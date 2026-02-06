"""Tick-to-OHLC bar aggregation."""

import pandas as pd


class BarAggregator:
    """Converts raw tick/second data into 1-minute OHLCV bars."""

    @staticmethod
    def build_minute_bars(sec_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build 1-minute bars from second-level tick data.

        Args:
            sec_df: DataFrame with columns [timestamp, price, volume, buys, sells].
                    A ``dt`` column will be created from ``timestamp``.

        Returns:
            DataFrame indexed by minute (LA timezone) with OHLCV columns.
        """
        df = sec_df.copy()
        df['dt'] = (
            pd.to_datetime(df['timestamp'], unit='s', utc=True)
            .dt.tz_convert('America/Los_Angeles')
        )
        df.set_index('dt', inplace=True)

        ohlc = df['price'].resample('1Min').ohlc().ffill()

        agg_dict = {
            'price': 'mean',
            'buys': 'sum',
            'sells': 'sum',
            'volume': 'sum',
        }
        min_df = df.resample('1Min').agg(agg_dict).bfill()
        min_df = min_df.join(ohlc)

        return min_df

    @staticmethod
    def merge_incremental(
        existing: pd.DataFrame,
        new_ticks: pd.DataFrame,
        deque_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge incrementally-fetched ticks into an existing minute bar DataFrame.

        Steps:
        1. Build fresh minute bars from the deque window.
        2. Update overlapping bars in *existing* with deque values.
        3. Append any truly new minutes.

        Args:
            existing: Current 1-min bar DataFrame (LA-indexed).
            new_ticks: Newly fetched raw ticks (used to determine affected minutes).
            deque_df: All ticks in the rolling deque window.

        Returns:
            Updated minute bar DataFrame.
        """
        deque_min = BarAggregator.build_minute_bars(deque_df)

        # Determine which minutes the new ticks touch
        tmp = new_ticks.copy()
        tmp['dt'] = (
            pd.to_datetime(tmp['timestamp'], unit='s', utc=True)
            .dt.tz_convert('America/Los_Angeles')
        )
        tmp.set_index('dt', inplace=True)
        new_min = tmp.resample('1Min').agg({'price': 'last'})  # just need the index

        # Rows from the deque that overlap with new minutes
        overlap_idx = deque_min.index.intersection(new_min.index)
        if not overlap_idx.empty:
            new_min_df = deque_min.loc[overlap_idx].copy()
            existing.update(new_min_df)

        # Truly new minutes not yet in existing
        new_idx = new_min.index.difference(existing.index)
        if len(new_idx):
            new_rows = deque_min.loc[deque_min.index.isin(new_idx)].copy()
            existing = pd.concat([existing, new_rows])

        return existing

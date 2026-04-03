"""Merge historical ES 1-min data with existing dataset for extended training.

Historical data (es_historical_public222.txt) is in Eastern Time.
Existing data (es_min_3y_clean_td_gamma.csv) is in Pacific Time.
Converts historical ET -> PT by subtracting 3 hours, derives ovn/trading_day,
and fills nearby_gamma_score=0.0.
"""

import numpy as np
import pandas as pd


def merge_historical_data(
    hist_path: str,
    existing_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Merge historical ES 1-min data with existing dataset.

    Returns the merged DataFrame (with dt column, pre-lowercase).
    """
    # ── Load historical ──
    print(f"Loading historical data from {hist_path}...")
    hist = pd.read_csv(hist_path, skipinitialspace=True)
    hist.columns = hist.columns.str.strip()
    print(f"  {len(hist):,} rows, columns: {list(hist.columns)}")

    # Rename columns to match existing format
    hist = hist.rename(columns={
        'Bid Volume': 'BidVolume',
        'Ask Volume': 'AskVolume',
    })

    # Drop extra columns not in existing data
    drop_cols = ['# of Trades', 'OHLC Avg', 'HLC Avg', 'HL Avg']
    hist = hist.drop(columns=[c for c in drop_cols if c in hist.columns])

    # Parse datetime (ET) — strip fractional seconds first
    hist['Time'] = hist['Time'].str.split('.').str[0]
    hist['dt'] = pd.to_datetime(
        hist['Date'] + ' ' + hist['Time'],
        format='%Y/%m/%d %H:%M:%S',
    )

    # Convert ET -> PT (subtract 3 hours)
    hist['dt'] = hist['dt'] - pd.Timedelta(hours=3)

    # Reformat Date and Time columns from converted dt (now PT)
    hist['Date'] = hist['dt'].dt.strftime('%m/%d/%Y')
    hist['Time'] = hist['dt'].dt.strftime('%H:%M:%S')

    # Fill gamma score
    hist['nearby_gamma_score'] = 0.0

    print(f"  Date range (PT): {hist['dt'].min()} — {hist['dt'].max()}")

    # ── Load existing ──
    print(f"\nLoading existing data from {existing_path}...")
    existing = pd.read_csv(existing_path)
    existing['dt'] = pd.to_datetime(
        existing['Date'] + ' ' + existing['Time'],
        format='%m/%d/%Y %H:%M:%S',
    )
    print(f"  {len(existing):,} rows, {existing['trading_day'].nunique()} trading days")
    print(f"  Date range: {existing['dt'].min()} — {existing['dt'].max()}")

    # ── Cut overlap: keep historical only before existing starts ──
    cutoff = existing['dt'].min()
    n_before = len(hist)
    hist = hist[hist['dt'] < cutoff].copy()
    n_dropped = n_before - len(hist)
    print(f"\n  Overlap: dropped {n_dropped:,} historical rows at/after {cutoff}")

    # ── Derive ovn for historical (PT times) ──
    # RTH = 06:30-14:00 PT -> ovn=0, else ovn=1
    time_minutes = hist['dt'].dt.hour * 60 + hist['dt'].dt.minute
    rth_start = 6 * 60 + 30   # 06:30
    rth_end = 14 * 60          # 14:00
    hist['ovn'] = np.where(
        (time_minutes >= rth_start) & (time_minutes < rth_end), 0, 1,
    ).astype(int)

    # ── Derive trading_day for historical ──
    # Session boundary at 15:00 PT: bars with hour >= 15 start a new session
    # Bars with hour < 15 belong to the session that started previous calendar day
    mask_before = hist['dt'].dt.hour < 15
    shift = mask_before.astype(int)
    session_date = hist['dt'].dt.normalize() - pd.to_timedelta(shift, unit='D')
    hist['trading_day'] = pd.factorize(session_date, sort=True)[0]

    # Drop stub sessions (< 30 bars) — stray weekend/holiday bars in historical data
    session_counts = hist.groupby('trading_day').size()
    stub_sessions = session_counts[session_counts < 30].index
    if len(stub_sessions) > 0:
        n_stub_bars = hist[hist['trading_day'].isin(stub_sessions)].shape[0]
        hist = hist[~hist['trading_day'].isin(stub_sessions)].copy()
        hist['trading_day'] = pd.factorize(hist['trading_day'], sort=True)[0]
        print(f"  Dropped {len(stub_sessions)} stub sessions ({n_stub_bars} bars)")

    hist_days = hist['trading_day'].nunique()
    hist_max_td = hist['trading_day'].max()
    print(f"  Historical trading days: {hist_days}")

    # ── Renumber existing trading days ──
    existing['trading_day'] = existing['trading_day'] + hist_max_td + 1

    # ── Ensure matching columns and order ──
    output_cols = [
        'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'BidVolume', 'AskVolume', 'ovn', 'trading_day', 'nearby_gamma_score',
    ]
    hist_out = hist[output_cols + ['dt']]
    existing_out = existing[output_cols + ['dt']]

    # ── Concat and sort ──
    merged = pd.concat([hist_out, existing_out], ignore_index=True)
    merged = merged.sort_values('dt').reset_index(drop=True)

    # ── Save (drop dt helper, preserve original column casing) ──
    out = merged[output_cols].copy()

    print(f"\nSaving to {output_path}...")
    out.to_csv(output_path, index=False)

    # ── Sanity checks ──
    total_days = out['trading_day'].nunique()
    print(f"\n{'='*60}")
    print("MERGE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows:    {len(out):,}")
    print(f"  Trading days:  {total_days}")
    print(f"  Date range:    {out['Date'].iloc[0]} — {out['Date'].iloc[-1]}")

    # Join point
    join_idx = len(hist)
    if 0 < join_idx < len(merged):
        last_hist = merged.iloc[join_idx - 1]
        first_exist = merged.iloc[join_idx]
        print(f"  Join point:")
        print(f"    Last historical: {last_hist['Date']} {last_hist['Time']}  Close={last_hist['Close']}")
        print(f"    First existing:  {first_exist['Date']} {first_exist['Time']}  Close={first_exist['Close']}")
        gap = abs(first_exist['Close'] - last_hist['Close'])
        print(f"    Price gap: {gap:.2f} pts")

    # OVN distribution
    ovn_pct = out['ovn'].mean() * 100
    print(f"  OVN bars:      {ovn_pct:.1f}%")

    print(f"\nDone. Output: {output_path}")
    return merged

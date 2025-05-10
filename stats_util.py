import pandas as pd

def calc_ib_stats(curr_td_df, is_pst=True):

    rth_open_hr = 9
    rth_close_hr = 16
    ib_close_hr = 10

    if is_pst:
        rth_open_hr = 6
        rth_close_hr = 13
        ib_close_hr = 7

    rth_open = f'{rth_open_hr}:30'
    rth_close = f'{rth_close_hr}:00'
    ib_close = f'{ib_close_hr}:30'

    ib_stats = {}

    time_df = curr_td_df.set_index("dt")

    rth_df = time_df.between_time(rth_open, rth_close)

    if rth_df.empty:
        return ib_stats

    ib_df = time_df.between_time(rth_open, ib_close)

    ib_stats['ib_lo'] = ib_df['Low'].min()
    ib_stats['ib_hi'] = ib_df['High'].max()
    ib_stats['ib_range'] = ib_stats['ib_hi'] - ib_stats['ib_lo']

    ib_stats['rth_lo'] = rth_df['Low'].min()
    ib_stats['rth_hi'] = rth_df['High'].max()

    ib_stats['ib_hi_ext'] = (ib_stats['rth_hi'] - ib_stats['ib_hi']) / ib_stats['ib_range']
    ib_stats['ib_lo_ext'] = (ib_stats['ib_lo'] - ib_stats['rth_lo']) / ib_stats['ib_range']

    # curr_td_df.reset_index(level=1, inplace=True)

    return ib_stats
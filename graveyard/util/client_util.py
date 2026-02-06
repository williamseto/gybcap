
import datetime
import zoneinfo
import pandas as pd
import time

from sqlalchemy import create_engine

HOST = '172.30.144.1'

def get_prev_day_df() -> pd.DataFrame:
    conn = create_engine(
            f"mysql+mysqlconnector://kibblesoup:kibblesoup@{HOST}:3306/sys"
    )

    now_ts = int(time.time())
    prev_trading_day_start_ts = get_prev_trading_day_start_ts(now_ts)

    prev_trading_day_end_ts = prev_trading_day_start_ts + 22 * 3600

    prev_day_df = pd.read_sql_query(
        "SELECT * FROM price_data "
        "WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s "
        "ORDER BY timestamp ASC",
        conn,
        params=('ES', prev_trading_day_start_ts, prev_trading_day_end_ts)
    )

    return prev_day_df

def get_trading_day_start_ts(now_ts: int) -> int:
    """
    Returns the UTC timestamp of the current trading day's start,
    defined as 15:00 America/Los_Angeles on the day containing now_ts
    (or the previous calendar day if now < 15:00 PST).
    """
    la = zoneinfo.ZoneInfo("America/Los_Angeles")
    # localize now
    dt_local = datetime.datetime.fromtimestamp(now_ts, la)
    # decide which calendar date's 15:00 marks the start
    if dt_local.hour >= 15:
        day_start_local = dt_local.replace(hour=15, minute=0, second=0, microsecond=0)
    else:
        prev = dt_local.date() - datetime.timedelta(days=1)
        day_start_local = datetime.datetime(prev.year, prev.month, prev.day,
                                    15, 0, 0, tzinfo=la)
        
    print(day_start_local)
    # convert back to UTC timestamp
    return int(day_start_local.astimezone(datetime.timezone.utc).timestamp())

def get_prev_trading_day_start_ts(now_ts: int) -> int:
    """
    Returns the UTC timestamp of the previous trading day's start,
    defined as 15:00 America/Los_Angeles on the day containing now_ts
    (or the previous calendar day if now < 15:00 PST).
    """
    la = zoneinfo.ZoneInfo("America/Los_Angeles")
    # localize now
    dt_local = datetime.datetime.fromtimestamp(now_ts, la)
    # decide which calendar date's 15:00 marks the start

    day_delta = 1 if dt_local.hour >= 15 else 2

    # check days past Thursday, if weekend
    weekday_diff = dt_local.weekday() - 4
    if weekday_diff >= 0:
        day_delta += weekday_diff

    prev = dt_local.date() - datetime.timedelta(days=day_delta)
    day_start_local = datetime.datetime(prev.year, prev.month, prev.day,
                                15, 0, 0, tzinfo=la)
        
    # convert back to UTC timestamp
    return int(day_start_local.astimezone(datetime.timezone.utc).timestamp())

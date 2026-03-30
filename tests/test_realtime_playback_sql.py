import datetime as dt
from zoneinfo import ZoneInfo

from strategies.realtime.data_source import (
    get_prev_trading_day_start_ts,
    get_session_window_for_trading_day,
)
from strategies.realtime.runner import _parse_local_datetime_to_ts


LA_TZ = ZoneInfo("America/Los_Angeles")


def test_trading_day_session_window_matches_1500_roll() -> None:
    start_ts, end_ts = get_session_window_for_trading_day("2026-02-20")
    start_local = dt.datetime.fromtimestamp(start_ts, tz=LA_TZ)
    end_local = dt.datetime.fromtimestamp(end_ts, tz=LA_TZ)

    assert start_local == dt.datetime(2026, 2, 19, 15, 0, 0, tzinfo=LA_TZ)
    assert end_local == dt.datetime(2026, 2, 20, 14, 59, 59, tzinfo=LA_TZ)


def test_parse_local_datetime_to_ts_accepts_minute_and_seconds() -> None:
    ts_min = _parse_local_datetime_to_ts("2026-02-20 13:14")
    ts_sec = _parse_local_datetime_to_ts("2026-02-20 13:14:00")
    assert ts_min == ts_sec


def test_prev_trading_day_start_skips_weekend_for_monday_morning() -> None:
    now_local = dt.datetime(2026, 3, 30, 10, 0, 0, tzinfo=LA_TZ)  # Monday pre-15:00
    prev_start_ts = get_prev_trading_day_start_ts(int(now_local.timestamp()))
    prev_start_local = dt.datetime.fromtimestamp(prev_start_ts, tz=LA_TZ)
    assert prev_start_local == dt.datetime(2026, 3, 27, 15, 0, 0, tzinfo=LA_TZ)  # Friday

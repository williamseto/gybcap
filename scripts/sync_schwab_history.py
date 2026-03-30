#!/usr/bin/env python3
"""Sync local historical market data from Schwab.

Examples:

  # Sync ES minute bars into the dedicated Schwab history CSV
  ~/ml-venv/bin/python scripts/sync_schwab_history.py sync-minute --symbol /ES

  # One-time token initialization (browser + callback flow)
  ~/ml-venv/bin/python scripts/sync_schwab_history.py init-token

  # Save an options-chain snapshot (for future GEX workflows)
  ~/ml-venv/bin/python scripts/sync_schwab_history.py snapshot-options --symbol SPX

  # Install/update a daily weekday cron job at 15:20 local time
  ~/ml-venv/bin/python scripts/sync_schwab_history.py install-cron
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.data.history_sync import (
    DEFAULT_SCHWAB_MINUTE_CSV,
    MinuteHistorySyncConfig,
    sync_minute_history,
)
from strategies.data.schwab import (
    OptionChainRequest,
    PriceHistoryRequest,
    SchwabAuthConfig,
    SchwabClient,
)


def _build_client(*, allow_interactive: bool = False, manual_login: bool = False) -> SchwabClient:
    auth = SchwabAuthConfig.from_env(allow_missing=False)
    assert auth is not None
    auth.interactive_login = bool(auth.interactive_login or allow_interactive)
    auth.manual_login = bool(auth.manual_login or manual_login)
    return SchwabClient(auth)


def _normalize_manual_redirect_url(raw: str, callback_url: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise ValueError("manual redirect URL is empty")

    if text.startswith("http://") or text.startswith("https://"):
        return text

    base = callback_url.rstrip("/")
    if text.startswith("?"):
        return f"{base}{text}"
    if text.startswith("/?"):
        return f"{base}{text[1:]}"
    if text.startswith("code=") or text.startswith("error="):
        return f"{base}?{text}"
    return text


def _seed_token_via_manual_redirect(auth: SchwabAuthConfig, redirect_url: str) -> None:
    """Create/update token file using manual flow with a pre-supplied redirect URL."""
    from schwab import auth as schwab_auth

    token_path = Path(auth.token_cache_path).expanduser()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    received_url = _normalize_manual_redirect_url(redirect_url, auth.callback_url)

    # Reuse schwab-py's manual flow while bypassing interactive input prompt.
    with mock.patch("builtins.input", return_value=received_url):
        schwab_auth.client_from_manual_flow(
            api_key=auth.app_key,
            app_secret=auth.app_secret,
            callback_url=auth.callback_url,
            token_path=str(token_path),
            enforce_enums=auth.enforce_enums,
        )


def cmd_sync_minute(args: argparse.Namespace) -> int:
    client = _build_client()
    cfg = MinuteHistorySyncConfig(
        symbol=args.symbol,
        csv_path=args.csv,
        frequency_type=args.frequency_type,
        frequency=args.frequency,
        include_extended_hours=not args.no_extended_hours,
        lookback_days_if_missing=args.lookback_days_if_missing,
        stale_after_minutes=args.stale_after_minutes,
    )
    result = sync_minute_history(client, cfg)

    print(
        "status={status} symbol={symbol} rows_before={rows_before} rows_after={rows_after} "
        "rows_added={rows_added} candles_fetched={candles_fetched} latest_dt={latest_dt} "
        "csv={csv} message='{message}'".format(
            status=result.status,
            symbol=result.symbol,
            rows_before=result.rows_before,
            rows_after=result.rows_after,
            rows_added=result.rows_added,
            candles_fetched=result.candles_fetched,
            latest_dt=result.latest_dt,
            csv=result.csv_path,
            message=result.message,
        )
    )
    return 0


def cmd_init_token(args: argparse.Namespace) -> int:
    auth = SchwabAuthConfig.from_env(allow_missing=False)
    assert auth is not None
    token_path = Path(auth.token_cache_path).expanduser()
    if args.force and token_path.exists():
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = token_path.with_name(f"{token_path.name}.bak.{ts}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        token_path.replace(backup)
        print(f"moved_stale_token={backup}")

    if args.manual and args.redirect_url:
        _seed_token_via_manual_redirect(auth, args.redirect_url)
        client = _build_client()
    else:
        client = _build_client(allow_interactive=True, manual_login=args.manual)

    # Basic verification call so we fail early if token auth succeeded but market data scope is broken.
    now_utc = dt.datetime.now(tz=dt.timezone.utc)
    req = PriceHistoryRequest(
        symbol=args.verify_symbol,
        start_ms=int((now_utc - dt.timedelta(days=1)).timestamp() * 1000),
        end_ms=int(now_utc.timestamp() * 1000),
        frequency_type="minute",
        frequency=1,
        need_extended_hours_data=True,
        need_previous_close=False,
    )
    candles = client.fetch_price_history(req)

    print(
        "token_ready=1 token_path={token_path} verify_symbol={symbol} candles={candles}".format(
            token_path=client.token_cache_path,
            symbol=args.verify_symbol,
            candles=len(candles),
        )
    )
    return 0


def cmd_snapshot_options(args: argparse.Namespace) -> int:
    client = _build_client()
    req = OptionChainRequest(
        symbol=args.symbol,
        contract_type=args.contract_type,
        strike_count=args.strike_count,
        include_quotes=not args.no_quotes,
        strategy="SINGLE",
    )
    payload = client.fetch_option_chain(req)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_clean = args.symbol.replace("/", "_")
    out_path = out_dir / f"{symbol_clean}_options_{ts}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"saved={out_path}")
    return 0


def _cron_line(python_path: str, repo_root: Path, args: argparse.Namespace) -> str:
    log_path = repo_root / "logs" / "schwab_history_sync.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = (
        f"cd {repo_root} && {python_path} scripts/sync_schwab_history.py sync-minute "
        f"--symbol {args.symbol} --csv {args.csv} --stale-after-minutes {args.stale_after_minutes} "
        f"--lookback-days-if-missing {args.lookback_days_if_missing}"
    )
    if args.no_extended_hours:
        command += " --no-extended-hours"
    return f"{args.schedule} {command} >> {log_path} 2>&1 # gybcap-schwab-history-sync"


def cmd_print_cron(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    python_path = args.python_path or os.path.expanduser("~/ml-venv/bin/python")
    print(_cron_line(python_path=python_path, repo_root=repo_root, args=args))
    return 0


def cmd_install_cron(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    python_path = args.python_path or os.path.expanduser("~/ml-venv/bin/python")
    new_line = _cron_line(python_path=python_path, repo_root=repo_root, args=args)

    try:
        existing = subprocess.run(
            ["crontab", "-l"],
            check=False,
            capture_output=True,
            text=True,
        )
        current_lines = existing.stdout.splitlines() if existing.returncode == 0 else []
    except FileNotFoundError as exc:
        raise RuntimeError("crontab command not found on this system") from exc

    filtered = [ln for ln in current_lines if "gybcap-schwab-history-sync" not in ln]
    filtered.append(new_line)
    new_content = "\n".join(filtered) + "\n"

    apply_res = subprocess.run(
        ["crontab", "-"],
        input=new_content,
        text=True,
        capture_output=True,
        check=False,
    )
    if apply_res.returncode != 0:
        raise RuntimeError(f"Failed to install cron entry: {apply_res.stderr.strip()}")

    print("installed cron entry:")
    print(new_line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schwab historical data sync")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init-token", help="Initialize/update Schwab token file via schwab-py")
    init_p.add_argument(
        "--manual",
        action="store_true",
        help="Use manual copy/paste OAuth flow instead of local callback server.",
    )
    init_p.add_argument(
        "--verify-symbol",
        default="SPY",
        help="Symbol used for a post-auth verification call (default: SPY).",
    )
    init_p.add_argument(
        "--redirect-url",
        default=os.getenv("SCHWAB_MANUAL_REDIRECT_URL", None),
        help=(
            "Manual flow redirect URL to bypass prompt. Accepts full URL or just "
            "query payload (e.g. '?code=...'). Defaults to SCHWAB_MANUAL_REDIRECT_URL."
        ),
    )
    init_p.add_argument(
        "--force",
        action="store_true",
        help="Move existing token file aside before login (use when refresh token is invalid).",
    )
    init_p.set_defaults(func=cmd_init_token)

    sync_p = sub.add_parser("sync-minute", help="Sync minute bars into local CSV")
    sync_p.add_argument("--symbol", default="/ES", help="Schwab symbol (default: /ES)")
    sync_p.add_argument(
        "--csv",
        default=DEFAULT_SCHWAB_MINUTE_CSV,
        help="Output CSV path (default keeps Schwab sync separate from legacy raw_data CSVs)",
    )
    sync_p.add_argument("--frequency-type", default="minute", help="frequencyType for pricehistory")
    sync_p.add_argument("--frequency", type=int, default=1, help="frequency for pricehistory")
    sync_p.add_argument(
        "--lookback-days-if-missing",
        type=int,
        default=365,
        help="Initial backfill window when CSV does not exist",
    )
    sync_p.add_argument(
        "--stale-after-minutes",
        type=int,
        default=30,
        help="Skip remote fetch when local CSV is newer than this",
    )
    sync_p.add_argument(
        "--no-extended-hours",
        action="store_true",
        help="Disable extended-hours candles",
    )
    sync_p.set_defaults(func=cmd_sync_minute)

    opt_p = sub.add_parser("snapshot-options", help="Save option-chain snapshot to JSON")
    opt_p.add_argument("--symbol", default="SPX", help="Underlying symbol for chain snapshot")
    opt_p.add_argument("--contract-type", default="ALL", choices=["CALL", "PUT", "ALL"])
    opt_p.add_argument("--strike-count", type=int, default=50)
    opt_p.add_argument("--no-quotes", action="store_true", help="Skip quote payload in chain response")
    opt_p.add_argument("--out-dir", default="raw_data/options")
    opt_p.set_defaults(func=cmd_snapshot_options)

    print_p = sub.add_parser("print-cron", help="Print recommended daily cron line")
    print_p.add_argument("--schedule", default="20 15 * * 1-5", help="Cron schedule")
    print_p.add_argument("--python-path", default=None, help="Python interpreter path")
    print_p.add_argument("--symbol", default="/ES")
    print_p.add_argument("--csv", default=DEFAULT_SCHWAB_MINUTE_CSV)
    print_p.add_argument("--lookback-days-if-missing", type=int, default=365)
    print_p.add_argument("--stale-after-minutes", type=int, default=60)
    print_p.add_argument("--no-extended-hours", action="store_true")
    print_p.set_defaults(func=cmd_print_cron)

    install_p = sub.add_parser("install-cron", help="Install/update daily cron entry")
    install_p.add_argument("--schedule", default="20 15 * * 1-5", help="Cron schedule")
    install_p.add_argument("--python-path", default=None, help="Python interpreter path")
    install_p.add_argument("--symbol", default="/ES")
    install_p.add_argument("--csv", default=DEFAULT_SCHWAB_MINUTE_CSV)
    install_p.add_argument("--lookback-days-if-missing", type=int, default=365)
    install_p.add_argument("--stale-after-minutes", type=int, default=60)
    install_p.add_argument("--no-extended-hours", action="store_true")
    install_p.set_defaults(func=cmd_install_cron)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch all newsletter emails and build comprehensive range dataset.

Fetches emails from the newsletter sender, parses them using the existing
regex/BeautifulSoup parser (scrape/gmail_fetcher.py), and outputs a clean CSV
with all instruments and timeframes.

Usage:
    python -m strategies.range_predictor.build_dataset [--since DATE] [--output PATH]
"""

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Reuse the existing parser from scrape/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scrape.gmail_fetcher import (
    extract_df_from_html,
    preclean_html_for_numbers,
    better_num,
)


# ── Gmail helpers ──────────────────────────────────────────────────────

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SENDER = os.environ.get("NEWSLETTER_SENDER", "")


def _get_gmail_service():
    """Authenticate and return Gmail API service."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    token_path = Path("scrape/token.json")
    creds_path = Path("scrape/credentials.json")

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, "w") as f:
                f.write(creds.to_json())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow

            if not creds_path.exists():
                raise FileNotFoundError(
                    f"{creds_path} not found. Set up Gmail OAuth first."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(creds_path), SCOPES
            )
            creds = flow.run_local_server(port=0)
            with open(token_path, "w") as f:
                f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _list_all_message_ids(service, since=None, before=None):
    """List all message IDs from newsletter sender, handling pagination."""
    q = f"from:{SENDER}"
    if since:
        q += f" after:{since.replace('-', '/')}"
    if before:
        q += f" before:{before.replace('-', '/')}"

    all_ids = []
    page_token = None

    while True:
        resp = service.users().messages().list(
            userId="me", q=q, maxResults=500, pageToken=page_token
        ).execute()
        messages = resp.get("messages", [])
        all_ids.extend(m["id"] for m in messages)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return all_ids


def _get_message(service, msg_id):
    """Fetch single message content."""
    msg = service.users().messages().get(
        userId="me", id=msg_id, format="full"
    ).execute()

    msg_date_secs = int(msg["internalDate"]) / 1000
    payload = msg.get("payload", {})
    parts = payload.get("parts", [])
    text = None
    html = None

    def decode_body(part):
        body = part.get("body", {}).get("data")
        if not body:
            return None
        data = base64.urlsafe_b64decode(body.encode("ASCII"))
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode("latin-1", errors="replace")

    if parts:
        stack = list(parts)
        while stack:
            p = stack.pop(0)
            mime = p.get("mimeType", "")
            if mime == "text/plain" and not text:
                text = decode_body(p)
            elif mime == "text/html" and not html:
                html = decode_body(p)
            if "parts" in p:
                stack.extend(p["parts"])
    else:
        mime = payload.get("mimeType", "")
        if mime == "text/plain":
            text = decode_body(payload)
        elif mime == "text/html":
            html = decode_body(payload)

    return {
        "id": msg_id,
        "snippet": msg.get("snippet", ""),
        "text": text,
        "html": html,
        "msg_date_secs": msg_date_secs,
    }


# ── Parsing ────────────────────────────────────────────────────────────

def _get_trading_date(msg_date_secs):
    """Convert email timestamp to trading date."""
    dt = datetime.fromtimestamp(msg_date_secs)
    if dt.hour >= 15 or dt.isoweekday() >= 6:
        return dt.strftime("%Y-%m-%d")
    prev = dt.date() - timedelta(days=1)
    return prev.strftime("%Y-%m-%d")


MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def _normalize_section_name(name):
    """Map section names to standard timeframe names."""
    name = name.lower().strip()
    if "day" in name or "daily" in name:
        return "daily"
    elif "week" in name:
        return "weekly"
    elif "month" in name:
        return "monthly"
    elif "quarter" in name:
        return "quarterly"
    # Check for raw month names like "october 2025"
    first_word = name.split()[0] if name.split() else ""
    if first_word in MONTH_NAMES:
        return "monthly"
    return name


def _normalize_symbol(sym):
    """Clean and normalize instrument symbol."""
    if not sym or not isinstance(sym, str):
        return None
    sym = sym.strip().upper()
    # Remove ratings like "(+)" from symbol
    sym = re.sub(r'\s*\([^)]*\)\s*', '', sym).strip()
    # Common aliases
    aliases = {
        'E-MINI': 'ES', 'E MINI': 'ES', 'S&P': 'ES', 'SPX': 'SPX',
        'NASDAQ': 'NQ', 'RUSSELL': 'RTY', 'DOW': 'YM',
        'CRUDE': 'CL', 'GOLD': 'GC', 'BONDS': 'ZB', 'NOTES': 'ZN',
        'EURO': 'EUR', 'DOLLAR': 'DX',
    }
    for alias, canonical in aliases.items():
        if alias in sym:
            return canonical
    # Return first word if multi-word
    parts = sym.split()
    if parts:
        return parts[0]
    return sym


def _extract_rating(text):
    """Extract rating (+, 0, -) from text."""
    if not text or not isinstance(text, str):
        return None
    m = re.search(r'\(([+\-0])\)', text)
    if m:
        return m.group(1)
    return None


def _safe_float(val):
    """Convert value to float, handling various formats."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip().replace(',', '').replace('\u00a0', '')
    # Handle tick notation (bonds): 112'170 -> 112.53125 (not needed for ES/VIX)
    if "'" in val:
        return None  # Skip bond-style notation
    try:
        return float(val)
    except (ValueError, TypeError):
        result = better_num(val)
        return float(result) if result is not None else None


def parse_email_to_rows(raw_email):
    """Parse a single email into a list of prediction rows.

    Returns list of dicts with keys:
        date, timeframe, symbol, range_low, range_high,
        inner_sup, inner_res, rating
    """
    html = raw_email.get("html")
    text = raw_email.get("text")
    trading_date = _get_trading_date(raw_email["msg_date_secs"])

    if not html and not text:
        return []

    # Try HTML parser first
    content = html or text
    if html:
        content = preclean_html_for_numbers(html)

    try:
        result = extract_df_from_html(content)
    except Exception:
        return []

    rows = []

    if isinstance(result, dict):
        # Text-only parser returns dict of DataFrames per section
        for section_key, section_df in result.items():
            timeframe = _normalize_section_name(section_key)
            if section_df is None or (hasattr(section_df, 'empty') and section_df.empty):
                continue
            for _, r in section_df.iterrows():
                symbol = _normalize_symbol(r.get("symbol"))
                if not symbol:
                    continue
                low = _safe_float(r.get("range_low"))
                high = _safe_float(r.get("range_high"))
                if low is None or high is None:
                    continue
                # Ensure low < high
                if low > high:
                    low, high = high, low
                rating = r.get("rating_raw")
                if rating:
                    rating = _extract_rating(f"({rating})") or rating.strip()
                rows.append({
                    "date": trading_date,
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "range_low": low,
                    "range_high": high,
                    "inner_sup": None,
                    "inner_res": None,
                    "rating": rating,
                })

    elif isinstance(result, pd.DataFrame) and not result.empty:
        # HTML table parser returns combined DataFrame with 'section' column
        for _, r in result.iterrows():
            section = r.get("section", "")
            timeframe = _normalize_section_name(section)
            symbol = _normalize_symbol(r.get("symbol"))
            if not symbol:
                continue

            low = _safe_float(r.get("range_low"))
            high = _safe_float(r.get("range_high"))
            if low is None or high is None:
                continue
            if low > high:
                low, high = high, low

            inner_sup = _safe_float(r.get("inner_sup"))
            inner_res = _safe_float(r.get("inner_res"))

            rows.append({
                "date": trading_date,
                "timeframe": timeframe,
                "symbol": symbol,
                "range_low": low,
                "range_high": high,
                "inner_sup": inner_sup,
                "inner_res": inner_res,
                "rating": None,
            })

    return rows


# ── Main pipeline ──────────────────────────────────────────────────────

def build_dataset(
    since=None,
    before=None,
    output_path="data/newsletter_ranges.csv",
    cache_dir="data/email_cache",
    verbose=True,
):
    """Fetch all emails and build comprehensive range dataset.

    Args:
        since: Fetch emails since this date (YYYY-MM-DD).
        before: Fetch emails before this date.
        output_path: Output CSV path.
        cache_dir: Directory for caching raw emails.
        verbose: Print progress.

    Returns:
        DataFrame with all parsed predictions.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Authenticating with Gmail...")
    service = _get_gmail_service()

    if verbose:
        print("Listing messages...")
    msg_ids = _list_all_message_ids(service, since=since, before=before)
    if verbose:
        print(f"Found {len(msg_ids)} messages")

    # Fetch and parse all emails
    all_rows = []
    n_cached = 0
    n_fetched = 0
    n_errors = 0

    for i, mid in enumerate(msg_ids):
        # Check cache
        cache_file = cache_path / f"{mid}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                raw = json.load(f)
            n_cached += 1
        else:
            try:
                raw = _get_message(service, mid)
                # Cache the raw email
                with open(cache_file, "w") as f:
                    json.dump(raw, f)
                n_fetched += 1
            except Exception as e:
                if verbose:
                    print(f"  ERROR fetching {mid}: {e}")
                n_errors += 1
                continue

        # Parse
        try:
            rows = parse_email_to_rows(raw)
            all_rows.extend(rows)
        except Exception as e:
            if verbose and i < 20:
                print(f"  Parse error on {mid}: {e}")
            n_errors += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(msg_ids)} "
                  f"(cached: {n_cached}, fetched: {n_fetched}, "
                  f"errors: {n_errors}, rows: {len(all_rows)})")

    if verbose:
        print(f"\nDone: {n_cached} cached, {n_fetched} fetched, {n_errors} errors")
        print(f"Total prediction rows: {len(all_rows)}")

    if not all_rows:
        print("WARNING: No predictions extracted")
        return pd.DataFrame()

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    # Sort by date descending, then timeframe
    timeframe_order = {"daily": 0, "weekly": 1, "monthly": 2, "quarterly": 3}
    df["_tf_order"] = df["timeframe"].map(timeframe_order).fillna(9)
    df = df.sort_values(["date", "_tf_order", "symbol"]).drop(columns=["_tf_order"])

    # Remove exact duplicates (same date + timeframe + symbol)
    df = df.drop_duplicates(subset=["date", "timeframe", "symbol"], keep="first")

    # Add range width
    df["range_width"] = df["range_high"] - df["range_low"]

    # Reorder columns
    cols = [
        "date", "timeframe", "symbol",
        "range_low", "range_high", "range_width",
        "inner_sup", "inner_res", "rating",
    ]
    df = df[[c for c in cols if c in df.columns]]

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"\nSaved to {output_path}")
        print(f"\nDataset summary:")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Total rows: {len(df)}")
        print(f"\n  By timeframe:")
        for tf, count in df["timeframe"].value_counts().items():
            print(f"    {tf}: {count}")
        print(f"\n  By symbol:")
        for sym, count in df["symbol"].value_counts().head(15).items():
            print(f"    {sym}: {count}")

        # Show ES daily sample
        es_daily = df[(df["symbol"] == "ES") & (df["timeframe"] == "daily")]
        if len(es_daily) > 0:
            print(f"\n  ES daily predictions: {len(es_daily)}")
            print(f"  ES daily avg width: {es_daily['range_width'].mean():.1f} pts")
            print(f"\n  Recent ES daily predictions:")
            print(es_daily.head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build newsletter range prediction dataset"
    )
    parser.add_argument("--since", help="Fetch emails since date (YYYY-MM-DD)")
    parser.add_argument("--before", help="Fetch emails before date")
    parser.add_argument(
        "--output", "-o",
        default="data/newsletter_ranges.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/email_cache",
        help="Cache directory for raw emails",
    )
    args = parser.parse_args()

    build_dataset(
        since=args.since,
        before=args.before,
        output_path=args.output,
        cache_dir=args.cache_dir,
    )

"""LLM-based email parser using Claude Haiku.

Replaces 400+ lines of regex/BeautifulSoup parsing with a single reliable
LLM call per email. Uses claude-haiku-4-5-20251001 (~$0.001/email).

Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from strategies.range_predictor.newsletter.email_pipeline.cache import EmailCache


PARSE_PROMPT = """\
Extract all range predictions from this newsletter email. The email contains
daily, weekly, and monthly (sometimes quarterly) range predictions for various
financial instruments (ES, VIX, NQ, etc.).

For each prediction, extract:
- symbol: The instrument symbol (e.g., "ES", "VIX", "NQ", "YM")
- range_low: The lower bound of the range (float)
- range_high: The upper bound of the range (float)
- inner_sup: Inner support level if present (float or null)
- inner_res: Inner resistance level if present (float or null)
- rating: Rating if present (e.g., "+", "0", "-", or null)

Return valid JSON in this exact format:
{
    "daily": [
        {"symbol": "ES", "range_low": 6746.0, "range_high": 6857.0, "inner_sup": 6780.0, "inner_res": 6820.0, "rating": "+"},
        {"symbol": "VIX", "range_low": 15.41, "range_high": 17.19, "inner_sup": null, "inner_res": null, "rating": null}
    ],
    "weekly": [...],
    "monthly": [...],
    "quarterly": [...]
}

If a timeframe section is not present, use an empty list [].
Only return the JSON, no other text.

EMAIL CONTENT:
"""


def _get_trading_date(msg_date_secs: float) -> str:
    """Convert email timestamp to trading date.

    Emails sent before 3pm belong to the previous trading day.
    Weekend emails belong to their date.
    """
    dt = datetime.fromtimestamp(msg_date_secs)
    if dt.hour >= 15 or dt.isoweekday() >= 6:
        return dt.strftime("%Y-%m-%d")
    prev = dt.date() - timedelta(days=1)
    return prev.strftime("%Y-%m-%d")


def parse_single_email(raw_email: Dict) -> Dict:
    """Parse a single raw email using Claude Haiku.

    Args:
        raw_email: Dict with keys: id, html, text, msg_date_secs

    Returns:
        Dict with keys: date, msg_id, predictions
    """
    import anthropic

    client = anthropic.Anthropic()

    # Prefer HTML (more structured), fall back to text
    content = raw_email.get("html") or raw_email.get("text") or ""
    if not content:
        return {
            "date": _get_trading_date(raw_email["msg_date_secs"]),
            "msg_id": raw_email["id"],
            "predictions": {},
        }

    # Truncate very long emails
    if len(content) > 15000:
        content = content[:15000]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": PARSE_PROMPT + content},
        ],
    )

    # Extract JSON from response
    response_text = response.content[0].text.strip()

    # Handle markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        response_text = "\n".join(lines)

    try:
        predictions = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse JSON for email {raw_email['id']}")
        predictions = {}

    return {
        "date": _get_trading_date(raw_email["msg_date_secs"]),
        "msg_id": raw_email["id"],
        "predictions": predictions,
    }


def parse_all_emails(
    raw_emails: List[Dict],
    cache: Optional[EmailCache] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Parse all raw emails, using cache when available.

    Args:
        raw_emails: List of raw email dicts from fetcher.
        cache: EmailCache for storing parsed results.
        verbose: Print progress.

    Returns:
        List of parsed prediction dicts.
    """
    results = []
    n_cached = 0
    n_parsed = 0

    for i, raw in enumerate(raw_emails):
        msg_id = raw["id"]

        if cache and cache.has_parsed(msg_id):
            parsed = cache.load_parsed(msg_id)
            n_cached += 1
        else:
            parsed = parse_single_email(raw)
            if cache:
                cache.save_parsed(msg_id, parsed)
            n_parsed += 1

        results.append(parsed)

        if verbose and (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(raw_emails)} "
                  f"(cached: {n_cached}, parsed: {n_parsed})")

    if verbose:
        print(f"Done: {n_cached} cached, {n_parsed} newly parsed")

    return results

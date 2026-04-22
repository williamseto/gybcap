"""Gmail API fetcher for newsletter emails.

Extracted from scrape/gmail_fetcher.py — handles OAuth auth, message listing,
and raw HTML/text extraction. Parameterized for reuse.
"""

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from strategies.range_predictor.newsletter.email_pipeline.cache import EmailCache


# Gmail API scope
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Default auth file locations (relative to project root)
DEFAULT_CREDENTIALS = Path("scrape/credentials.json")
DEFAULT_TOKEN = Path("scrape/token.json")

# Newsletter sender — set via NEWSLETTER_SENDER env var or pass explicitly
DEFAULT_SENDER = os.environ.get("NEWSLETTER_SENDER", "")


def get_gmail_service(
    credentials_path: Path = DEFAULT_CREDENTIALS,
    token_path: Path = DEFAULT_TOKEN,
):
    """Return an authorized Gmail API service.

    On first run, opens a browser for OAuth consent and saves token.
    Subsequent runs use the cached token.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"{credentials_path} not found. "
                    "Create OAuth 2.0 Client ID in Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def list_message_ids(
    service,
    sender: str = DEFAULT_SENDER,
    since: Optional[str] = None,
    before: Optional[str] = None,
    max_results: int = 2000,
) -> List[str]:
    """List message IDs from a sender.

    Args:
        service: Gmail API service.
        sender: Sender email address.
        since: Date string (YYYY-MM-DD or YYYY/MM/DD).
        before: Date string (YYYY-MM-DD or YYYY/MM/DD).
        max_results: Max messages to return.

    Returns:
        List of Gmail message IDs.
    """
    q = f"from:{sender}"
    if since:
        q += f" after:{since.replace('-', '/')}"
    if before:
        q += f" before:{before.replace('-', '/')}"

    all_ids = []
    page_token = None

    while True:
        response = service.users().messages().list(
            userId="me", q=q, maxResults=min(max_results - len(all_ids), 500),
            pageToken=page_token,
        ).execute()

        messages = response.get("messages", [])
        all_ids.extend(m["id"] for m in messages)

        page_token = response.get("nextPageToken")
        if not page_token or len(all_ids) >= max_results:
            break

    return all_ids


def get_message_content(service, msg_id: str) -> Dict:
    """Fetch a single message's HTML and text content.

    Returns:
        Dict with keys: id, snippet, text, html, msg_date_secs
    """
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
        "snippet": msg.get("snippet"),
        "text": text,
        "html": html,
        "msg_date_secs": msg_date_secs,
    }


def fetch_emails(
    since: Optional[str] = None,
    before: Optional[str] = None,
    cache: Optional[EmailCache] = None,
    sender: str = DEFAULT_SENDER,
) -> List[Dict]:
    """Fetch all newsletter emails, using cache when available.

    Args:
        since: Fetch emails after this date (YYYY-MM-DD).
        before: Fetch emails before this date.
        cache: EmailCache instance for storing raw emails.
        sender: Sender email address.

    Returns:
        List of raw email dicts (id, html, text, msg_date_secs).
    """
    service = get_gmail_service()
    msg_ids = list_message_ids(service, sender=sender, since=since, before=before)
    print(f"Found {len(msg_ids)} message IDs")

    results = []
    for i, mid in enumerate(msg_ids):
        # Check cache first
        if cache and cache.has_raw(mid):
            raw = cache.load_raw(mid)
        else:
            raw = get_message_content(service, mid)
            if cache:
                cache.save_raw(mid, raw)

        results.append(raw)

        if (i + 1) % 50 == 0:
            print(f"  Fetched {i + 1}/{len(msg_ids)}")

    return results

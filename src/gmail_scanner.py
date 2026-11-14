
"""URLSight Gmail scanner.

Reuses token.json created by auth_gmail.py.
Extracts links locally; it does not send or modify emails.
"""

import base64
import re
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


PROJECT_DIR = Path(__file__).resolve().parents[1]
TOKEN_FILE = PROJECT_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MAX_URLS = 200

URL_PATTERN = re.compile(r"""https?://[^\s<>"'`]+""", re.IGNORECASE)


def get_gmail_service():
    """Load the existing token and create the Gmail API client."""
    if not TOKEN_FILE.exists():
        raise RuntimeError(
            "Gmail token not found. Run `python auth_gmail.py` "
            "from the project root."
        )

    try:
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_FILE), SCOPES
        )
    except (ValueError, OSError) as exc:
        raise RuntimeError(
            "Could not read token.json. Run `python auth_gmail.py` again."
        ) from exc

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(
                "Gmail token refresh failed. Run `python auth_gmail.py` again."
            ) from exc

    if not creds.valid:
        raise RuntimeError(
            "Gmail authorization is invalid. Run `python auth_gmail.py` again."
        )

    return build(
        "gmail",
        "v1",
        credentials=creds,
        cache_discovery=False,
    )


def _decode_body(data):
    """Decode a Gmail base64url-encoded body part."""
    if not data:
        return ""

    try:
        decoded = base64.urlsafe_b64decode(
            data + "=" * (-len(data) % 4)
        )
        return decoded.decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return ""


def _collect_body_parts(payload):
    """Collect plain-text and HTML body parts, skipping attachments."""
    text_parts = []
    html_parts = []

    def visit(part):
        if part.get("filename"):
            return

        mime_type = part.get("mimeType", "")
        data = part.get("body", {}).get("data")

        if data and mime_type == "text/plain":
            text_parts.append(_decode_body(data))
        elif data and mime_type == "text/html":
            html_parts.append(_decode_body(data))

        for child in part.get("parts", []):
            visit(child)

    visit(payload)
    return text_parts, html_parts


def _normalize_url(candidate):
    """Validate HTTP(S) URLs and trim common trailing punctuation."""
    candidate = candidate.strip().rstrip(".,;:!?)]}")

    try:
        parsed = urlsplit(candidate)

        if parsed.scheme.lower() not in {"http", "https"}:
            return None
        if not parsed.hostname:
            return None
        if parsed.username or parsed.password:
            return None

        # Validate a supplied port.
        _ = parsed.port
        return candidate

    except (ValueError, TypeError):
        return None


def _extract_links(text_parts, html_parts):
    """Extract and deduplicate HTTP(S) links from message content."""
    candidates = []

    for text in text_parts:
        candidates.extend(URL_PATTERN.findall(text))

    for html in html_parts:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup.find_all(href=True):
            href = tag.get("href", "").strip()
            if href.lower().startswith(("http://", "https://")):
                candidates.append(href)

        candidates.extend(
            URL_PATTERN.findall(soup.get_text(" ", strip=True))
        )

    urls = []
    seen = set()

    for candidate in candidates:
        url = _normalize_url(candidate)

        if url and url not in seen:
            seen.add(url)
            urls.append(url)

            if len(urls) >= MAX_URLS:
                break

    return urls


def scan_recent_email_links(limit=5):
    """Extract unique URLs from the five most recent inbox messages."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")

    # Enforce the requested maximum of five emails.
    limit = min(limit, 5)

    service = get_gmail_service()

    response = (
        service.users()
        .messages()
        .list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=limit,
        )
        .execute()
    )

    messages = response.get("messages", [])[:limit]
    urls = []
    seen = set()

    for message in messages:
        full_message = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message["id"],
                format="full",
            )
            .execute()
        )

        text_parts, html_parts = _collect_body_parts(
            full_message.get("payload", {})
        )

        for url in _extract_links(text_parts, html_parts):
            if url not in seen:
                seen.add(url)
                urls.append(url)

                if len(urls) >= MAX_URLS:
                    return urls

    return urls


# Compatibility with older app.py imports.
def scan_labeled_email_links():
    """Legacy function name; scans the five most recent inbox emails."""
    return scan_recent_email_links(limit=5)


import base64

from googleapiclient.discovery import build


def get_gmail_service(credentials):

    return build(
        "gmail",
        "v1",
        credentials=credentials,
        cache_discovery=False
    )


def get_gmail_profile(credentials):

    service = get_gmail_service(credentials)

    return service.users().getProfile(
        userId="me"
    ).execute()


def decode_body(data):

    if not data:
        return ""

    try:

        padding = "=" * (-len(data) % 4)

        decoded = base64.urlsafe_b64decode(
            data + padding
        )

        return decoded.decode(
            "utf-8",
            errors="replace"
        )

    except Exception:
        return ""


def extract_body(payload):

    mime_type = payload.get("mimeType", "")

    body_data = payload.get(
        "body", {}
    ).get("data")

    # Prefer plain text.
    if mime_type == "text/plain" and body_data:
        return decode_body(body_data)

    # Search nested MIME parts.
    parts = payload.get("parts", [])

    plain_text = ""
    html_text = ""

    for part in parts:

        result = extract_body(part)

        if not result:
            continue

        part_type = part.get("mimeType", "")

        if part_type == "text/plain":
            plain_text += result

        elif part_type == "text/html":
            html_text += result

    if plain_text:
        return plain_text

    if html_text:
        return html_text

    if body_data:
        return decode_body(body_data)

    return ""


def fetch_emails(credentials, max_results=20):

    service = get_gmail_service(credentials)

    response = service.users().messages().list(
        userId="me",
        maxResults=max_results,
        q="in:inbox"
    ).execute()

    messages = response.get("messages", [])

    emails = []

    for message in messages:

        details = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full"
        ).execute()

        payload = details.get("payload", {})

        headers = payload.get("headers", [])

        header_map = {
            header["name"].lower(): header["value"]
            for header in headers
        }

        emails.append({

            "id": message["id"],

            "sender": header_map.get(
                "from", "Unknown"
            ),

            "to": header_map.get(
                "to", ""
            ),

            "subject": header_map.get(
                "subject", "No subject"
            ),

            "date": header_map.get(
                "date", ""
            ),

            "snippet": details.get(
                "snippet", ""
            ),

            "body": extract_body(payload),

            "thread_id": details.get(
                "threadId", ""
            )

        })

    return emails

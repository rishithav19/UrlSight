
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

PROJECT_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = PROJECT_DIR / "credentials" / "client_secret.json"
TOKEN_FILE = PROJECT_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
REDIRECT_URI = "http://localhost:8765/"


def main():
    creds = None

    # Load existing credentials
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                str(TOKEN_FILE), SCOPES
            )

            # Refresh expired credentials
            if creds.expired and creds.refresh_token:
                print("Refreshing existing Gmail token...")
                creds.refresh(Request())

        except RefreshError as e:
            print(f"Existing token is invalid: {e}")
            print("Starting fresh Gmail authentication...")
            creds = None
            TOKEN_FILE.unlink(missing_ok=True)

        except Exception as e:
            print(f"Could not load existing token: {e}")
            creds = None
            TOKEN_FILE.unlink(missing_ok=True)

    # Start fresh authentication if necessary
    if not creds or not creds.valid:

        if not CREDENTIALS_FILE.exists():
            raise FileNotFoundError(
                f"OAuth credentials not found: {CREDENTIALS_FILE}"
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            SCOPES,
            redirect_uri=REDIRECT_URI
        )

        auth_url, _ = flow.authorization_url(
            access_type="offline",
            prompt="consent",
        )

        print("\n1. Open this URL in your Windows browser:\n")
        print(auth_url)

        print(
            "\n2. Sign in to your Google account and approve access."
            "\n3. Copy the FULL redirected URL from the browser address bar."
            "\n   It should start with http://localhost:8765/"
            "\n"
        )

        response_url = input("Paste the full redirected URL: ").strip()

        flow.fetch_token(authorization_response=response_url)
        creds = flow.credentials

    # Save credentials
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    print("\nGmail authentication successful!")
    print(f"Token saved at: {TOKEN_FILE}")


if __name__ == "__main__":
    main()

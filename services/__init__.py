
import json
import sqlite3
from pathlib import Path

import requests
import streamlit as st

from cryptography.fernet import Fernet
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "gmail_tokens.db"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly"
]


# ---------------------------------------
# Database and encryption
# ---------------------------------------

def get_fernet():
    key = st.secrets["security"]["token_encryption_key"]
    return Fernet(key.encode())


def init_database():

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS gmail_tokens (
                email TEXT PRIMARY KEY,
                encrypted_token BLOB NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS oauth_states (
                state TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


# ---------------------------------------
# OAuth configuration
# ---------------------------------------

def get_client_config():

    return {
        "web": {
            "client_id": st.secrets["google"]["client_id"],
            "client_secret": st.secrets["google"]["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [
                st.secrets["google"]["redirect_uri"]
            ]
        }
    }


def create_flow(state=None):

    return Flow.from_client_config(
        get_client_config(),
        scopes=SCOPES,
        redirect_uri=st.secrets["google"]["redirect_uri"],
        state=state
    )


# ---------------------------------------
# OAuth state persistence
# ---------------------------------------

def save_oauth_state(state):

    init_database()

    with sqlite3.connect(DB_PATH) as conn:

        # Remove stale pending states older than 15 minutes.
        conn.execute("""
            DELETE FROM oauth_states
            WHERE created_at < datetime('now', '-15 minutes')
        """)

        conn.execute("""
            INSERT INTO oauth_states (state)
            VALUES (?)
        """, (state,))


def consume_oauth_state(state):

    if not state:
        return False

    init_database()

    with sqlite3.connect(DB_PATH) as conn:

        result = conn.execute("""
            SELECT state FROM oauth_states
            WHERE state = ?
        """, (state,)).fetchone()

        if not result:
            return False

        conn.execute("""
            DELETE FROM oauth_states
            WHERE state = ?
        """, (state,))

    return True


# ---------------------------------------
# Generate Google authorization URL
# ---------------------------------------

def get_authorization_url():

    flow = create_flow()

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        prompt="select_account consent",
        include_granted_scopes="true"
    )

    save_oauth_state(state)

    return authorization_url, state


# ---------------------------------------
# Exchange authorization code
# ---------------------------------------

def exchange_code(code, state):

    if not consume_oauth_state(state):
        raise ValueError("Invalid or expired OAuth state.")

    flow = create_flow(state=state)

    flow.fetch_token(code=code)

    credentials = flow.credentials

    if not credentials.refresh_token:
        raise ValueError(
            "Google did not return a refresh token. "
            "Please reconnect and grant consent."
        )

    return credentials


# ---------------------------------------
# Save encrypted credentials
# ---------------------------------------

def save_credentials(email, credentials):

    init_database()

    token_json = credentials.to_json().encode("utf-8")

    encrypted_token = get_fernet().encrypt(token_json)

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            INSERT OR REPLACE INTO gmail_tokens
            (email, encrypted_token)
            VALUES (?, ?)
        """, (email, encrypted_token))


# ---------------------------------------
# Load credentials
# ---------------------------------------

def load_credentials(email):

    init_database()

    with sqlite3.connect(DB_PATH) as conn:

        result = conn.execute("""
            SELECT encrypted_token
            FROM gmail_tokens
            WHERE email = ?
        """, (email,)).fetchone()

    if not result:
        return None

    try:

        decrypted = get_fernet().decrypt(result[0])

        token_data = json.loads(decrypted.decode("utf-8"))

        credentials = Credentials.from_authorized_user_info(
            token_data,
            SCOPES
        )

        if credentials.expired:

            if not credentials.refresh_token:
                return None

            credentials.refresh(Request())

            save_credentials(email, credentials)

        if credentials.valid:
            return credentials

    except Exception as error:

        print(f"Gmail credential error: {error}")

    return None


# ---------------------------------------
# Disconnect Gmail
# ---------------------------------------

def disconnect_gmail(email):

    credentials = load_credentials(email)

    # Revoke Google authorization when possible.
    if credentials:

        token = (
            credentials.refresh_token
            or credentials.token
        )

        if token:

            try:

                response = requests.post(
                    "https://oauth2.googleapis.com/revoke",
                    params={"token": token},
                    timeout=15
                )

                if response.status_code not in (200, 400):
                    print(
                        "Google token revocation response:",
                        response.status_code
                    )

            except requests.RequestException as error:

                print("Token revocation error:", error)

    # Always remove locally stored credentials.
    init_database()

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            DELETE FROM gmail_tokens
            WHERE email = ?
        """, (email,))

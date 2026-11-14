
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


# ==========================================
# DATABASE AND ENCRYPTION
# ==========================================

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
                code_verifier TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate an existing database created by the previous version.
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(oauth_states)"
            ).fetchall()
        }

        if "code_verifier" not in columns:
            conn.execute("""
                ALTER TABLE oauth_states
                ADD COLUMN code_verifier TEXT
            """)


# ==========================================
# OAUTH CONFIGURATION
# ==========================================

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


def create_flow(state=None, code_verifier=None,
                autogenerate_code_verifier=False):

    return Flow.from_client_config(
        get_client_config(),
        scopes=SCOPES,
        redirect_uri=st.secrets["google"]["redirect_uri"],
        state=state,
        code_verifier=code_verifier,
        autogenerate_code_verifier=autogenerate_code_verifier
    )


# ==========================================
# SAVE AND VALIDATE OAUTH STATE
# ==========================================

def save_oauth_state(state, code_verifier):

    init_database()

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            DELETE FROM oauth_states
            WHERE created_at < datetime('now', '-15 minutes')
        """)

        conn.execute("""
            INSERT INTO oauth_states
            (state, code_verifier)
            VALUES (?, ?)
        """, (state, code_verifier))


def consume_oauth_state(state):

    if not state:
        return None

    init_database()

    with sqlite3.connect(DB_PATH) as conn:

        result = conn.execute("""
            SELECT code_verifier
            FROM oauth_states
            WHERE state = ?
            AND created_at >= datetime('now', '-15 minutes')
        """, (state,)).fetchone()

        if not result:
            return None

        conn.execute("""
            DELETE FROM oauth_states
            WHERE state = ?
        """, (state,))

    return result[0]


# ==========================================
# GENERATE GOOGLE AUTHORIZATION URL
# ==========================================

def get_authorization_url():

    # Generate PKCE verifier before redirecting.
    flow = create_flow(
        autogenerate_code_verifier=True
    )

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        prompt="select_account consent",
        include_granted_scopes="true"
    )

    # Preserve verifier across Streamlit reruns.
    save_oauth_state(
        state,
        flow.code_verifier
    )

    return authorization_url, state


# ==========================================
# EXCHANGE AUTHORIZATION CODE
# ==========================================

def exchange_code(code, state):

    # Retrieve the original verifier.
    code_verifier = consume_oauth_state(state)

    if not code_verifier:
        raise ValueError(
            "OAuth code verifier is missing or expired. "
            "Please click Connect Gmail and try again."
        )

    # Recreate the flow with the original verifier.
    flow = create_flow(
        state=state,
        code_verifier=code_verifier
    )

    flow.fetch_token(code=code)

    credentials = flow.credentials

    if not credentials.refresh_token:
        raise ValueError(
            "Google did not return a refresh token. "
            "Please reconnect and grant consent."
        )

    return credentials


# ==========================================
# SAVE ENCRYPTED CREDENTIALS
# ==========================================

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


# ==========================================
# LOAD AND REFRESH CREDENTIALS
# ==========================================

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

        token_data = json.loads(
            decrypted.decode("utf-8")
        )

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

        print("Gmail credential error:", error)

    return None


# ==========================================
# DISCONNECT GMAIL
# ==========================================

def disconnect_gmail(email):

    credentials = load_credentials(email)

    if credentials:

        token = (
            credentials.refresh_token
            or credentials.token
        )

        if token:

            try:

                requests.post(
                    "https://oauth2.googleapis.com/revoke",
                    params={"token": token},
                    timeout=15
                )

            except requests.RequestException as error:

                print("Token revocation error:", error)

    init_database()

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            DELETE FROM gmail_tokens
            WHERE email = ?
        """, (email,))

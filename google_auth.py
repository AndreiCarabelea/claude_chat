import base64
import hashlib
import hmac
import json
import os
import secrets
import time

import streamlit as st
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
APP_URL = os.getenv("APP_URL", "http://localhost:8501")
OAUTH_STATE_SECRET = os.getenv(
    "OAUTH_STATE_SECRET",
    GOOGLE_CLIENT_SECRET or "change-me-in-production",
)

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

SESSION_EMAIL_KEY = "authenticated_email"
SESSION_NAME_KEY = "authenticated_name"


def _redirect_uri():
    return APP_URL.rstrip("/") + "/"


def _oauth_configured():
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def _make_signed_state():
    payload = {
        "nonce": secrets.token_urlsafe(16),
        "ts": int(time.time()),
    }
    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8")
    signature = hmac.new(
        OAUTH_STATE_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload_b64}.{signature}"


def _verify_signed_state(state, max_age_seconds=900):
    if not state or "." not in state:
        return False

    payload_b64, signature = state.rsplit(".", 1)
    expected = hmac.new(
        OAUTH_STATE_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return False

    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("utf-8")))
    except Exception:
        return False

    ts = int(payload.get("ts", 0))
    return (time.time() - ts) <= max_age_seconds


def _build_flow():
    return Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [_redirect_uri()],
            }
        },
        scopes=SCOPES,
        redirect_uri=_redirect_uri(),
    )


def _clear_oauth_query_params():
    for key in ("code", "state", "scope", "authuser", "prompt", "hd"):
        if key in st.query_params:
            del st.query_params[key]


def _extract_user_from_credentials(credentials):
    if credentials.id_token:
        id_info = id_token.verify_oauth2_token(
            credentials.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
        email = (id_info.get("email") or "").strip().lower()
        name = id_info.get("name") or email
        if not email or not id_info.get("email_verified", False):
            raise ValueError("Google account email is not verified.")
        return email, name

    token = credentials.token
    if not token:
        raise ValueError("Missing OAuth access token.")

    import requests

    response = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    response.raise_for_status()
    profile = response.json()
    email = (profile.get("email") or "").strip().lower()
    name = profile.get("name") or email
    if not email or not profile.get("email_verified", False):
        raise ValueError("Google account email is not verified.")
    return email, name


def _handle_oauth_callback():
    code = st.query_params.get("code")
    state = st.query_params.get("state")
    if not code:
        return False

    if not _verify_signed_state(state):
        st.error("Sign-in failed: invalid or expired OAuth state. Please try again.")
        _clear_oauth_query_params()
        return True

    try:
        flow = _build_flow()
        flow.fetch_token(code=code)
        email, name = _extract_user_from_credentials(flow.credentials)
        st.session_state[SESSION_EMAIL_KEY] = email
        st.session_state[SESSION_NAME_KEY] = name
        _clear_oauth_query_params()
        st.rerun()
    except Exception as exc:
        st.error(f"Sign-in failed: {exc}")
        _clear_oauth_query_params()
    return True


def get_authenticated_user():
    if SESSION_EMAIL_KEY in st.session_state:
        return st.session_state[SESSION_EMAIL_KEY]

    if _handle_oauth_callback():
        return st.session_state.get(SESSION_EMAIL_KEY)

    return None


def get_authenticated_name():
    return st.session_state.get(SESSION_NAME_KEY)


def logout_user():
    for key in (SESSION_EMAIL_KEY, SESSION_NAME_KEY, "donation_clicked"):
        st.session_state.pop(key, None)
    _clear_oauth_query_params()


def build_google_sign_in_url():
    flow = _build_flow()
    state = _make_signed_state()
    auth_url, _ = flow.authorization_url(
        access_type="online",
        include_granted_scopes="true",
        prompt="select_account",
        state=state,
    )
    return auth_url


def render_google_login():
    st.subheader("Sign in required")
    st.write("Sign in with your Google account to use the Academic Learning Assistant.")

    if not _oauth_configured():
        st.error(
            "Google OAuth is not configured. Set GOOGLE_CLIENT_ID, "
            "GOOGLE_CLIENT_SECRET, and APP_URL in your environment."
        )
        return

    auth_url = build_google_sign_in_url()
    st.link_button("Sign in with Google", auth_url, type="primary")

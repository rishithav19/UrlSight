import html
import json
import math
import re

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import pandas as pd
import streamlit as st

from src.screenshot import capture_screenshot
from src.analysis import analyze_url
from src.ai_explainer import generate_explanation
from url_detector.src.predict_url import PhishingPredictor

from services.gmail_auth import (
    get_authorization_url,
    exchange_code,
    save_credentials,
    load_credentials,
    disconnect_gmail,
)
from services.gmail_service import (
    get_gmail_profile,
    fetch_emails,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
URL_DETECTOR_DIR = PROJECT_DIR / "url_detector"

st.set_page_config(
    page_title="URLSight | URL Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }
    #MainMenu, footer { visibility: hidden; }

    /* Hero */
    .hero {
        padding: 34px 36px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
        color: #fff;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px; color: #fff; }
    .hero p  { margin: 6px 0 0 0; font-size: 1.1rem; color: #cbd5e1; }
    .hero .pill {
        display: inline-block; margin-top: 16px; margin-right: 8px;
        padding: 4px 12px; border-radius: 999px; font-size: 0.78rem;
        background: rgba(255,255,255,0.14); color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Verdict card */
    .verdict {
        border-radius: 18px; padding: 24px 28px; margin: 6px 0 18px 0;
        border: 1px solid; display: flex; align-items: center; gap: 20px;
    }
    .verdict.bad  { background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.45); }
    .verdict.good { background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.45); }
    .verdict .icon { font-size: 3rem; line-height: 1; }
    .verdict .label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1.2px; opacity: 0.7; }
    .verdict .title { font-size: 1.8rem; font-weight: 800; margin: 2px 0; }
    .verdict.bad  .title { color: #ef4444; }
    .verdict.good .title { color: #16a34a; }
    .verdict .url {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.9rem; opacity: 0.85; word-break: break-all;
    }

    /* Score meter */
    .meter-wrap { margin: 4px 0 6px 0; }
    .meter {
        position: relative; height: 12px; border-radius: 999px;
        background: linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%);
    }
    .meter .marker {
        position: absolute; top: -5px; width: 6px; height: 22px;
        background: #0f172a; border: 2px solid #fff; border-radius: 4px;
        transform: translateX(-50%); box-shadow: 0 1px 4px rgba(0,0,0,0.4);
    }
    .meter-labels { display: flex; justify-content: space-between; font-size: 0.75rem; opacity: 0.65; margin-top: 8px; }

    /* Stat cards */
    .stat {
        border: 1px solid rgba(148,163,184,0.35); border-radius: 14px;
        padding: 14px 16px; background: rgba(148,163,184,0.07); height: 100%;
    }
    .stat .k { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; opacity: 0.65; }
    .stat .v { font-size: 1.5rem; font-weight: 700; margin-top: 2px; word-break: break-all; }

    /* Indicator rows */
    .ind {
        display: flex; gap: 10px; align-items: flex-start;
        padding: 10px 14px; margin-bottom: 8px; border-radius: 10px;
        border-left: 4px solid; background: rgba(148,163,184,0.07);
    }
    .ind.warn { border-color: #f59e0b; }
    .ind.ok   { border-color: #22c55e; }

    /* Section headings */
    .section-title { font-size: 1.15rem; font-weight: 700; margin: 8px 0 10px 0; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 18px; border-radius: 10px 10px 0 0; font-weight: 600; }

    /* Primary button */
    div[data-testid="stForm"] { border: none; padding: 0; }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 12px; font-weight: 700; height: 3rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------
# Gmail OAuth callback and session state
# --------------------------------------------------

for state_key, default_value in {
    "gmail_email": None,
    "gmail_emails": [],
    "gmail_extracted_urls": [],
}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value

# Google redirects back to the configured redirect URI with ?code=...&state=...
oauth_error = st.query_params.get("error")
oauth_code = st.query_params.get("code")
oauth_returned_state = st.query_params.get("state")

if oauth_error:
    st.error(f"Google Gmail authorization failed: {oauth_error}")
    st.query_params.clear()

elif oauth_code:
    try:
        with st.spinner("Completing Gmail connection..."):
            gmail_credentials = exchange_code(
                oauth_code,
                oauth_returned_state,
            )
            gmail_profile = get_gmail_profile(gmail_credentials)
            connected_email = gmail_profile["emailAddress"]

            save_credentials(connected_email, gmail_credentials)

            st.session_state["gmail_email"] = connected_email
            st.session_state["gmail_emails"] = []
            st.session_state["gmail_extracted_urls"] = []

        st.query_params.clear()
        st.success(f"Connected Gmail account: {connected_email}")
        st.rerun()

    except Exception as exc:
        st.query_params.clear()
        st.error(f"Could not complete Gmail connection: {exc}")


class _EmailLinkParser(HTMLParser):
    """Collect href attributes from HTML email content."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            for name, value in attrs:
                if name.lower() == "href" and value:
                    self.links.append(value)


def extract_urls_from_email(email_record):
    """Extract HTTP(S) links from an email's body and snippet."""
    candidates = []

    body = email_record.get("body", "") or ""
    snippet = email_record.get("snippet", "") or ""

    # Extract links from HTML href attributes.
    parser = _EmailLinkParser()
    try:
        parser.feed(body)
        candidates.extend(parser.links)
    except Exception:
        pass

    # Also find URLs written directly in plain text or HTML.
    combined_text = html.unescape(f"{body}\n{snippet}")
    candidates.extend(
        re.findall(r"""https?://[^\s<>"'`]+""", combined_text, flags=re.IGNORECASE)
    )

    unique_urls = []
    seen = set()

    for candidate in candidates:
        candidate = html.unescape(str(candidate)).strip()
        candidate = candidate.rstrip(".,;:!?)]}")

        try:
            parsed = urlparse(candidate)
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
                continue
        except ValueError:
            continue

        if candidate not in seen:
            seen.add(candidate)
            unique_urls.append(candidate)

    return unique_urls


# --------------------------------------------------
# Load the existing saved model
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the frozen Hybrid URL detector once per Streamlit session."""
    if not URL_DETECTOR_DIR.exists():
        raise FileNotFoundError(
            f"Hybrid URL detector directory not found: {URL_DETECTOR_DIR}"
        )
    return PhishingPredictor(project_dir=str(URL_DETECTOR_DIR), device="cpu")


# --------------------------------------------------
# URL validation
# --------------------------------------------------

def validate_url(raw_url):
    raw_url = raw_url.strip()

    if not raw_url:
        return None, "Enter a URL to analyze."

    if any(ord(ch) < 32 for ch in raw_url):
        return None, "The URL contains invalid control characters."

    candidate = (
        raw_url if "://" in raw_url else "https://" + raw_url
    )

    parsed = urlparse(candidate)

    if parsed.scheme.lower() not in {"http", "https"}:
        return None, "Only HTTP and HTTPS URLs are supported."

    if not parsed.hostname:
        return None, "Enter a valid URL with a hostname."

    if parsed.username or parsed.password:
        return None, "URLs containing embedded credentials are not supported."

    try:
        parsed.port
    except ValueError:
        return None, "The URL contains an invalid port."

    return candidate, None


# --------------------------------------------------
# Stage C  URL prediction
# --------------------------------------------------

def predict_url(url, predictor):
    """Run Hybrid URL prediction and normalize app labels to 1=phishing, 0=legitimate.

    Hybrid URL model's native convention is 0=phishing and 1=legitimate. URLSight's
    presentation layer historically uses 1=phishing, so normalize it here.
    The negated SVM margin is a signed decision score, NOT a probability.
    """
    result = predictor.predict(url)
    features = result["raw_features"]
    prediction = 1 if result["is_phishing"] else 0
    score = -float(result["decision_margin"])
    return features, prediction, score


# --------------------------------------------------
# UI helpers
# --------------------------------------------------

def stat_card(label, value):
    return (
        f'<div class="stat"><div class="k">{html.escape(str(label))}</div>'
        f'<div class="v">{html.escape(str(value))}</div></div>'
    )


def score_meter(score):
    # Map the unbounded SVM decision score onto 0-100% for display only.
    position = 50 + 50 * math.tanh(score)
    position = max(2, min(98, position))

    return f"""
    <div class="meter-wrap">
        <div class="meter"><div class="marker" style="left:{position:.1f}%"></div></div>
        <div class="meter-labels">
            <span>Legitimate side</span><span>Decision boundary (0)</span><span>Phishing side</span>
        </div>
    </div>
    """


def fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🛡️ URLSight")
    st.caption("Research prototype for phishing URL detection.")

    st.markdown("### How it works")
    st.markdown(
        """
        1. **Validate** the URL you enter
        2. **Classify** it with the Hybrid deep-hybrid URL model
        3. **Explain** it with rules and AI
        4. **Preview** the page via screenshot
        """
    )


    if st.session_state.get("analysis_result"):
        if st.button("🧹 Clear results", use_container_width=True):
            for key in [
                "analysis_result",
                "automatic_result_url",
                "automatic_explanation",
                "automatic_ai_error",
                "automatic_screenshot",
                "automatic_screenshot_error",
            ]:
                st.session_state.pop(key, None)
            st.rerun()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ URLSight</h1>
        <p>Hybrid-ready phishing analysis with URL intelligence and webpage preview.</p>
        <span class="pill"> Hybrid URL model</span>
        <span class="pill"> AI explanation</span>
        <span class="pill"> Screenshot preview</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

try:
    predictor = load_model()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()


# --------------------------------------------------
# URL form
# --------------------------------------------------

with st.form("url_analysis_form"):
    input_col, button_col = st.columns([5, 1.3], vertical_alignment="bottom")

    with input_col:
        entered_url = st.text_input(
            "Website URL",
            placeholder="https://example.com/login",
        )

    with button_col:
        submitted = st.form_submit_button(
            "🔍 Analyze",
            use_container_width=True,
            type="primary",
        )


# --------------------------------------------------
# Gmail Integration and Link Scanner
# --------------------------------------------------

with st.expander("📧 Gmail integration – connect, switch accounts and scan links", expanded=False):
    st.write(
        "Connect a Gmail account using Google OAuth, fetch recent inbox emails, "
        "and extract links for URLSight analysis."
    )
    st.caption(
        "URLSight requests Gmail read-only access. It does not send, delete, "
        "or modify your emails."
    )

    connected_email = st.session_state.get("gmail_email")

    if not connected_email:
        st.info("No Gmail account is currently connected.")

        if st.button(
            "Connect Gmail",
            key="gmail_connect_button",
            type="primary",
        ):
            try:
                authorization_url, _state = get_authorization_url()
                st.link_button(
                    "Continue with Google",
                    authorization_url,
                    type="primary",
                    use_container_width=True,
                )
                st.caption(
                    "Select the Google account you want URLSight to use, "
                    "then grant read-only Gmail access."
                )
            except Exception as exc:
                st.error(f"Could not start Google sign-in: {exc}")

    else:
        gmail_credentials = load_credentials(connected_email)

        if not gmail_credentials:
            st.warning(
                f"The saved Gmail authorization for {connected_email} is "
                "expired or invalid. Disconnect it and connect again."
            )

            if st.button(
                "Remove connection",
                key="gmail_remove_invalid_button",
            ):
                disconnect_gmail(connected_email)
                st.session_state["gmail_email"] = None
                st.session_state["gmail_emails"] = []
                st.session_state["gmail_extracted_urls"] = []
                st.session_state.pop("gmail_selected_url", None)
                st.rerun()

        else:
            status_col, disconnect_col = st.columns([3, 1])

            with status_col:
                st.success(f"Connected account: {connected_email}")

            with disconnect_col:
                if st.button(
                    "Disconnect",
                    key="gmail_disconnect_button",
                    use_container_width=True,
                ):
                    try:
                        disconnect_gmail(connected_email)
                    except Exception as exc:
                        st.warning(
                            "The local connection will be cleared, but Google "
                            f"token revocation reported an issue: {exc}"
                        )

                    st.session_state["gmail_email"] = None
                    st.session_state["gmail_emails"] = []
                    st.session_state["gmail_extracted_urls"] = []
                    st.session_state.pop("gmail_selected_url", None)
                    st.rerun()

            st.caption(
                "To use another Gmail account, disconnect this account first "
                "and then select Connect Gmail."
            )

            fetch_col, count_col = st.columns([2, 1])

            with count_col:
                email_limit = st.number_input(
                    "Number of emails to fetch",
                    min_value=1,
                    max_value=500,
                    value=5,
                    step=1,
                    help="Enter how many recent inbox emails URLSight should scan (1–500).",
                    key="gmail_email_limit",
                )

            with fetch_col:
                if st.button(
                    "Fetch inbox emails and extract links",
                    key="gmail_fetch_button",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        with st.spinner("Fetching recent Gmail messages..."):
                            current_credentials = load_credentials(connected_email)

                            if not current_credentials:
                                raise RuntimeError(
                                    "Gmail authorization expired. Disconnect "
                                    "and reconnect your account."
                                )

                            fetched_emails = fetch_emails(
                                current_credentials,
                                max_results=email_limit,
                            )

                            extracted_urls = []
                            seen_urls = set()

                            for email_record in fetched_emails:
                                for extracted_url in extract_urls_from_email(email_record):
                                    if extracted_url not in seen_urls:
                                        seen_urls.add(extracted_url)
                                        extracted_urls.append(extracted_url)

                            st.session_state["gmail_emails"] = fetched_emails
                            st.session_state["gmail_extracted_urls"] = extracted_urls
                            st.session_state.pop("gmail_selected_url", None)

                        st.success(
                            f"Fetched {len(fetched_emails)} email(s) and found "
                            f"{len(extracted_urls)} unique URL(s)."
                        )

                    except Exception as exc:
                        st.error(f"Gmail scan failed: {exc}")

            fetched_emails = st.session_state.get("gmail_emails", [])
            gmail_urls = st.session_state.get("gmail_extracted_urls", [])

            if fetched_emails:
                st.caption(f"Fetched {len(fetched_emails)} email(s).")

                with st.expander("View fetched email summaries"):
                    for index, email_record in enumerate(fetched_emails, start=1):
                        subject = email_record.get("subject") or "No subject"
                        with st.expander(f"{index}. {subject}"):
                            st.write("**From:**", email_record.get("sender", "Unknown"))
                            st.write("**Date:**", email_record.get("date", ""))
                            st.write("**Preview:**", email_record.get("snippet", ""))

            if gmail_urls:
                st.success(f"Found {len(gmail_urls)} unique URL(s) in fetched emails.")

                selected_gmail_url = st.selectbox(
                    "Choose a URL extracted from Gmail",
                    options=gmail_urls,
                    key="gmail_selected_url",
                )

                if st.button(
                    "Analyze selected Gmail URL",
                    key="gmail_analyze_button",
                ):
                    normalized_url, validation_error = validate_url(selected_gmail_url)

                    if validation_error:
                        st.error(validation_error)
                    else:
                        try:
                            with st.spinner("Running Hybrid URL prediction..."):
                                gmail_features, gmail_prediction, gmail_score = predict_url(
                                    normalized_url,
                                    predictor,
                                )

                            st.session_state["analysis_result"] = {
                                "url": normalized_url,
                                "features": gmail_features,
                                "prediction": gmail_prediction,
                                "score": gmail_score,
                            }

                            # Clear the previous URL's explanation and screenshot
                            # so the normal results workflow runs for this URL.
                            for state_key in [
                                "automatic_result_url",
                                "automatic_explanation",
                                "automatic_ai_error",
                                "automatic_screenshot",
                                "automatic_screenshot_error",
                            ]:
                                st.session_state.pop(state_key, None)

                            st.success(
                                "URL selected. Its prediction, explanation "
                                "and screenshot will appear below."
                            )

                        except Exception as exc:
                            st.error(
                                f"Could not analyze the selected Gmail URL: {exc}"
                            )

            elif fetched_emails:
                st.info(
                    "No HTTP or HTTPS links were found in the fetched emails."
                )

# --------------------------------------------------
# Run prediction first
# --------------------------------------------------

if submitted:
    normalized_url, error = validate_url(entered_url)

    if error:
        st.error(error)

    else:
        try:
            with st.spinner("Running Hybrid URL prediction..."):
                features, prediction, score = predict_url(
                    normalized_url, predictor
                )

            # A fresh result starts a fresh automatic workflow.
            st.session_state["analysis_result"] = {
                "url": normalized_url,
                "features": features,
                "prediction": prediction,
                "score": score,
            }

            st.session_state.pop("automatic_result_url", None)
            st.session_state.pop("automatic_explanation", None)
            st.session_state.pop("automatic_ai_error", None)
            st.session_state.pop("automatic_screenshot", None)
            st.session_state.pop("automatic_screenshot_error", None)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# Display prediction and supporting analysis
# --------------------------------------------------

result = st.session_state.get("analysis_result")

if result:
    url = result["url"]
    features = result["features"]
    prediction = result["prediction"]
    score = result["score"]

    is_phishing = prediction == 1

    # ---------------- Rule-based analysis ----------------

    analysis = analyze_url(
        url,
        features,
        prediction,
        score,
    )

    # ---------------- AI explanation + screenshot ----------------
    # Run both tasks concurrently. Keep the prediction banner hidden
    # until both tasks finish and at least 2 seconds have elapsed.

    completed_url = st.session_state.get("automatic_result_url")

    if completed_url != url:
        explanation = None
        ai_error = None
        screenshot_bytes = None
        screenshot_error = None

        start_time = time.monotonic()

        with st.status(
            "Preparing URLSight security analysis...",
            expanded=True,
        ) as status:
            status.write("✔️ URL model prediction completed.")
            status.write("⏳ Running AI analysis and webpage screenshot in parallel...")

            with ThreadPoolExecutor(max_workers=2) as executor:
                ai_future = executor.submit(
                    generate_explanation,
                    url=result["url"],
                    features=result["features"],
                    prediction=(
                        "PHISHING"
                        if result["prediction"] == 1
                        else "LEGITIMATE"
                    ),
                    decision_score=result["score"],
                )

                screenshot_future = executor.submit(
                    capture_screenshot,
                    url,
                )

                # Collect AI result. Screenshot runs concurrently.
                try:
                    explanation = ai_future.result()
                    status.write("✔️ AI explanation generated.")
                except Exception as exc:
                    ai_error = str(exc)
                    status.write("⚠️ AI explanation unavailable.")

                # Collect screenshot result. AI has already been running.
                try:
                    screenshot_bytes = screenshot_future.result()
                    status.write("✔️ Webpage screenshot captured.")
                except Exception as exc:
                    screenshot_error = str(exc)
                    status.write("⚠️ Screenshot unavailable.")

            # Enforce a minimum 2-second preparation period.
            elapsed = time.monotonic() - start_time
            remaining = max(0.0, 2.0 - elapsed)

            if remaining > 0:
                status.write("⏳ Finalizing security assessment...")
                time.sleep(remaining)

            status.update(
                label="URLSight analysis finished",
                state="complete",
                expanded=False,
            )

        # Cache results so Streamlit reruns don't repeat API/screenshot work.
        st.session_state["automatic_explanation"] = explanation
        st.session_state["automatic_ai_error"] = ai_error
        st.session_state["automatic_screenshot"] = screenshot_bytes
        st.session_state["automatic_screenshot_error"] = screenshot_error
        st.session_state["automatic_result_url"] = url

    explanation = st.session_state.get("automatic_explanation")
    ai_error = st.session_state.get("automatic_ai_error")
    screenshot_bytes = st.session_state.get("automatic_screenshot")
    screenshot_error = st.session_state.get("automatic_screenshot_error")

    # ---------------- Show prediction only after preparation ----------------

    st.markdown(
        f"""
        <div class="verdict {'bad' if is_phishing else 'good'}">
            <div class="icon">{'🚨' if is_phishing else '✅'}</div>
            <div>
                <div class="label">Hybrid URL prediction</div>
                <div class="title">{'PHISHING' if is_phishing else 'LEGITIMATE'}</div>
                <div class="url">{html.escape(url)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Key stats + meter ----------------

    stats = [
        ("Signed decision score", f"{score:.4f}"),
        ("URL length", features.get("url_length", "-")),
        ("HTTPS", "Yes" if features.get("https_flag") else "No"),
        ("Subdomains", features.get("subdomain_count", "-")),
        ("URL entropy", fmt(features.get("url_entropy", "-"))),
    ]

    stat_cols = st.columns(len(stats))
    for col, (label, value) in zip(stat_cols, stats):
        with col:
            st.markdown(stat_card(label, value), unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(score_meter(score), unsafe_allow_html=True)

    st.caption(
        "The signed SVM decision score is not a probability. Positive values "
        "favor phishing; negative values favor legitimate. The marker is a "
        "visual aid only and is not a calibrated risk percentage."
    )

    # ---------------- Tabs ----------------

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    tab_overview, tab_features, tab_ai, tab_shot = st.tabs(
        [
            "📋 Overview",
            "🔬 URL Features",
            "🧠 AI Explanation",
            "📸 Screenshot",
        ]
    )

    # ---- Overview (rule-based) ----
    with tab_overview:
        st.markdown(
            '<div class="section-title">Analysis</div>',
            unsafe_allow_html=True,
        )
        st.write(analysis["verdict"])

        left, right = st.columns(2)

        with left:
            st.markdown(
                '<div class="section-title">URL indicators</div>',
                unsafe_allow_html=True,
            )

            if analysis["indicators"]:
                for indicator in analysis["indicators"]:
                    st.markdown(
                        f'<div class="ind warn">{html.escape(str(indicator))}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No additional indicators were identified by the rules.")

        with right:
            st.markdown(
                '<div class="section-title">Other observations</div>',
                unsafe_allow_html=True,
            )

            if analysis["reassuring_signals"]:
                for signal in analysis["reassuring_signals"]:
                    st.markdown(
                        f'<div class="ind ok">{html.escape(str(signal))}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No reassuring signals were identified.")

        st.markdown(
            '<div class="section-title"> </div>',
            unsafe_allow_html=True,
        )
        if is_phishing:
            st.warning(analysis["recommendation"])

    # ---- Features ----
    with tab_features:
        feature_descriptions = {
            "url_length": "Total URL length (characters)",
            "has_ip_address": "Uses an IP address as host",
            "dot_count": "Number of dots in URL",
            "https_flag": "Uses HTTPS",
            "url_entropy": "Character entropy",
            "token_count": "Number of URL tokens",
            "subdomain_count": "Number of subdomains",
            "query_param_count": "Query parameter count (dataset convention)",
            "tld_length": "Top-level domain length",
            "path_length": "URL path length",
            "has_hyphen_in_domain": "Domain contains a hyphen",
            "number_of_digits": "Number of numeric digits",
            "tld_popularity": "TLD in the model's common-TLD set",
            "suspicious_file_extension": "Suspicious file extension flag",
            "domain_name_length": "Domain label length",
            "percentage_numeric_chars": "Percentage of numeric characters",
        }

        display_features = {
            feature_descriptions.get(key, key): value
            for key, value in features.items()
        }

        feature_df = pd.DataFrame(
            [(k, str(v)) for k, v in display_features.items()],
            columns=["Feature", "Value"],
        )

        search = st.text_input(
            "Filter features",
            placeholder="e.g. domain, count, ratio",
            label_visibility="collapsed",
        )

        if search:
            feature_df = feature_df[
                feature_df["Feature"].str.contains(search, case=False, na=False)
            ]

        st.dataframe(
            feature_df,
            hide_index=True,
            use_container_width=True,
            height=420,
        )

        report = {
            "url": url,
            "prediction": "PHISHING" if is_phishing else "LEGITIMATE",
            "decision_score": score,
            "features": features,
            "rule_based_analysis": analysis,
        }

        st.download_button(
            "⬇️ Download analysis (JSON)",
            data=json.dumps(report, indent=2, default=str),
            file_name="urlsight_analysis.json",
            mime="application/json",
        )

    # ---- AI explanation ----
    with tab_ai:
        if explanation:
            st.markdown(explanation)
        else:
            st.warning(
                "The AI explanation could not be generated. "
                "Hybrid prediction and analysis is still available."
            )

            if ai_error:
                with st.expander("AI error details"):
                    st.code(ai_error)

        st.caption(
            "The AI explains the Hybrid URL-model result. It does not independently "
            "verify the website or guarantee that the prediction is correct."
        )

    # ---- Screenshot ----
    with tab_shot:
        if screenshot_bytes:
            st.image(
                screenshot_bytes,
                use_container_width=True,
            )
        else:
            st.warning("A webpage screenshot could not be captured.")

            if screenshot_error:
                with st.expander("Screenshot error details"):
                    st.code(screenshot_error)

        st.caption(
            "A screenshot is not proof that a page is safe or malicious."
        )

else:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            stat_card("Step 1", "Enter a URL"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            stat_card("Step 2", "Hybrid prediction"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            stat_card("Step 3", "Explanation and preview"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.info(
        "Enter a URL and click Analyze. URLSight will predict first, "
        "then attempt the AI explanation and screenshot automatically."
    )


# ---------------------------------- ----------------
# Footer
# --------------------------------------------------

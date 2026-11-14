# URLSight – Stage C URL detector integration

This version integrates the Stage C URL-only inference pipeline from `projsem` into URLSight.

## What changed

- Gmail scanning, URL extraction, screenshot capture, and AI explanation flow are retained.
- The previous URLSight 24-feature Linear SVM inference path is replaced by the Stage C pipeline.
- Stage C's native labels are `0 = phishing`, `1 = legitimate`. The Streamlit presentation layer normalizes them to `1 = phishing`, `0 = legitimate`.
- The displayed signed score is `-decision_margin`, so positive values favor phishing and negative values favor legitimate. It is not a probability.
- The screenshot is still a preview only. No visual classifier is integrated in this phase.
- Only the seven artifacts required for Stage C inference are included. Training CSVs, latent feature matrices, GAN artifacts, and training scripts are not needed for inference.

## Setup

Use Python 3.10 or 3.11 if available, create a fresh virtual environment, and install:

```bash
python -m venv .venv
# Linux / WSL
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
```

The serialized Stage C scikit-learn artifacts report version 1.9.0. Keep `scikit-learn==1.9.0` to avoid cross-version pickle compatibility warnings or behavior changes.

## Run a URL-only smoke test

```bash
python url_detector/src/predict_url.py --json "https://www.google.com"
python url_detector/src/predict_url.py --json "http://192.168.1.10/login.exe"
```

## Configure Gmail and AI providers

1. Copy `.env.example` to `.env` and add newly rotated Groq and/or Gemini API keys.
2. Download your own Google OAuth Desktop client JSON and save it as `credentials/client_secret.json`.
3. Run `python auth_gmail.py` and complete the sign-in flow. This creates a local `token.json`.
4. Start the app with `streamlit run app.py`.

Never commit `.env`, `token.json`, or `credentials/client_secret.json`. The credentials that were present in the supplied archive have deliberately not been copied into this integration package; rotate/revoke those credentials before reuse.

## Important validation note

The supplied README reports high benchmark metrics, but these must not be treated as proof of generalization. A small smoke test showed that one PayPal-lookalike test URL was classified as legitimate. Evaluate the model on a separate, recent, domain-disjoint set of benign and phishing URLs before relying on it. The SVM decision margin is not a calibrated probability.

## Current scope

The Stage C URL detector is integrated. Screenshot capture remains an independent preview; the visual branch and URL/visual fusion are intentionally deferred until a separately trained screenshot model is ready.

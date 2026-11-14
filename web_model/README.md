# URLSight Web Scraper

A standalone starter module that renders a public webpage with Playwright,
saves its HTML and viewport screenshot, and extracts basic DOM features.

## Install in your existing WSL virtual environment

From this folder:

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

If Chromium reports missing Ubuntu libraries:

```bash
python -m playwright install-deps chromium
```

## Run

From the directory containing the `webscraper` package:

```bash
python -m webscraper.cli https://example.com --name example
```

Outputs are written to `webscraper_output/example/`:
- `page.html`
- `screenshot.png`

The CLI also prints JSON containing page metadata and extracted features.

## Use from Python

```python
from webscraper import WebScraper

result = WebScraper(output_dir="webscraper_output").scrape(
    "https://example.com", sample_name="example"
)
print(result.features)
```

## Extracted features

Form count, password-field count, hidden-input count, iframe count, anchor
count, external-link count, script count, external-script count, image count,
external-image count, insecure/empty form actions, title and approximate text
length. These are observations, not a phishing verdict.

## Security limitations

This is a development starter, not a hardened hostile-content sandbox. It
accepts only HTTP(S), rejects destinations resolving to non-public IPs, and
re-checks browser requests/redirects. Application-level DNS checks cannot
fully prevent DNS rebinding or every SSRF/browser edge case. For production,
run browser workers in disposable containers/VMs with network-level egress
rules denying loopback, private, link-local, metadata and internal ranges.
Do not expose credentials, mounted secrets, or internal services to the
browser. Use only URLs you are authorised to analyse; do not submit credentials
or interact with forms.

Keep this module independent from the URL classifier. If scraping fails, the
URL-only prediction should still be available.

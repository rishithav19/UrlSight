# URLSight HTML/DOM phishing model

A separate HTML/DOM branch for URLSight. It loads the Phish360 directory structure, extracts HTML/DOM features, compares Logistic Regression and Linear SVM, evaluates on validation data, saves a model, and evaluates once on the untouched test split.

## Expected data layout
`Phish360/{trainval,test}/<P-or-L sample>/{RAW-HTML/index.html,URL/url.txt,Label/label.txt}`

Labels are derived from folder prefixes: `P` = phishing (1), `L` = legitimate (0). The `Label/label.txt` file contains a brand/category string (e.g. `amex`), not the binary class.

## Install
```bash
pip install -r requirements.txt
```

## Train
Run from this package directory, with your existing URLSight venv activated:
```bash
python -m html_dom.train --dataset-root ../datasets/phish360/Phish360 --output-dir artifacts
```
Adjust `--dataset-root` if needed. It must point to the folder containing `trainval` and `test`.

The script groups validation samples by URL hostname to reduce same-host leakage. Samples with missing/empty HTML are skipped and recorded in `artifacts/skipped_samples.csv`. It does not use screenshots; those are for the separate visual branch.

## Predict from saved HTML
```bash
python -m html_dom.predict --model artifacts/best_html_dom_model.joblib --html path/to/page.html --url https://example.com
```

## Scrape then predict
```bash
python -m webscraper.cli https://example.com --name example
python -m html_dom.predict --model artifacts/best_html_dom_model.joblib --html webscraper_output/example/page.html --url https://example.com
python -m playwright install chromium
```

## Notes
A login form, password field, external link, or iframe is not automatically malicious. Model output is not proof of safety. The starter scraper is not a hardened sandbox; production URL acquisition needs an isolated worker/container and network-level egress controls to prevent SSRF, DNS rebinding, private-network and metadata access. Never submit credentials or interact with forms on pages being analysed.

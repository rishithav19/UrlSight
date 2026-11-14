
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from html_dom.features import extract_dom_features

url = "https://www.google.com"

path = Path("webscraper_output/google/page.html")

html = path.read_text(
    encoding="utf-8",
    errors="replace"
)

soup = BeautifulSoup(html, "html.parser")

dom_features, html_text = extract_dom_features(
    html,
    url
)

print("=" * 65)
print("GOOGLE DOM INSPECTION")
print("=" * 65)

print("\n1. Extracted DOM features")

for name, value in dom_features.items():
    print(f"{name}: {value}")

print("\n2. HTML forms")

forms = soup.find_all("form")

print("Total forms:", len(forms))

for index, form in enumerate(forms, 1):

    action = form.get("action", "")
    method = form.get("method", "get").lower()

    resolved_action = urljoin(url, action)

    print(f"\nForm {index}")
    print("Raw action:", repr(action))
    print("Resolved action:", resolved_action)
    print("Action scheme:", urlparse(resolved_action).scheme)
    print("Action host:", urlparse(resolved_action).hostname)
    print("Method:", method)

    inputs = form.find_all("input")

    print("Inputs:")

    for item in inputs:
        print(
            "  Type:", item.get("type", "text"),
            "| Name:", item.get("name"),
            "| Hidden:", item.get("type") == "hidden"
        )

print("\n3. Mailto links")

mailto_links = soup.find_all(
    "a",
    href=lambda value: value and value.lower().startswith("mailto:")
)

print("Total mailto links:", len(mailto_links))

for link in mailto_links[:10]:
    print(link.get("href"))

print("\n4. Anchor statistics")

anchors = soup.find_all("a", href=True)

print("Total anchors:", len(anchors))

hosts = set()

for anchor in anchors:

    resolved = urljoin(url, anchor["href"])
    hostname = urlparse(resolved).hostname

    if hostname:
        hosts.add(hostname.lower())

print("Unique resolved anchor hosts:", len(hosts))
print("Sample hosts:")

for host in sorted(hosts)[:20]:
    print(" ", host)

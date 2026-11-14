from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

def extract_html_features(html: str, page_url: str = "") -> dict:
    """Extract basic DOM observations. These are features, not a phishing verdict."""
    soup = BeautifulSoup(html or "", "html.parser")
    page_host = (urlparse(page_url).hostname or "").lower()
    forms = soup.find_all("form")
    links = soup.find_all("a", href=True)
    scripts = soup.find_all("script", src=True)
    images = soup.find_all("img", src=True)
    passwords = soup.find_all("input", attrs={"type": "password"})
    hidden = soup.find_all("input", attrs={"type": "hidden"})
    iframes = soup.find_all("iframe")

    def host(raw):
        try:
            return (urlparse(urljoin(page_url, raw)).hostname or "").lower()
        except ValueError:
            return ""

    external_links = sum(bool(host(x.get("href", ""))) and host(x.get("href", "")) != page_host for x in links)
    external_scripts = sum(bool(host(x.get("src", ""))) and host(x.get("src", "")) != page_host for x in scripts)
    external_images = sum(bool(host(x.get("src", ""))) and host(x.get("src", "")) != page_host for x in images)

    insecure_actions = 0
    empty_actions = 0
    for form in forms:
        action = (form.get("action") or "").strip()
        if not action:
            empty_actions += 1
        elif urlparse(urljoin(page_url, action)).scheme == "http":
            insecure_actions += 1

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text(" ", strip=True)
    return {
        "page_host": page_host, "page_title": title,
        "html_length": len(html or ""), "visible_text_length_approx": len(text),
        "form_count": len(forms), "password_field_count": len(passwords),
        "hidden_input_count": len(hidden), "iframe_count": len(iframes),
        "anchor_count": len(links), "external_link_count": external_links,
        "script_count": len(scripts), "external_script_count": external_scripts,
        "image_count": len(images), "external_image_count": external_images,
        "insecure_form_action_count": insecure_actions,
        "empty_form_action_count": empty_actions, "has_title": bool(title),
        "has_favicon": bool(soup.find("link", rel=lambda v: v and "icon" in str(v).lower())),
    }

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def host(url):
    try:
        return (urlparse(url).hostname or "").lower().strip(".")
    except Exception:
        return ""


def related(a, b):
    """Retain the original project's related-host comparison behavior."""
    return bool(a and b) and (
        a == b or a.endswith("." + b) or b.endswith("." + a)
    )


def extract_dom_features(html, page_url=""):
    """Extract the numerical DOM features used by the existing HTML models.

    Keep feature names and semantics stable for compatibility with V1/V2/V3
    serialized models. New V4 form-level features should be introduced as a
    separate, versioned feature pipeline rather than silently changing this
    model's input schema.
    """
    html = html or ""
    soup = BeautifulSoup(html, "lxml")
    page_host = host(page_url)

    anchors = soup.find_all("a", href=True)
    anchor_urls = [
        urljoin(page_url, (anchor.get("href") or "").strip())
        for anchor in anchors
    ]
    anchor_hosts = [host(item) for item in anchor_urls if host(item)]
    external_anchor_hosts = [
        item for item in anchor_hosts
        if page_host and not related(item, page_host)
    ]

    forms = soup.find_all("form")
    form_actions = [
        urljoin(page_url, (form.get("action") or "").strip())
        for form in forms
    ]
    form_action_hosts = [host(item) for item in form_actions if host(item)]

    inputs = soup.find_all("input")
    password_inputs = soup.find_all(
        "input", attrs={"type": re.compile(r"^password$", re.I)}
    )
    hidden_inputs = soup.find_all(
        "input", attrs={"type": re.compile(r"^hidden$", re.I)}
    )

    scripts = soup.find_all("script")
    images = soup.find_all("img")
    script_hosts = [
        host(urljoin(page_url, script["src"]))
        for script in scripts if script.get("src")
    ]
    image_hosts = [
        host(urljoin(page_url, image["src"]))
        for image in images if image.get("src")
    ]

    def external_count(hosts):
        return sum(
            1 for item in hosts
            if page_host and item and not related(item, page_host)
        )

    features = {
        "html_length": len(html),
        "visible_text_length": len(soup.get_text(" ", strip=True)),
        "title_length": len(soup.title.get_text(" ", strip=True)) if soup.title else 0,
        "has_title": int(soup.title is not None),
        "anchor_count": len(anchors),
        "external_anchor_count": len(external_anchor_hosts),
        "anchor_external_ratio": len(external_anchor_hosts) / max(1, len(anchors)),
        "unique_anchor_hosts": len(set(anchor_hosts)),
        "forms_count": len(forms),
        "password_input_count": len(password_inputs),
        "input_count": len(inputs),
        "hidden_input_count": len(hidden_inputs),
        "iframe_count": len(soup.find_all("iframe")),
        "script_count": len(scripts),
        "external_script_count": external_count(script_hosts),
        "image_count": len(images),
        "external_image_count": external_count(image_hosts),
        "javascript_href_count": sum(
            (anchor.get("href") or "").lower().startswith("javascript:")
            for anchor in anchors
        ),
        "mailto_href_count": sum(
            (anchor.get("href") or "").lower().startswith("mailto:")
            for anchor in anchors
        ),
        "forms_without_action": sum(
            not (form.get("action") or "").strip() for form in forms
        ),
        "form_external_action_count": external_count(form_action_hosts),
        "form_http_action_count": sum(
            urlparse(action).scheme.lower() == "http" for action in form_actions
        ),
        "form_https_action_count": sum(
            urlparse(action).scheme.lower() == "https" for action in form_actions
        ),
        "forms_with_password": sum(
            form.find("input", attrs={"type": re.compile(r"^password$", re.I)}) is not None
            for form in forms
        ),
        "external_form_action_ratio": external_count(form_action_hosts) / max(1, len(forms)),
        "meta_refresh_count": len(
            soup.find_all("meta", attrs={"http-equiv": re.compile(r"^refresh$", re.I)})
        ),
        "button_count": len(soup.find_all("button")),
        "email_input_count": sum(
            "email" in (
                (element.get("type", "") or "") + " " +
                (element.get("name", "") or "") + " " +
                (element.get("id", "") or "")
            ).lower()
            for element in inputs
        ),
        "form_count_with_get": sum(
            (form.get("method") or "get").lower() == "get" for form in forms
        ),
        "form_count_with_post": sum(
            (form.get("method") or "get").lower() == "post" for form in forms
        ),
        "page_url_is_https": int(urlparse(page_url).scheme.lower() == "https"),
        "page_host_length": len(page_host),
        "page_host_looks_like_ip": int(
            bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", page_host))
        ),
    }

    # Preserve the original text-cleaning convention used by the model.
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    visible_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return features, visible_text


def extract_dom_evidence(html, page_url=""):
    """Return readable link and per-form evidence for the Streamlit UI.

    This evidence is for display and inspection. It does not alter the
    numerical feature schema consumed by existing trained models.
    """
    soup = BeautifulSoup(html or "", "lxml")
    page_host = host(page_url)

    links = []
    for anchor in soup.find_all("a", href=True):
        raw_href = (anchor.get("href") or "").strip()
        resolved = urljoin(page_url, raw_href)
        parsed = urlparse(resolved)
        destination_host = host(resolved)

        if parsed.scheme.lower() not in ("http", "https"):
            link_type = "Other"
        elif page_host and destination_host and not related(destination_host, page_host):
            link_type = "External"
        else:
            link_type = "Internal"

        links.append({
            "Link Text": anchor.get_text(" ", strip=True)[:200],
            "Original Href": raw_href,
            "Destination": resolved,
            "Type": link_type,
        })

    forms = []
    for index, form in enumerate(soup.find_all("form"), start=1):
        raw_action = (form.get("action") or "").strip()
        resolved_action = urljoin(page_url, raw_action or page_url)
        parsed_action = urlparse(resolved_action)
        action_host = host(resolved_action)

        # Inspect only controls associated with this form, not every input on the page.
        controls = form.find_all(["input", "select", "textarea", "button"])
        input_types = [
            (element.get("type") or element.name).lower()
            for element in controls
        ]

        has_password = any(
            element.name == "input"
            and (element.get("type") or "").lower() == "password"
            for element in controls
        )
        has_email = any(
            element.name == "input" and (
                (element.get("type") or "").lower() == "email"
                or "email" in (
                    (element.get("name") or "") + " " +
                    (element.get("id") or "")
                ).lower()
            )
            for element in controls
        )

        if parsed_action.scheme.lower() not in ("http", "https"):
            action_type = "Other"
        elif page_host and action_host and not related(action_host, page_host):
            action_type = "External"
        else:
            action_type = "Same origin"

        forms.append({
            "Form": index,
            "Method": (form.get("method") or "GET").upper(),
            "Action": resolved_action,
            "Action Type": action_type,
            "Protocol": parsed_action.scheme.upper() or "Relative/unknown",
            "Password Field": "Yes" if has_password else "No",
            "Email Field": "Yes" if has_email else "No",
            "Input Count": len(controls),
            "Input Types": ", ".join(input_types),
        })

    return links, forms

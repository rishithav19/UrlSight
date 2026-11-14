from __future__ import annotations
import ipaddress
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from .features import extract_html_features

class UnsafeURLError(ValueError):
    """URL rejected by the scraper's basic destination checks."""

@dataclass
class ScrapeResult:
    requested_url: str
    final_url: str | None
    success: bool
    status_code: int | None
    page_title: str | None
    html_path: str | None
    screenshot_path: str | None
    features: dict
    error: str | None = None
    def to_dict(self):
        return asdict(self)

def _normalise_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        raise UnsafeURLError("URL is empty.")
    parsed = urlparse(value if "://" in value else "https://" + value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UnsafeURLError("Only valid http:// and https:// URLs are accepted.")
    if parsed.username or parsed.password:
        raise UnsafeURLError("URLs containing embedded credentials are not allowed.")
    return parsed.geturl()

def _is_public_destination(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        results = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    addresses = {entry[4][0] for entry in results}
    if not addresses:
        return False
    for value in addresses:
        try:
            if not ipaddress.ip_address(value.split("%")[0]).is_global:
                return False
        except ValueError:
            return False
    return True

class WebScraper:
    def __init__(self, output_dir="webscraper_output", timeout_ms=20000,
                 viewport=(1280, 960), headless=True):
        self.output_dir = Path(output_dir)
        self.timeout_ms = timeout_ms
        self.viewport = viewport
        self.headless = headless

    def scrape(self, url: str, sample_name="page") -> ScrapeResult:
        requested = _normalise_url(url)
        if not _is_public_destination(requested):
            raise UnsafeURLError("URL must resolve exclusively to public IP addresses.")

        name = "".join(c if c.isalnum() or c in "-_." else "_" for c in sample_name)[:80] or "page"
        out = self.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        html_file, image_file = out / "page.html", out / "screenshot.png"
        blocked = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless, args=["--disable-dev-shm-usage", "--no-sandbox"])
            context = browser.new_context(
                viewport={"width": self.viewport[0], "height": self.viewport[1]},
                accept_downloads=False, service_workers="block", ignore_https_errors=False
            )
            page = context.new_page()

            def guard(route, request):
                if not _is_public_destination(request.url):
                    blocked.append(request.url)
                    route.abort("blockedbyclient")
                else:
                    route.continue_()

            page.route("**/*", guard)
            status, final_url, title, html, error = None, None, None, "", None
            try:
                response = page.goto(requested, wait_until="domcontentloaded", timeout=self.timeout_ms)
                status = response.status if response else None
                page.wait_for_timeout(800)
                final_url, title, html = page.url, page.title(), page.content()
                html_file.write_text(html, encoding="utf-8", errors="replace")
                page.screenshot(path=str(image_file), full_page=False)
            except PlaywrightTimeoutError:
                error = f"Navigation timed out after {self.timeout_ms} ms."
                try:
                    final_url, title, html = page.url, page.title(), page.content()
                    if html:
                        html_file.write_text(html, encoding="utf-8", errors="replace")
                        page.screenshot(path=str(image_file), full_page=False)
                except PlaywrightError:
                    pass
            except PlaywrightError as exc:
                error = str(exc)[:500]
            finally:
                context.close()
                browser.close()

        features = extract_html_features(html, final_url or requested)
        features["blocked_request_count"] = len(blocked)
        features["blocked_request_hosts"] = [urlparse(x).hostname for x in blocked[:20]]
        return ScrapeResult(
            requested_url=requested, final_url=final_url, success=bool(html),
            status_code=status, page_title=title,
            html_path=str(html_file) if html_file.exists() else None,
            screenshot_path=str(image_file) if image_file.exists() else None,
            features=features, error=error
        )

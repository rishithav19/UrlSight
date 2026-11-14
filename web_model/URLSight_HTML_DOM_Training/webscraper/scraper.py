import ipaddress, socket
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

def is_public_host(host):
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(x[4][0]).is_global for x in infos)

def scrape(url, output_dir, timeout_ms=25000):
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password:
        raise ValueError("Provide an HTTP/HTTPS URL without embedded credentials.")
    if not is_public_host(u.hostname):
        raise ValueError("Host did not resolve exclusively to public IPs.")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    blocked = []
    def guard(route):
        p = urlparse(route.request.url)
        if p.scheme not in ("http", "https") or not p.hostname or not is_public_host(p.hostname):
            blocked.append(p.hostname or route.request.url); route.abort()
        else:
            route.continue_()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width":1280,"height":960}, accept_downloads=False, service_workers="block")
        page = context.new_page(); page.route("**/*", guard)
        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(1000)
        (out/"page.html").write_text(page.content(), encoding="utf-8")
        page.screenshot(path=str(out/"screenshot.png"), full_page=True)
        result = {"requested_url":url, "final_url":page.url, "status_code":response.status if response else None,
                  "title":page.title(), "html_path":str(out/"page.html"), "screenshot_path":str(out/"screenshot.png"),
                  "blocked_request_hosts":sorted(set(blocked))}
        context.close(); browser.close()
    return result

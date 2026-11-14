import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

from playwright.async_api import async_playwright


def validate_public_host(url):
    """Reject hostnames that resolve to non-public IP addresses.

    This is a prototype-level check, not a complete SSRF defense. A hardened
    deployment should run the browser in an isolated worker/container with
    network-level egress restrictions and DNS rebinding protection.
    """
    parsed = urlparse(url)
    host = parsed.hostname

    if not host:
        raise ValueError("The URL has no valid hostname.")

    host = host.rstrip(".").lower()
    if host == "localhost" or host.endswith(".local"):
        raise ValueError("Local hostnames are not allowed.")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as exc:
        raise ValueError("The hostname could not be resolved.") from exc

    if not addresses:
        raise ValueError("No IP addresses were found for this host.")

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Private or non-public hosts are not allowed.")

    return host


async def _capture_page_artifacts(url):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox"],
        )

        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            ignore_https_errors=False,
        )

        async def guard_request(route):
            request_url = route.request.url
            parsed = urlparse(request_url)

            if parsed.scheme not in ("http", "https"):
                await route.abort()
                return

            try:
                validate_public_host(request_url)
                await route.continue_()
            except Exception:
                await route.abort()

        await context.route("**/*", guard_request)
        page = await context.new_page()

        try:
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=20000,
            )
            await page.wait_for_timeout(1500)

            screenshot_bytes = await page.screenshot(
                full_page=False,
                type="png",
            )
            html_content = await page.content()
            final_url = page.url
            page_title = await page.title()

            return {
                "screenshot": screenshot_bytes,
                "html": html_content,
                "final_url": final_url,
                "title": page_title,
            }
        finally:
            await context.close()
            await browser.close()


def capture_page_artifacts(url):
    """Return screenshot and rendered HTML from one browser visit."""
    validate_public_host(url)
    return asyncio.run(_capture_page_artifacts(url))


def capture_screenshot(url):
    """Backward-compatible screenshot-only wrapper."""
    return capture_page_artifacts(url)["screenshot"]

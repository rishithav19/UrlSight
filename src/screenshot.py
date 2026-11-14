
import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

from playwright.async_api import async_playwright


def validate_public_host(url):
    """Reject non-public hostnames before opening a webpage."""
    parsed = urlparse(url)
    host = parsed.hostname

    if not host:
        raise ValueError("The URL has no valid hostname.")

    if host.lower() == "localhost" or host.endswith(".local"):
        raise ValueError("Local hostnames are not allowed.")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(host, None)
        }
    except socket.gaierror:
        raise ValueError("The hostname could not be resolved.")

    if not addresses:
        raise ValueError("No IP addresses were found.")

    for address in addresses:
        ip = ipaddress.ip_address(address)

        if not ip.is_global:
            raise ValueError("Private or non-public hosts are not allowed.")

    return host


async def _capture_page_artifacts(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
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

            # Capture both artifacts from the SAME rendered page.
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
    """Capture screenshot and rendered HTML in one browser session."""
    validate_public_host(url)
    return asyncio.run(_capture_page_artifacts(url))


def capture_screenshot(url):
    """Backward-compatible screenshot-only function."""
    artifacts = capture_page_artifacts(url)
    return artifacts["screenshot"]

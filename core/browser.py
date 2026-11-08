"""Shared Playwright browser helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from playwright.sync_api import Browser, Page, Playwright, sync_playwright
from playwright_stealth import Stealth

_STEALTH = Stealth(
    navigator_platform_override="MacIntel",
    navigator_languages_override=("en-US", "en"),
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@contextmanager
def launch_page(
    *,
    headless: bool = True,
    timeout_ms: int = 45_000,
) -> Iterator[tuple[Playwright, Browser, Page]]:
    """Yield a Chromium page and clean up on exit."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=headless,
            ignore_default_args=["--enable-automation"],
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/Toronto",
            color_scheme="light",
            user_agent=_DEFAULT_USER_AGENT,
        )
        _STEALTH.apply_stealth_sync(context)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        try:
            yield playwright, browser, page
        finally:
            context.close()
            browser.close()

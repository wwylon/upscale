"""Playwright smoke: Scale/Test MODE hides SIZE. Skip unless playwright installed."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("playwright")

playwright = pytest.importorskip("playwright.sync_api")


@pytest.mark.e2e
def test_test_tab_hides_size():
    """Set UPSCALE_E2E_URL to a running `upscale serve` base (e.g. http://127.0.0.1:8787/en/)."""
    url = os.environ.get("UPSCALE_E2E_URL", "").strip()
    if not url:
        pytest.skip("Set UPSCALE_E2E_URL to run Playwright Scale/Test smoke")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.get_by_role("button", name="Test").click()
        assert page.get_by_text("01 · SIZE").count() == 0
        page.get_by_role("button", name="Scale").first.click()
        assert page.get_by_text("01 · SIZE").count() >= 1
        browser.close()

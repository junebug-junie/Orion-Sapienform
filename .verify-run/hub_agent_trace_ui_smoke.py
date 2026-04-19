#!/usr/bin/env python3
"""Headless Hub UI smoke: Agent mode + mesh prompt; screenshot Agent Trace timeline."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    out = Path(__file__).resolve().parent / "hub_agent_trace_timeline.png"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto("http://127.0.0.1:8080/", wait_until="networkidle", timeout=60000)
        page.click('button.mode-btn[data-mode="agent"]')
        page.fill("#chatInput", "Check tailscale mesh status on this host.")
        page.click("#sendButton")
        # Hub prefers WebSocket when connected; HTTP /api/chat may not fire.
        page.wait_for_selector("text=tailscale_not_installed", timeout=180000)
        # Full-page capture: inline Agent Trace debug can be present but not "visible" to Playwright
        # (zero-size parent, another tab, or collapsed). Modal path is most reliable.
        page.evaluate(
            """() => {
              const el = document.getElementById('agentTraceDebugOpenModal');
              if (el) el.click();
            }"""
        )
        page.wait_for_selector("#agentTraceTimelineBody", timeout=15000)
        page.locator("#agentTraceModal").screenshot(path=str(out))
        body = page.inner_text("#agentTraceTimelineBody")
        browser.close()
    print(f"WROTE {out}")
    print("--- agentTraceDebugTimeline text (excerpt) ---")
    print(body[:4000])
    if "success" in body.lower() and "agent_chain" in body:
        # timeline rows include status labels; ensure delegate row is not success
        lines = [ln.strip() for ln in body.splitlines() if "agent_chain" in ln or "fail" in ln.lower()]
        print("--- relevant lines ---")
        for ln in lines[:30]:
            print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Headless live Hub run: Quick lane → send message → capture conversation + console + HTTP hints."""
from __future__ import annotations

import argparse
import sys
import time

from playwright.sync_api import sync_playwright


def _run_fast_two_turns(
    *,
    base_url: str,
    headless: bool,
    timeout_ms: int,
    probe1: str | None,
    probe2: str | None,
) -> int:
    """Quick (fast) only: send probe1, wait for reply, immediately send probe2, wait again. Prints per-turn wall ms."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    p1 = (probe1 or "").strip() or f"pw-2turn-a-{ts}-{time.time():.0f}"
    p2 = (probe2 or "").strip() or f"pw-2turn-b-{ts}-{time.time():.0f}"
    print(f"TRACE_PROBE_TURN1={p1}", flush=True)
    print(f"TRACE_PROBE_TURN2={p2}", flush=True)

    wait_js = """(probe) => {
      const root = document.getElementById('conversation');
      if (!root) return false;
      const kids = [...root.children];
      let userIdx = -1;
      for (let i = 0; i < kids.length; i++) {
        const el = kids[i];
        if (el.getAttribute('data-role') === 'user' && (el.innerText || '').includes(probe)) {
          userIdx = i;
          break;
        }
      }
      if (userIdx < 0) return false;
      for (let j = userIdx + 1; j < kids.length; j++) {
        const el = kids[j];
        if (el.getAttribute('data-role') === 'assistant') {
          const body = el.querySelector('p.whitespace-pre-wrap');
          const txt = (body && body.textContent) ? body.textContent.trim() : (el.innerText || '').trim();
          return txt.length > 15;
        }
      }
      return false;
    }"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"{base_url.rstrip('/')}/", wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_selector("#quickModeBtn", state="visible", timeout=timeout_ms)
        page.wait_for_selector("#chatInput", state="visible", timeout=timeout_ms)
        page.click("#quickModeBtn")
        page.wait_for_timeout(300)

        for turn, probe in ((1, p1), (2, p2)):
            t0 = time.perf_counter()
            page.fill("#chatInput", probe)
            page.click("#sendButton")
            try:
                page.wait_for_function(wait_js, arg=probe, timeout=timeout_ms)
            except Exception as exc:
                print(f"TURN{turn}_FAIL", repr(exc))
                print("--- #status ---")
                print(page.inner_text("#status")[:2000])
                browser.close()
                return 1
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"TURN{turn}_WALL_MS", round(elapsed_ms, 1), "PROBE", probe, flush=True)

        print("--- #status ---", page.inner_text("#status")[:400], flush=True)
        browser.close()
    return 0


def _run_one(
    *,
    base_url: str,
    quick_variant: str,
    headless: bool,
    timeout_ms: int,
    probe: str | None,
) -> int:
    probe = (probe or "").strip() or f"pw-quick-live-{time.time():.0f}"
    print(f"TRACE_PROBE={probe}", flush=True)
    console_lines: list[str] = []
    http_lines: list[str] = []
    ws_lines: list[str] = []

    def main() -> int:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            page.on("console", lambda msg: console_lines.append(f"{msg.type}: {msg.text}"))

            def _ws_log(direction: str, data: str) -> None:
                one = (data or "").replace("\n", " ").strip()
                if len(one) > 900:
                    one = one[:900] + "…"
                ws_lines.append(f"{direction} {one}")

            def on_websocket(ws):
                def on_sent(frame):
                    if isinstance(frame, str):
                        _ws_log("WS_OUT", frame)
                    else:
                        txt = getattr(frame, "text", None) or ""
                        _ws_log("WS_OUT", str(txt))

                def on_recv(frame):
                    if isinstance(frame, str):
                        _ws_log("WS_IN", frame)
                    else:
                        txt = getattr(frame, "text", None)
                        if txt is None and getattr(frame, "binary", None):
                            try:
                                txt = frame.binary.decode("utf-8", "replace")
                            except Exception:
                                txt = "[binary]"
                        _ws_log("WS_IN", str(txt or ""))

                try:
                    ws.on("framesent", on_sent)
                    ws.on("framereceived", on_recv)
                except Exception:
                    pass

            page.on("websocket", on_websocket)

            def on_response(resp):
                u = resp.url
                if "/api/chat" in u or "/api/session" in u or "ws" in u.lower():
                    try:
                        http_lines.append(f"{resp.status} {resp.request.method} {u[:160]}")
                    except Exception:
                        pass

            page.on("response", on_response)

            page.goto(f"{base_url.rstrip('/')}/", wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_selector("#quickModeBtn", state="visible", timeout=timeout_ms)
            page.wait_for_selector("#chatInput", state="visible", timeout=timeout_ms)

            if quick_variant == "stance":
                page.click("#quickModeMenuBtn")
                page.wait_for_selector("#quickModeMenu", state="visible", timeout=10000)
                page.click('.quick-variant-item[data-quick-variant="stance"]')
            else:
                page.click("#quickModeBtn")

            page.wait_for_timeout(300)

            page.fill("#chatInput", probe)
            page.click("#sendButton")

            try:
                page.wait_for_function(
                    """(probe) => {
                      const root = document.getElementById('conversation');
                      if (!root) return false;
                      const kids = [...root.children];
                      let userIdx = -1;
                      for (let i = 0; i < kids.length; i++) {
                        const el = kids[i];
                        if (el.getAttribute('data-role') === 'user' && (el.innerText || '').includes(probe)) {
                          userIdx = i;
                          break;
                        }
                      }
                      if (userIdx < 0) return false;
                      for (let j = userIdx + 1; j < kids.length; j++) {
                        const el = kids[j];
                        if (el.getAttribute('data-role') === 'assistant') {
                          const body = el.querySelector('p.whitespace-pre-wrap');
                          const txt = (body && body.textContent) ? body.textContent.trim() : (el.innerText || '').trim();
                          return txt.length > 15;
                        }
                      }
                      return false;
                    }""",
                    arg=probe,
                    timeout=timeout_ms,
                )
            except Exception as exc:
                print("WAIT_ASSISTANT_FAIL", repr(exc))
                print("--- #status ---")
                print(page.inner_text("#status")[:2000])
                print("--- #conversation (excerpt) ---")
                print(page.inner_text("#conversation")[:4000])
                print("--- http (last 30) ---")
                for ln in http_lines[-30:]:
                    print(ln)
                print("--- console (last 40) ---")
                for ln in console_lines[-40:]:
                    print(ln)
                print("--- websocket (last 30) ---")
                for ln in ws_lines[-30:]:
                    print(ln)
                browser.close()
                return 1

            last = page.evaluate(
                """(probe) => {
                  const root = document.getElementById('conversation');
                  if (!root) return '';
                  const kids = [...root.children];
                  let userIdx = -1;
                  for (let i = 0; i < kids.length; i++) {
                    const el = kids[i];
                    if (el.getAttribute('data-role') === 'user' && (el.innerText || '').includes(probe)) {
                      userIdx = i;
                      break;
                    }
                  }
                  if (userIdx < 0) return '';
                  for (let j = userIdx + 1; j < kids.length; j++) {
                    const el = kids[j];
                    if (el.getAttribute('data-role') === 'assistant') {
                      const body = el.querySelector('p.whitespace-pre-wrap');
                      return (body && body.textContent) ? body.textContent.trim() : (el.innerText || '').trim();
                    }
                  }
                  return '';
                }""",
                probe,
            )
            texts = page.locator('#conversation [data-role="assistant"]').all_inner_texts()
            print("QUICK_VARIANT", quick_variant)
            print("PROBE", probe)
            print("ASSISTANT_COUNT", len(texts))
            print("LAST_ASSISTANT_EXCERPT")
            print(last[:4000])
            print("--- #status ---")
            print(page.inner_text("#status")[:800])
            print("--- http (last 25) ---")
            for ln in http_lines[-25:]:
                print(ln)
            print("--- console (last 25) ---")
            for ln in console_lines[-25:]:
                print(ln)
            print("--- websocket (last 20) ---")
            for ln in ws_lines[-20:]:
                print(ln)
            browser.close()
        return 0

    return main()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080", help="Hub base URL")
    ap.add_argument(
        "--variant",
        choices=("fast", "stance", "both"),
        default="both",
        help="Quick (fast) only, Quick+stance only, or run both sequentially",
    )
    ap.add_argument("--headed", action="store_true", help="Show browser window")
    ap.add_argument("--timeout-ms", type=int, default=180000, help="Max wait for assistant reply")
    ap.add_argument(
        "--probe",
        default="",
        help="Exact user text to send (for log correlation). Default: auto-generated.",
    )
    ap.add_argument(
        "--two-fast-turns",
        action="store_true",
        help="Quick (fast) only: two back-to-back user messages; prints TURN1_WALL_MS / TURN2_WALL_MS (second often slower).",
    )
    ap.add_argument(
        "--probe2",
        default="",
        help="Second user message (with --two-fast-turns). Default: auto-generated.",
    )
    args = ap.parse_args()

    headless = not args.headed
    probe_opt = args.probe.strip() or None
    probe2_opt = args.probe2.strip() or None

    if args.two_fast_turns:
        if args.variant not in ("fast", "both"):
            print("--two-fast-turns requires --variant fast (ignoring stance).", flush=True)
        return _run_fast_two_turns(
            base_url=args.base_url,
            headless=headless,
            timeout_ms=args.timeout_ms,
            probe1=probe_opt,
            probe2=probe2_opt,
        )

    if args.variant == "both":
        a = _run_one(
            base_url=args.base_url,
            quick_variant="fast",
            headless=headless,
            timeout_ms=args.timeout_ms,
            probe=probe_opt,
        )
        b = _run_one(
            base_url=args.base_url,
            quick_variant="stance",
            headless=headless,
            timeout_ms=args.timeout_ms,
            probe=(f"{probe_opt or 'pw-quick-live'}-stance" if probe_opt else None),
        )
        return 0 if (a == 0 and b == 0) else 1
    return _run_one(
        base_url=args.base_url,
        quick_variant=args.variant,
        headless=headless,
        timeout_ms=args.timeout_ms,
        probe=probe_opt,
    )


if __name__ == "__main__":
    sys.exit(main())

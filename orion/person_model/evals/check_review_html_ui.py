#!/usr/bin/env python3
"""UI interaction check for the cleanup review HTML (AGENTS.md section 9).

Opens the file in headless Chromium, picks a choice and a note on the first
needs-your-call card, clicks Export, parses the downloaded CSV, reloads to check
choices persist, and asserts the page made no network request and threw no error.

    pip install playwright
    python orion/person_model/evals/check_review_html_ui.py \
        /tmp/person-linkage/person-cleanup-review.html  (never inside the repo: it holds chat snippets and the repo is public) [--chrome /path/to/chrome]
"""

import argparse
import csv
import io
from pathlib import Path


def _formula_cells_neutralized(chrome) -> bool:
    """Render a fixture card whose label starts with '=' and export it."""
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from playwright.sync_api import sync_playwright

    from orion.person_model import linkage as L
    from orion.person_model.review_html import render

    d = L.decide(L.Anchor("a", "x"), L.Candidate("c", "=HYPERLINK(\"http://e\")"), L.load_config(), L.LinkageContext())
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "f.html"
        f.write_text(render([d], title="t", run_id="fixture"), encoding="utf-8")
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=chrome) if chrome else p.chromium.launch()
            page = b.new_context(accept_downloads=True).new_page()
            page.goto(f.as_uri())
            with page.expect_download() as dl:
                page.click("#export")
            rows = list(csv.DictReader(io.StringIO(Path(dl.value.path()).read_text(encoding="utf-8"))))
            b.close()
    return rows[0]["candidate_label"].startswith("'=")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("html", type=Path)
    ap.add_argument("--chrome", default=None)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chrome) if args.chrome else p.chromium.launch()
        page = browser.new_context(accept_downloads=True).new_page()
        reqs, errs = [], []
        page.on("request", lambda r: reqs.append(r.url))
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(args.html.resolve().as_uri())
        n_cards = page.locator("article.card").count()
        card = page.locator("#needs_your_call article").first
        card.locator("input[value=reject]").check()
        note = 'not me, "quoted", with, commas'
        card.locator("textarea").fill(note)
        with page.expect_download() as dl:
            page.click("#export")
        rows = list(csv.DictReader(io.StringIO(Path(dl.value.path()).read_text(encoding="utf-8"))))
        page.reload()
        persisted = page.locator("#needs_your_call article").first.locator("textarea").input_value()
        browser.close()
    formula_ok = _formula_cells_neutralized(args.chrome)
    external = [u for u in reqs if not u.startswith(("file://", "blob:"))]
    noted = [r for r in rows if r["notes"] == note]
    checks = {
        "csv_has_every_card": len(rows) == n_cards,
        "choice_and_note_exported": len(noted) == 1 and noted[0]["your_choice"] == "reject",
        "persisted_after_reload": persisted == note,
        "no_network_requests": not external,
        "no_page_errors": not errs,
        "default_choices_marked_defaulted": all(r["defaulted"] in ("yes", "no", "") for r in rows) and any(r["defaulted"] == "yes" for r in rows),
        "formula_cells_neutralized": formula_ok,
    }
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'} {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

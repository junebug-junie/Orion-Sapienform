"""Inner Sky feed eval: is the live sky drawing real rows, or an empty shell?

Reads the running Hub's /api/inner-sky/events and /api/inner-sky/state (read-only) and checks:

* every experience has words (no blank label) and belongs to a registered organ;
* how many labels are only the kind's own word (no template yet) -- reported, not failed;
* the service graph and the day frame both answered;
* how many services spoke in the last hour and how many links are live.

Exit 1 when the sky would show empty lights or a part is down. Usage:

    python services/orion-hub/evals/run_inner_sky_feed_eval.py --hub http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.request


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as resp:  # noqa: S310 - operator-supplied local Hub URL
        return json.loads(resp.read())


def evaluate(events_doc: dict, state_doc: dict) -> dict:
    ev = events_doc.get("events") or []
    blank = [e for e in ev if not str(e.get("label") or "").strip()]
    bare = [e for e in ev if e.get("label") == str(e.get("kind", "")).replace("_", " ")]
    unregistered = [e for e in ev if e.get("organ") == "unregistered"]
    organs = collections.Counter(e.get("organ") for e in ev)
    dark = [o for o in events_doc.get("organs") or [] if not organs.get(o)]
    body = state_doc.get("body") or {}
    now = float(state_doc.get("now") or 0)
    services = body.get("services") or []
    live = [s for s in services if s.get("last") is not None and now - float(s["last"]) < 3600]
    failures = []
    if not ev:
        failures.append("no experiences in the last 12 hours")
    if blank:
        failures.append(f"{len(blank)} experiences with no words")
    if unregistered:
        failures.append(f"{len(unregistered)} experiences from unregistered kinds: "
                        f"{sorted({e.get('kind') for e in unregistered})}")
    if state_doc.get("body_error"):
        failures.append(f"service graph: {state_doc['body_error']}")
    if state_doc.get("focus_error"):
        failures.append(f"day frame: {state_doc['focus_error']}")
    return {
        "experiences": len(ev),
        "bindings": len(events_doc.get("cites") or []),
        "blank_labels": len(blank),
        "kind_word_only_labels": dict(collections.Counter(e.get("kind") for e in bare)),
        "experiences_by_organ": dict(organs.most_common()),
        "organs_with_no_experience": dark,
        "services": len(services),
        "services_live_last_hour": len(live),
        "links_live_24h": len(body.get("links") or []),
        "attention": (state_doc.get("focus") or {}).get("attention"),
        "failures": failures,
        "verdict": "FAIL" if failures else "PASS",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hub", default="http://127.0.0.1:8080", help="Hub base URL")
    args = ap.parse_args()
    base = args.hub.rstrip("/")
    report = evaluate(get(f"{base}/api/inner-sky/events"), get(f"{base}/api/inner-sky/state"))
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

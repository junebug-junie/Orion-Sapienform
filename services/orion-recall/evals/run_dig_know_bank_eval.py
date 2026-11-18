"""Live eval for ``orion.dig know`` (issue #3535): does a lookup with plain words find the target?

For each bank item that has lookup words, POST them to orion-recall's ``/lived/know`` (the same
endpoint ``python -P -m orion.dig know`` calls) and report whether any of the item's target ids
came back, as strong or weak, and at what rank. Prints ids and labels only, never memory text.

The bank (``~/.orion/evals/lived_memory_recall_bank.json``) and the words file
(``~/.orion/evals/dig_know_words.json``: ``{"<bank label>": "<words>", ...}``) are host-local and
never committed: both name things from Juniper's life.

  python services/orion-recall/evals/run_dig_know_bank_eval.py \\
      --url http://127.0.0.1:8260 [--bank ...] [--words ...]

Pass bar (Recall rev 2, #3535 report): the target is found for at least 5 items. Read-only, except
that orion-recall logs each lookup as a ``dig.know`` row in recall_telemetry (``turn_id`` is
``eval:dig-know``, so read-demand counts can exclude it).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List

DEFAULT_BANK = Path.home() / ".orion" / "evals" / "lived_memory_recall_bank.json"
DEFAULT_WORDS = Path.home() / ".orion" / "evals" / "dig_know_words.json"
EVAL_TURN_ID = "eval:dig-know"
PASS_AT_LEAST = 5


def _targets(item: Dict[str, Any]) -> List[str]:
    t = item.get("targets") or []
    if isinstance(t, str):
        t = ast.literal_eval(t)
    return [str(x) for x in t]


def score(answer: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
    hits = [h for h in answer.get("hits") or [] if isinstance(h, dict)]
    ids = [str(h.get("id")) for h in hits]
    found = [{"id": d, "rank": ids.index(d) + 1, "strength": hits[ids.index(d)].get("strength")}
             for d in targets if d in ids]
    return {"status": answer.get("status"), "backend": answer.get("backend"), "hits": len(hits),
            "found": found, "ok": bool(found)}


def _post(url: str, payload: dict) -> Any:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 -- internal service URL
        return json.loads(resp.read())


def run(bank: Dict[str, Any], words: Dict[str, str], url: str,
        post: Callable[[str, dict], Any] = _post) -> Dict[str, Any]:
    rows = []
    for item in bank.get("referring") or []:
        label = str(item.get("label"))
        if label not in words:
            continue
        t = time.perf_counter()
        answer = post(url.rstrip("/") + "/lived/know", {"words": words[label], "turn_id": EVAL_TURN_ID})
        rows.append({"label": label, "ms": int((time.perf_counter() - t) * 1000), **score(answer, _targets(item))})
    found = sum(1 for r in rows if r["ok"])
    return {"items": len(rows), "found": found, "pass": found >= PASS_AT_LEAST, "rows": rows}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8260")
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    ns = ap.parse_args(argv)
    report = run(json.loads(ns.bank.read_text()), json.loads(ns.words.read_text()), ns.url)
    for r in report["rows"]:
        print(json.dumps(r))
    print(f"found the target for {report['found']} of {report['items']} items "
          f"(bar: {PASS_AT_LEAST}) -> {'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""``python -P -m orion.dig know <words>``: look up your own lived memories (issue #3535).

A thin client of orion-recall's ``POST /lived/know``: the same lived-memory retriever a chat turn
uses (episode memories, stance messages, person facts), so a hit here is a memory recall would
inject. It never reads Postgres itself. orion-recall logs each lookup under the turn's id, so what
Orion looks up counts toward read demand.

Strong hits cleared recall's bar. Weak hits are below it but share a word with the search: they
are shown, labelled, never hidden behind "nothing found".
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from orion.dig.look import turn_id

#: Override first; Hub (host network) already has RECALL_SERVICE_URL=http://127.0.0.1:8260;
#: the harness governor (app-net) reaches the container by name.
RECALL_URL_ENV = "ORION_DIG_RECALL_URL"
FALLBACK_URL_ENV = "RECALL_SERVICE_URL"
DEFAULT_RECALL_URL = "http://orion-athena-recall:8260"
HTTP_TIMEOUT_SEC = 15.0
LINE_CHARS = 240
MAX_WORDS_CHARS = 1000

Post = Callable[[str, dict], Any]


class KnowError(Exception):
    pass


def recall_url(environ: Optional[dict] = None) -> str:
    env = os.environ if environ is None else environ
    return (env.get(RECALL_URL_ENV) or env.get(FALLBACK_URL_ENV) or DEFAULT_RECALL_URL).rstrip("/")


def _post_json(url: str, payload: dict) -> Any:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:  # noqa: S310 -- internal service URL
        return json.loads(resp.read())


def lookup(words: str, *, post: Optional[Post] = None, environ: Optional[dict] = None) -> dict:
    """POST the words to orion-recall. Raises KnowError when recall cannot be asked or answers
    with something unreadable: an unread lookup is never the same as "no memory"."""
    words = " ".join(str(words or "").split())[:MAX_WORDS_CHARS]
    if not words:
        raise KnowError("know: give the words to look for, e.g. `know pet stitches vet`")
    payload: dict = {"words": words}
    tid = turn_id(environ)
    if tid:
        payload["turn_id"] = tid
    url = recall_url(environ) + "/lived/know"
    try:
        data = (post or _post_json)(url, payload)
    except urllib.error.HTTPError as exc:
        raise KnowError(f"know: memory search could not run (orion-recall HTTP {exc.code}). "
                        "This is NOT the same as having no memory of it.") from exc
    except Exception as exc:  # noqa: BLE001 -- unreachable is unread, never "nothing found"
        raise KnowError(f"know: memory search could not run (orion-recall unreachable: {type(exc).__name__}). "
                        "This is NOT the same as having no memory of it.") from exc
    if not isinstance(data, dict) or "status" not in data:
        raise KnowError("know: orion-recall gave an unreadable answer. This is NOT the same as having no memory of it.")
    return data


def _line(hit: dict) -> str:
    kind, purpose = str(hit.get("kind") or "?"), str(hit.get("purpose") or "")
    label = kind if not purpose or purpose == kind or kind == "person_fact" else f"{kind}/{purpose}"
    text = " ".join(str(hit.get("text") or "").split())
    if len(text) > LINE_CHARS:
        text = text[: LINE_CHARS - 1].rstrip() + "…"
    return f"- ({label} · {hit.get('id')} · {hit.get('date')}) {text}"


def render(words: str, data: dict) -> list[str]:
    status = str(data.get("status") or "")
    hits = [h for h in data.get("hits") or [] if isinstance(h, dict)]
    strong = [h for h in hits if h.get("strength") == "strong"]
    weak = [h for h in hits if h.get("strength") != "strong"]
    degraded = [str(x) for x in data.get("degraded") or []]
    is_degraded = data.get("backend") == "degraded" or status == "degraded"
    unverified = int(data.get("unverified") or 0)
    lines = [f'know "{words}": {len(strong)} strong, {len(weak)} weak']
    if status in ("timeout", "error"):
        lines.append(f"Memory search could not finish ({status}). This is NOT the same as having no memory "
                     "of it: try again, or try fewer, more specific words.")
    elif status == "busy":
        lines.append("Memory search is busy with another lookup. Try again in a few seconds. "
                     "This is NOT the same as having no memory of it.")
    elif status == "skipped_short":
        lines.append("Your words had nothing to search with (only very common words), so nothing was searched. "
                     "This is NOT the same as having no memory of it: use names, places, things.")
    elif status == "off":
        lines.append("Memory search is switched off in orion-recall. This is NOT the same as having no memory of it.")
    if is_degraded:
        why = ", ".join(degraded) or "meaning search unavailable"
        lines.append(f"DEGRADED: searched by exact words only ({why}). A miss here does not mean no memory.")
    if unverified:
        lines.append(f"{unverified} more match(es) found but could not be checked against the live record, "
                     "so they are not shown. This is NOT the same as having no memory of it: try again.")
    if strong:
        lines.append("Exact-word matches (no reranker ran, so no bar was applied):" if is_degraded else
                     "Strong (cleared recall's bar: the kind of memory recall brings into a chat):")
        lines.extend(_line(h) for h in strong)
    if weak:
        lines.append("Weak (share words with your search but scored below the bar; check before relying on one):")
        lines.extend(_line(h) for h in weak)
    if (not hits and not unverified and not is_degraded
            and status not in ("timeout", "error", "off", "skipped_short", "busy", "unverified")):
        lines.append("No stored memory shares these words or fits their meaning. Try other words "
                     "(a name, a place, the thing itself) before saying you have no record.")
    return lines


__all__ = ["DEFAULT_RECALL_URL", "KnowError", "lookup", "recall_url", "render"]

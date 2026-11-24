"""Hourly rollup of world-first attention frames: the counts `dig attention` reads. Pure, stdlib only.

Why it exists (2026-11-18, PR "dig attention reads an hourly rollup"): `dig attention` (#3600) read
``substrate_attention_frames.frame_json`` (~8 KB a frame, ~41k frames a day, all TOAST). On a cold cache
the Postgres disk (a busy spinning disk, ~76% utilized, ~200 random reads/s) took 29 s for one day and
Orion's own reads hit the 25 s statement timeout 2 of 2. No index can help: every answer needs the JSON.
So orion-attention-runtime adds each frame's counts to this rollup when it saves the frame, in the same
transaction, and `dig attention` reads a few hundred small rows instead of 40k documents. The frames
themselves are pruned after 72 h; the rollup is not, so drift can reach a real 7-day prior (#3652).

One row per (hour, part, target, cause), counts summed over that hour's frames:

* part ``frame`` (target '', cause ''): a = frames, b = rest (no eligible target), c = contested
  (more than one eligible target).
* part ``won`` (cause ''): a = frames the target was eligible (in ``dominant_targets``), b = ranked
  first, c = eligible in a contested frame, d = ranked first in a contested frame.
* part ``not``: a = frames the target was judged and not attended, for that cause.

Only world-first targets count (``reasons[0]`` starts ``world-first:``); the frame's other suppressed
entries (capabilities, novelty-only nodes) are a different policy's view. "Ranked first" is first among
the world-first entries of ``dominant_targets`` (the same as the list head whenever the head is
world-first: every live frame checked on 2026-11-18). An entry with no ``target_id`` is skipped.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

WORLD_FIRST_PREFIX = "world-first:"

# World-first verdict causes, first match wins. The verdict is world_first.judge_candidate's reason
# (the text between ' -- ' and the first ';'). Patterns are SQL-LIKE shaped ('%' = any text) for
# readability; matched in Python only. "body high for itself, not sustained" is "not sustained".
CAUSES: tuple[tuple[str, str], ...] = (
    ("absent", "absent%"),
    ("insufficient history", "insufficient_history%"),
    ("past the orienting window", "%orienting window%"),
    ("not sustained", "%not sustained%"),
    ("below its band floor", "%floor%"),
    ("stale reading", "stale%"),
    ("not a measurement", "%not a measurement%"),
    ("no polarity", "no declared polarity%"),
    # world_first.yield_reading_to_chat: reading steps aside while Juniper is talking
    ("yields to chat", "yields to Juniper%"),
    # world:play while Orion is not watching the sitting (world_first.play_watch_reason)
    ("not watching", "not watching%"),
    ("quiet for itself", "%quiet for itself%"),
    ("usual for itself", "%usual for itself%"),
    ("high for itself", "%high for itself%"),
)
OTHER_CAUSE = "other"

ROLLUP_TABLE = "substrate_attention_hour_rollup"
ROLLUP_UPSERT_SQL = f"""
INSERT INTO {ROLLUP_TABLE} (hour, part, target, cause, a, b, c, d)
VALUES (:hour, :part, :target, :cause, :a, :b, :c, :d)
ON CONFLICT (hour, part, target, cause) DO UPDATE SET
    a = {ROLLUP_TABLE}.a + EXCLUDED.a,
    b = {ROLLUP_TABLE}.b + EXCLUDED.b,
    c = {ROLLUP_TABLE}.c + EXCLUDED.c,
    d = {ROLLUP_TABLE}.d + EXCLUDED.d
"""


def _like(pattern: str, text: str) -> bool:
    rx = "^" + ".*".join(re.escape(p) for p in pattern.split("%")) + "$"
    return re.match(rx, text, re.S) is not None


def verdict_of(reason: str) -> str:
    """'world-first: not attended -- body quiet for itself; ...' -> 'body quiet for itself'."""
    if " -- " not in reason:
        return ""
    return reason.split(" -- ", 1)[1].split(";", 1)[0].strip()


def classify_verdict(verdict: str) -> str:
    verdict = verdict.strip()
    for label, pattern in CAUSES:
        if _like(pattern, verdict):
            return label
    return OTHER_CAUSE


def classify(reason: str) -> str:
    """The cause class of one whole not-attended reason."""
    return classify_verdict(verdict_of(reason))


def _first_reason(target: Mapping[str, Any]) -> str:
    reasons = target.get("reasons") or []
    return str(reasons[0]) if reasons else ""


def _world_first(targets: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    return [t for t in targets or [] if _first_reason(t).startswith(WORLD_FIRST_PREFIX)]


def hour_of(ts: datetime | str) -> datetime:
    if isinstance(ts, str):
        t = ts.strip().replace(" ", "T", 1)
        ts = datetime.fromisoformat(t[:-1] + "+00:00" if t.endswith("Z") else t)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def frame_increments(frame: Mapping[str, Any]) -> dict[tuple[str, str, str], list[int]]:
    """(part, target, cause) -> [a, b, c, d] for one frame (``FieldAttentionFrameV1`` as a JSON dict)."""
    dominant = [t for t in _world_first(frame.get("dominant_targets")) if t.get("target_id")]
    n = len(dominant)
    out: dict[tuple[str, str, str], list[int]] = {("frame", "", ""): [1, int(n == 0), int(n > 1), 0]}

    def add(key: tuple[str, str, str], inc: list[int]) -> None:
        row = out.setdefault(key, [0, 0, 0, 0])
        for i, v in enumerate(inc):
            row[i] += v

    for rank, t in enumerate(dominant, start=1):
        first, contested = int(rank == 1), int(n > 1)
        add(("won", str(t.get("target_id") or ""), ""), [1, first, contested, first * contested])
    for t in _world_first(frame.get("suppressed_targets")):
        if not t.get("target_id"):
            continue
        add(("not", str(t.get("target_id") or ""), classify(_first_reason(t))), [1, 0, 0, 0])
    return out


def upsert_params(frame: Mapping[str, Any], generated_at: datetime | str) -> list[dict[str, Any]]:
    """Rows for ROLLUP_UPSERT_SQL, in key order (a fixed lock order across concurrent writers)."""
    hour = hour_of(generated_at)
    return [
        {"hour": hour, "part": p, "target": t, "cause": c, "a": v[0], "b": v[1], "c": v[2], "d": v[3]}
        for (p, t, c), v in sorted(frame_increments(frame).items())
    ]

#!/usr/bin/env python3
"""PreToolUse/Edit|Write hook: Phase 3 of the metric semantic layer.

docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md, "Phase 3
-- edit-time PreToolUse hook": the design's own stated highest-payoff piece,
and the one phase that was never built (phases 1/2/4 shipped 2026-08-12 and
2026-08-13; this one sat unbuilt until 2026-08-19). Same shape as the
graphify PreToolUse nudge that already works in this repo: informational,
fails open, never blocks.

Juniper's decision recorded in the design doc (2026-08-12): "always show the
full lineage card, on every edit touching a registered metric -- not only on
dead/suspect verdicts." A card gated on already-known-bad verdicts can't
prevent the failure where nobody knew the metric was bad yet.

Reads `.cache/metric_lineage.json` (built by
`scripts/refresh_metric_lineage_cache.py`) rather than recomputing the join
live: `orion.metrics.consumers.scan_repo()` walks ~3900 files and takes ~13s
(measured in the design doc), far too slow to run synchronously on every
Edit/Write. If the cache is missing or stale, this hook still returns near-
instantly -- it kicks a refresh off in the background (same "hand the
expensive part to a detached process" shape as
scripts/hooks/stop_worktree_wip_snapshot.py) and, for THIS call, either shows
nothing (no cache yet) or shows a card built from whatever cache exists
(possibly stale, never wrong-shaped -- it's the same join, just from an
earlier commit).

Fails open throughout: any missing cache, unreadable JSON, structurally
unexpected cache content, or unexpected tool_input shape means print nothing
and exit 0, exactly like RTK's and graphify's own "no match" behavior. This
is a nudge, not a gate -- it must never be the reason an unrelated edit
fails, and it must never be the reason a stderr traceback appears on every
subsequent Edit/Write in the session either.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / ".cache" / "metric_lineage.json"
REFRESH_LOCK_PATH = REPO_ROOT / ".cache" / "metric_lineage.refresh.lock"
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "refresh_metric_lineage_cache.py"

STALE_AFTER_SECONDS = 60 * 60  # 1h -- informational nudge, not a safety gate
REFRESH_COOLDOWN_SECONDS = 60  # don't spawn a second refresh while one runs
MAX_CARDS = 3
MAX_CONSUMERS_SHOWN = 4

_TOKEN_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b")


def _extract_edited_text(tool_input: dict) -> str:
    """Best-effort: the text actually being introduced by this edit.

    Edit: new_string. Write: content. NotebookEdit: new_source. Missing or
    unexpected shape -> empty string, which just means no tokens match.
    """
    for key in ("new_string", "content", "new_source"):
        val = tool_input.get(key)
        if isinstance(val, str):
            return val
    return ""


def _maybe_refresh_in_background() -> None:
    now = time.time()
    try:
        if REFRESH_LOCK_PATH.exists() and (now - REFRESH_LOCK_PATH.stat().st_mtime) < REFRESH_COOLDOWN_SECONDS:
            return  # a refresh was kicked off recently, don't pile on
        REFRESH_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        REFRESH_LOCK_PATH.touch()
        subprocess.Popen(
            [sys.executable, str(REFRESH_SCRIPT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(REPO_ROOT),
        )
    except Exception:
        pass  # informational nudge; a failed background refresh is not this call's problem


def _load_cache() -> dict | None:
    try:
        raw = CACHE_PATH.read_text(encoding="utf-8")
        return json.loads(raw)
    except Exception:
        return None


def _format_card(name: str, nodes: list[dict], hits: list[dict]) -> str:
    # Defensive against a structurally-unexpected cache (not just unreadable
    # JSON): a non-dict item here must degrade this one card, never crash the
    # hook on every subsequent Edit/Write until the cache is fixed. Review
    # finding 2026-08-19: a future field-rename in orion.metrics.lineage/
    # consumers.py this hook wasn't updated for, or a hand-edited/interrupted
    # cache file, tripped an unhandled AttributeError/ValueError here.
    non_test = [
        h for h in hits
        if isinstance(h, dict) and not h.get("is_test") and h.get("token") == name
    ]
    lines = [f"metric lineage: {name}"]
    for node in nodes:
        if not isinstance(node, dict):
            continue
        lines.append(
            f"  urn={node.get('urn')} surface={node.get('surface')} "
            f"producer={node.get('producer_service')}"
        )
        if node.get("meaning"):
            lines.append(f"  meaning: {node['meaning']}")
        declared = node.get("declared_consumers") or []
        feeds = node.get("feeds_dimensions") or []
        if declared:
            lines.append(f"  declared consumers: {', '.join(declared)}")
        if feeds:
            lines.append(f"  feeds dimensions: {', '.join(feeds)}")
    lines.append(f"  blast radius (discovered, non-test): {len(non_test)}")
    for hit in non_test[:MAX_CONSUMERS_SHOWN]:
        lines.append(f"      {hit.get('path')}:{hit.get('line')} [{hit.get('kind')}]")
    if len(non_test) > MAX_CONSUMERS_SHOWN:
        lines.append(f"      ... and {len(non_test) - MAX_CONSUMERS_SHOWN} more")
    if not non_test:
        lines.append("      none found by static scan -- see --generic-consumers before assuming dead")
    lines.append(
        f"  full card (blast-radius floor, generic whole-vector consumers, liveness): "
        f"python scripts/check_metric_lineage.py --metric {name}"
    )
    return "\n".join(lines)


def build_nudge(payload: dict, cache: dict | None, *, now: float | None = None) -> str | None:
    """Pure: payload + cache in, JSON hook response (or None) out.

    Split out from main() so tests exercise this directly instead of
    mocking stdin/subprocess -- the I/O and background-refresh side effects
    live only in main(), this function has none.
    """
    if not isinstance(payload, dict):
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None

    edited_text = _extract_edited_text(tool_input)
    if not edited_text.strip():
        return None
    if not isinstance(cache, dict):
        return None  # missing cache (None) and structurally-wrong cache alike

    nodes = cache.get("nodes") or []
    hits = cache.get("hits") or []
    if not isinstance(nodes, list) or not isinstance(hits, list):
        return None

    by_name: dict[str, list[dict]] = {}
    for node in nodes:
        if isinstance(node, dict):
            by_name.setdefault(node.get("name", ""), []).append(node)

    edited_tokens = set(_TOKEN_RE.findall(edited_text))
    matched = sorted(t for t in edited_tokens if t in by_name)
    if not matched:
        return None

    shown = matched[:MAX_CARDS]
    cards = [_format_card(name, by_name[name], hits) for name in shown]
    body = "\n\n".join(cards)
    if len(matched) > MAX_CARDS:
        body += f"\n\n({len(matched) - MAX_CARDS} more registered metric(s) touched, not shown)"

    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": body,
            }
        },
        ensure_ascii=False,
    )


def main() -> int:
    # Same convention as scripts/hooks/graphify_hook_guard_gate.sh: FCC
    # chat-turn subprocesses are token-budget-constrained and get no
    # code-navigation payoff from an edit-time nudge, so skip entirely.
    if os.environ.get("ORION_FCC_SUBPROCESS"):
        return 0

    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
    except Exception:
        return 0

    # Everything from here on touches a cache file this process did not
    # write and cannot fully trust the shape of. build_nudge() and
    # _format_card() are already defensive against a structurally-wrong
    # cache, but this outer guard is the actual contract: a crash here must
    # never become a stderr traceback on every subsequent Edit/Write in the
    # session (review finding 2026-08-19) -- print nothing, exit 0, same as
    # every other fail-open path in this hook.
    try:
        cache = _load_cache()
        if not isinstance(cache, dict):
            _maybe_refresh_in_background()
            return 0  # no cache yet, or not a dict -- treat the same

        try:
            generated_at = float(cache.get("generated_at") or 0)
        except (TypeError, ValueError):
            generated_at = 0.0
        age = time.time() - generated_at
        if age > STALE_AFTER_SECONDS:
            _maybe_refresh_in_background()  # use the stale cache below anyway

        out = build_nudge(payload, cache)
        if out:
            sys.stdout.write(out)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

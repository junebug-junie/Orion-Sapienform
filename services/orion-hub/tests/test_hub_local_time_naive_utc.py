"""Behavioral regression test for `formatHubLocalTime` in app.js.

Bug: orion-notify's `ChatAttentionState.created_at` / `NotificationRequest.created_at`
are naive-UTC `datetime` values (`Field(default_factory=datetime.utcnow)`), which
FastAPI/Pydantic serializes with no trailing `Z`/offset (e.g.
"2026-06-23T02:45:53.571957"). `new Date(...)` on a timezone-less ISO 8601
date-*time* string parses it as browser-local time (not UTC), so on a browser
whose local timezone is America/Denver the naive-UTC clock reading gets
mis-treated as if it were already Denver-local -- shifting the displayed
instant (and sometimes the displayed calendar day) by the full UTC-Denver
offset.

This test extracts the *actual* `formatHubLocalTime` source out of the shipped
app.js (not a hand-duplicated reimplementation) and executes it under Node
with TZ=America/Denver, to reproduce the real browser-timezone condition under
which the bug appeared. It asserts:

  1. A naive-UTC timestamp (no designator) renders as the correct Denver wall
     clock/day, not the pre-fix (wrong-day) result.
  2. The same instant expressed with an explicit `Z` or `+00:00` offset
     renders identically (proving no double-conversion / no regression for
     already-correct payloads, e.g. notification `received_at` which is built
     from an aware `datetime.now(timezone.utc).isoformat()`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
APP_JS = HUB_ROOT / "static" / "js" / "app.js"

_START_ANCHOR = "const HUB_LOCAL_TIMEZONE = 'America/Denver';"
_END_ANCHOR = "\nconst URL_PREFIX = '';"

NODE_AVAILABLE = shutil.which("node") is not None


def _denver_env() -> dict:
    env = dict(os.environ)
    env["TZ"] = "America/Denver"
    return env


def _extract_format_hub_local_time_source() -> str:
    """Pull the real `formatHubLocalTime` (and its helpers) straight out of app.js.

    Using the live source (instead of re-typing the function in the test)
    means this test actually exercises what ships, and fails if the
    extraction anchors drift out of sync with the real file.
    """
    src = APP_JS.read_text(encoding="utf-8")
    start = src.index(_START_ANCHOR)
    end = src.index(_END_ANCHOR, start)
    assert end > start, "formatHubLocalTime source block not found in app.js"
    return src[start:end]


def _run_node_denver(script: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", script],
        cwd=HUB_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_denver_env(),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout.strip())


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node is not available in this environment")
def test_naive_utc_created_at_renders_correct_denver_day_and_time() -> None:
    fn_src = _extract_format_hub_local_time_source()

    # Real payload observed live from GET /attention?status=pending on
    # orion-notify: a naive-UTC created_at with no trailing Z/offset.
    naive_utc = "2026-06-23T02:45:53.571957"
    # Same instant, but with an explicit Z designator.
    with_z = "2026-06-23T02:45:53.571957Z"
    # Same instant, explicit +00:00 offset (shape produced by an aware
    # datetime.now(timezone.utc).isoformat(), e.g. notification received_at).
    with_offset = "2026-06-23T02:45:53.571957+00:00"

    script = f"""
{fn_src}
console.log(JSON.stringify({{
  naive: formatHubLocalTime({json.dumps(naive_utc)}),
  withZ: formatHubLocalTime({json.dumps(with_z)}),
  withOffset: formatHubLocalTime({json.dumps(with_offset)}),
  resolvedTz: Intl.DateTimeFormat().resolvedOptions().timeZone,
}}));
"""
    out = _run_node_denver(script)

    # Sanity: the subprocess really did run with a Denver-local default TZ,
    # which is the precondition needed to reproduce the browser-local-time bug.
    assert out["resolvedTz"] == "America/Denver"

    # 2026-06-23T02:45:53 UTC = 2026-06-22T20:45:53 MDT (UTC-6, DST in effect
    # in late June). The pre-fix code showed "Jun 23, 2026, 2:45 AM MDT"
    # (the raw UTC digits re-labeled as Denver-local) -- wrong day, wrong time.
    expected = "Jun 22, 2026, 8:45 PM MDT"

    assert out["naive"] == expected, out
    assert "Jun 23" not in out["naive"], out

    # An explicit Z or +00:00 offset must resolve to the exact same wall
    # clock, i.e. no double-conversion/second Z appended to an
    # already-correct payload.
    assert out["withZ"] == expected, out
    assert out["withOffset"] == expected, out


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node is not available in this environment")
def test_offset_suffixed_value_is_not_double_converted() -> None:
    fn_src = _extract_format_hub_local_time_source()

    # A handful of shapes that already carry a timezone designator and must
    # be passed straight to `new Date(...)` untouched.
    already_tagged = [
        "2026-06-23T02:45:53.571957Z",
        "2026-06-23T02:45:53Z",
        "2026-06-23T02:45:53.571957+00:00",
        "2026-06-23T02:45:53-05:00",
    ]

    script = f"""
{fn_src}
const values = {json.dumps(already_tagged)};
console.log(JSON.stringify(values.map((v) => formatHubLocalTime(v))));
"""
    out_raw = subprocess.run(
        ["node", "-e", script],
        cwd=HUB_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_denver_env(),
    )
    assert out_raw.returncode == 0, out_raw.stderr or out_raw.stdout
    results = json.loads(out_raw.stdout.strip())

    # None of these should be null (a second Z appended to an already
    # offset-suffixed string would produce an invalid date -> null).
    assert all(r is not None for r in results), results

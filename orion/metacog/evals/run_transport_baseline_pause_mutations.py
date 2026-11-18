"""Mutation check: the silence-pause clauses of the transport baseline reducer
must each be load-bearing in ``orion/metacog/tests/test_transport_baseline.py``.

2026-10-13 fix (silence pauses latency episodes instead of closing them; see
``orion/metacog/transport_baseline.py`` "Episodes"). Each mutant below
re-introduces one way the fix could silently regress; the test suite must fail
on every one. A mutant whose anchor text is not found is an ERROR, not a pass
(a harness that cannot apply its mutation would otherwise read as all green).

The mutant is loaded under the real module name in a fresh interpreter, so the
tracked source file is never edited.

Run: ``PYTHONPATH=. python orion/metacog/evals/run_transport_baseline_pause_mutations.py``
Exit 0 = every mutant killed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "orion" / "metacog" / "transport_baseline.py"
TESTS = "orion/metacog/tests/test_transport_baseline.py"

# name -> (anchor, replacement). Each anchor must occur exactly once.
MUTANTS: dict[str, tuple[str, str]] = {
    "silence closes latency episodes again (pre-fix aging)": (
        "        # No evaluated latency this window (no traffic, or a pool still\n",
        "        if not evaluated and ks.pend_windows == 0:\n"
        "            for _c in (\"spike\", \"saturation\"):\n"
        "                _e = ks.episodes.get(_c)\n"
        "                if _e is not None and (now - _e.last_hot_ts) >= config.close_quiet_s:\n"
        "                    del ks.episodes[_c]\n"
        "                    emit(_c, \"close\", _e, close_reason=\"calm\")\n"
        "        # No evaluated latency this window (no traffic, or a pool still\n",
    ),
    "silence between calls credited as active time": (
        "            if ks.last_success_ts and 0 < now - ks.last_success_ts <= config.close_quiet_s:",
        "            if ks.last_success_ts and 0 < now - ks.last_success_ts:",
    ),
    "pre-fix hot credit (gap between evaluations <= close_quiet_s)": (
        "        ep.hot_s += credit_s",
        "        ep.hot_s += (now - ep.last_obs_ts) if 0 < now - ep.last_obs_ts <= config.close_quiet_s else 0.0",
    ),
    "silence resets hot_s": (
        "        ep.hot_s += credit_s",
        "        if now - ep.last_obs_ts > config.close_quiet_s:\n"
        "            ep.hot_s = 0.0\n"
        "        ep.hot_s += credit_s",
    ),
    "silent gap credited as calm time": (
        "    ep.calm_s += credit_s",
        "    ep.calm_s += now - ep.last_obs_ts",
    ),
    "credit only when the verdict agrees (flapping holds an episode forever)": (
        "        ep.hot_s += credit_s",
        "        if ep.last_obs_ts == ep.last_hot_ts:\n            ep.hot_s += credit_s",
    ),
    "an outage feeds latency active time (timeout-only windows count as traffic)": (
        "        if success > 0:\n            if ks.last_success_ts",
        "        if success > 0 or timeouts > 0:\n            if ks.last_success_ts",
    ),
    "no per-verdict credit cap": (
        "credit_s, ks.active_s = min(ks.active_s, config.max_credit_s), 0.0",
        "credit_s, ks.active_s = ks.active_s, 0.0",
    ),
    "floor not frozen while saturation is open or paused": (
        'elif "saturation" in ks.episodes or would_saturate:',
        "elif would_saturate:",
    ),
    "opening window may creep the floor": (
        'elif "saturation" in ks.episodes or would_saturate:',
        'elif "saturation" in ks.episodes:',
    ),
    "unwarm would-saturate freezes the floor": (
        "would_saturate = warm and (",
        "would_saturate = (",
    ),
    "max-silence bound removed": (
        "        if ep is not None and (now - ep.last_obs_ts) >= config.max_silence_s:",
        "        if False:",
    ),
}

_RUNNER = """
import importlib.util, sys
import orion.metacog  # parent package first
spec = importlib.util.spec_from_file_location("orion.metacog.transport_baseline", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
sys.modules["orion.metacog.transport_baseline"] = mod
spec.loader.exec_module(mod)
import orion.metacog as pkg
pkg.transport_baseline = mod
import pytest
sys.exit(pytest.main(["-q", "-x", "-p", "no:warnings", "-p", "no:cacheprovider", sys.argv[2]]))
"""


def run() -> dict[str, str]:
    src = MODULE.read_text()
    out: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as td:
        for name, (anchor, repl) in MUTANTS.items():
            n = src.count(anchor)
            if n != 1:
                out[name] = f"ERROR anchor found {n}x"
                continue
            path = Path(td) / "transport_baseline_mutant.py"
            path.write_text(src.replace(anchor, repl))
            r = subprocess.run(
                [sys.executable, "-c", _RUNNER, str(path), TESTS],
                cwd=ROOT, capture_output=True, text=True,
            )
            out[name] = "KILLED" if r.returncode == 1 else (
                "SURVIVED" if r.returncode == 0 else f"ERROR rc={r.returncode}: {r.stdout[-300:]}{r.stderr[-300:]}"
            )
    return out


def main() -> int:
    results = run()
    for name, verdict in results.items():
        print(f"{verdict:<10} {name}")
    bad = [n for n, v in results.items() if v != "KILLED"]
    print(f"{len(results) - len(bad)}/{len(results)} mutants killed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

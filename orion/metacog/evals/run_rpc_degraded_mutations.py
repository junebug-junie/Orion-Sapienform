"""Mutation check: every clause that keeps degraded RPC replies out of latency must be
load-bearing in its tests (2026-10-13; orion/core/bus/rpc_degraded.py).

Each mutant re-introduces one way the fix could silently regress -- most directly
"a degraded reply is recorded as a success", the live 2026-10-08 bug that dragged the
background lane's floor 13.9 s -> 1.26 s. The tests must fail on every one. A mutant
whose anchor is not found exactly once is an ERROR, not a pass (a harness that cannot
apply its mutation would otherwise read as all green).

The mutant is loaded under the real module name in a fresh interpreter, so no tracked
file is edited.

Run: ``PYTHONPATH=. python orion/metacog/evals/run_rpc_degraded_mutations.py``
Exit 0 = every mutant killed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

REDUCER = ("orion.metacog.transport_baseline", "orion/metacog/transport_baseline.py",
           "orion/metacog/tests/test_transport_baseline_degraded.py")
AGG = ("orion.core.bus.rpc_health", "orion/core/bus/rpc_health.py", "orion/core/bus/tests/test_rpc_degraded.py")
MARK = ("orion.core.bus.rpc_degraded", "orion/core/bus/rpc_degraded.py", "orion/core/bus/tests/test_rpc_degraded.py")
BUS = ("orion.core.bus.async_service", "orion/core/bus/async_service.py", "orion/core/bus/tests/test_rpc_degraded.py")
CHASSIS = ("orion.core.bus.bus_service_chassis", "orion/core/bus/bus_service_chassis.py",
           "orion/core/bus/tests/test_rpc_degraded.py")

# name -> (target, anchor, replacement). Each anchor must occur exactly once.
MUTANTS: dict[str, tuple[tuple[str, str, str], str, str]] = {
    # --- producer: the aggregator --------------------------------------------
    "degraded reply recorded as a success (the live 2026-10-08 bug)": (
        AGG,
        "            self._degraded_count += 1\n"
        "            self._bump_channel(request_channel)\n"
        "            if request_channel:\n"
        "                self._hop(hop_key(request_channel, health_label)).add_degraded()\n",
        "            self.record_success(\n"
        "                request_channel=request_channel, latency_ms=latency_ms or 0.0, health_label=health_label\n"
        "            )\n",
    ),
    "hop merge drops degraded_count": (
        AGG, "        self.degraded_count += other.degraded_count\n", "",
    ),
    "absorb drops pooled degraded_count": (
        AGG, "            self._degraded_count += other._degraded_count\n", "",
    ),
    # --- the marker ----------------------------------------------------------
    "a marker echoed from the request counts as the reply's own": (
        MARK, "                if reason != inherited:", "                if True:",
    ),
    "system.error replies are not degraded": (
        MARK, "        if getattr(envelope, \"kind\", None) in SYSTEM_ERROR_KINDS:",
        "        if False:",
    ),
    "note_degraded inside a request is dropped": (
        MARK, "    if bucket is not None and clean is not None and clean not in bucket:",
        "    if False:",
    ),
    # --- rpc_request ----------------------------------------------------------
    "worker path ignores the reply marker": (
        BUS, "                    reply_envelope=self._rpc_reply_envelopes.pop(key, None),",
        "                    reply_envelope=None,",
    ),
    "inline path ignores the reply marker": (
        BUS, "                            reply_env = decoded.envelope if decoded.ok else None",
        "                            reply_env = None",
    ),
    "worker path leaks stashed reply envelopes": (
        BUS, "                self._rpc_reply_envelopes.pop(key, None)\n", "",
    ),
    # --- Rabbit stamps the reply ---------------------------------------------
    "Rabbit never stamps a noted degradation": (
        CHASSIS, "            out = _stamp_degraded_reply(out, request=env, noted=noted)", "            pass",
    ),
    # --- consumer: the reducer ------------------------------------------------
    "reducer ignores degraded_count (an old consumer)": (
        REDUCER, "        degraded = _as_int(stats.get(\"degraded_count\"))", "        degraded = 0",
    ),
    "degraded pool may move the floor down": (
        REDUCER, "                if lv < ks.floor and degraded_pool:", "                if False:",
    ),
    "mostly-degraded pool does not taint (fast folds)": (
        REDUCER, "            tainted = ks.pend_tainted or degraded_pool", "            tainted = ks.pend_tainted",
    ),
    "any single refusal taints the pool (a trickle freezes the baseline)": (
        REDUCER, "            degraded_pool = ks.pend_degraded_n > ks.pend_n",
        "            degraded_pool = ks.pend_degraded_n > 0",
    ),
    "refusals between pools are forgotten": (
        REDUCER, "        ks.pend_degraded_n += degraded",
        "        if ks.pend_windows:\n            ks.pend_degraded_n += degraded",
    ),
    "refusals plus a timeout open zero_success": (
        REDUCER, "            timeouts > 0 and success == 0 and degraded == 0 and had_successes",
        "            timeouts > 0 and success == 0 and had_successes",
    ),
    "a degraded-only window keeps the traffic-time chain": (
        REDUCER, "        elif timeouts > 0 or degraded > 0:", "        elif timeouts > 0:",
    ),
    "degraded replies not counted as calls": (
        REDUCER, "        calls = success + timeouts + degraded", "        calls = success + timeouts",
    ),
    "pend_degraded_n not restored from a checkpoint": (
        REDUCER,
        '_KEY_INTS = ("fast_count", "level_count", "calls_count", "pend_n", "pend_windows", "hot_streak", "pend_degraded_n")',
        '_KEY_INTS = ("fast_count", "level_count", "calls_count", "pend_n", "pend_windows", "hot_streak")',
    ),
    "pend_degraded_n not reset with the pool": (
        REDUCER, "    ks.pend_tainted = False\n    ks.pend_degraded_n = 0\n", "    ks.pend_tainted = False\n",
    ),
    # --- review findings 2 and 5 ----------------------------------------------
    "a foreign pending future's reply is stashed (leak)": (
        BUS, "            if key in self._rpc_classify_keys:\n                self._rpc_reply_envelopes[key] = decoded.envelope",
        "            if True:\n                self._rpc_reply_envelopes[key] = decoded.envelope",
    ),
    "Rabbit hands the handler the request's marker": (
        CHASSIS, "        if (env.trace or {}).get(DEGRADED_TRACE_KEY) is not None:\n            env = env.model_copy(",
        "        if False:\n            env = env.model_copy(",
    ),
}

_RUNNER = """
import importlib, importlib.util, sys
name, path, tests = sys.argv[1], sys.argv[2], sys.argv[3:]
parent_name, _, child = name.rpartition(".")
parent = importlib.import_module(parent_name)  # parent package first
spec = importlib.util.spec_from_file_location(name, path)
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
spec.loader.exec_module(mod)
setattr(parent, child, mod)
import pytest
sys.exit(pytest.main(["-q", "-x", "-p", "no:warnings", "-p", "no:cacheprovider", *tests]))
"""


def run() -> dict[str, str]:
    out: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as td:
        for idx, (name, ((module, rel, tests), anchor, repl)) in enumerate(MUTANTS.items()):
            src = (ROOT / rel).read_text()
            n = src.count(anchor)
            if n != 1:
                out[name] = f"ERROR anchor found {n}x"
                continue
            path = Path(td) / f"mutant_{idx}.py"
            path.write_text(src.replace(anchor, repl))
            r = subprocess.run(
                [sys.executable, "-c", _RUNNER, module, str(path), tests],
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

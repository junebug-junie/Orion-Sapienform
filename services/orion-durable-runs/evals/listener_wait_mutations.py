"""Mutation eval for the listener wait (2026-11-03): every mutant must make the focused tests fail.

Each mutant breaks one piece of "a step whose RPC reached no listener waits out the deploy and
replays, no attempt spent" (admitted_graph.listener_wait, AdmissionRuntime.register's park, each
graph's call, the runner's raise, the bus's typed exception). A survivor means a test passes for
the wrong reason. The postgres tests need ORION_ADMISSION_TEST_DSN (without it they skip, and
mutants only they would catch survive -- so it is required here). Restores every file it touches.
"""
import os
import subprocess
import sys
from pathlib import Path

SVC = str(Path(__file__).resolve().parents[1])
WT = str(Path(SVC).parents[1])
PY = sys.executable
MUTANTS = [
 ("listener_wait never parks", "app/admitted_graph.py", "    if max_tries <= 0:\n        return None", "    if True:\n        return None"),
 ("cap off by one", "app/admitted_graph.py", "    if count > max_tries:", "    if count >= max_tries:"),
 ("backoff exponent", "app/admitted_graph.py", "    delay = min(cap, base * 2 ** (count - 1))", "    delay = min(cap, base * 2 ** count)"),
 ("no cause chain", "app/admitted_graph.py", "        exc = exc.__cause__ if exc.__cause__ is not None else (None if exc.__suppress_context__ else exc.__context__)", "        exc = None"),
 ("no deadline check", "app/admitted_graph.py", "    if retry_cannot_finish(state, retry_at, attempt_sec):\n        return None\n    released = await admission.release(state, \"listener_absent\")", "    released = await admission.release(state, \"listener_absent\")"),
 ("no hold park", "app/admitted_graph.py", "\"hold\": {\"retry_at\": at, LISTENER_WAIT_HOLD_KEY: count, \"case\": case}, **(extra or {})}", "**(extra or {})}"),
 ("failed_attempt skips wait", "app/admitted_graph.py", "    parked = await listener_wait(admission, state, exc, waiting=waiting, extra=extra, attempt_sec=attempt_sec)\n    if parked is not None:\n        return parked\n", ""),
 ("register ignores park", "app/admission_runtime.py", "        if hold.get(LISTENER_WAIT_HOLD_KEY) and not hold.get(\"request_id\") and hold.get(\"retry_at\") \\", "        if False and hold.get(LISTENER_WAIT_HOLD_KEY) and not hold.get(\"request_id\") and hold.get(\"retry_at\") \\"),
 ("reading skips wait", "app/reading_graph.py", "            if parked is not None:\n                return parked\n            # Reading's", "            # Reading's"),
 ("reading state drops counter", "app/reading_graph.py", "    listener_waits: int", "    _unused_listener_waits: int"),
 ("compactor skips wait", "app/compactor_digest_graph.py", "            if parked is not None:\n                return parked\n            # An urgent", "            # An urgent"),
 ("journal skips wait", "app/journal_compose_graph.py", "            if parked is not None:\n                return parked\n", ""),
 ("distill skips wait", "app/episode_distill_graph.py", "            if parked is not None:\n                return parked\n", ""),
 # The day letter is one harness turn since living Prime Part 3: it waits through the shared failed_attempt
 # (mutant "failed_attempt skips wait" above); this one checks it still counts its waits.
 ("orion_day state drops counter", "app/orion_day_graph.py", "    listener_waits: int", "    _unused_listener_waits: int"),
 ("runner swallows admitted", "app/runner.py", "            if admitted:\n                # Nobody received it", "            if False:\n                # Nobody received it"),
 ("bus raises base class", "../../orion/core/bus/async_service.py", "raise RpcPublishedToNobody(f\"no listener on {request_channel}: request published to nobody\")", "raise RpcNoListener(f\"no listener on {request_channel}: request published to nobody\")"),
 ("urgent budget ignored", "app/admitted_graph.py", "    if is_urgent(state):\n        # Urgent runs end", "    if False:\n        # Urgent runs end"),
 ("orphan chained to unheard replayed", "app/admitted_graph.py", "            return LOST_IN_FLIGHT, exc", "            pass"),
 ("self-sense state drops counter", "app/self_sense_graph.py", "    listener_waits: int", "    _unused_listener_waits: int"),
 ("awaiting also replayed", "app/admitted_graph.py", "        if isinstance(exc, RpcPublishedToNobody):", "        if isinstance(exc, RpcNoListener):"),
 # 2026-11-03: reading.turn re-sends a turn whose Hub vanished mid-reply, at most 2 times; no one else.
 ("in-flight opt-in ignored", "app/admitted_graph.py", "        max_tries, counter = min(max_tries, max(int(in_flight_resends), 0)), \"listener_in_flight_resends\"", "        max_tries, counter = max_tries, \"listener_in_flight_resends\""),
 ("in-flight never resends", "app/admitted_graph.py", "        max_tries, counter = min(max_tries, max(int(in_flight_resends), 0)), \"listener_in_flight_resends\"", "        max_tries, counter = 0, \"listener_in_flight_resends\""),
 ("in-flight ignores off switch", "app/admitted_graph.py", "        max_tries, counter = min(max_tries, max(int(in_flight_resends), 0)), \"listener_in_flight_resends\"", "        max_tries, counter = max(int(in_flight_resends), 0), \"listener_in_flight_resends\""),
 ("in-flight shares counter", "app/admitted_graph.py", "        max_tries, counter = min(max_tries, max(int(in_flight_resends), 0)), \"listener_in_flight_resends\"", "        max_tries, counter = min(max_tries, max(int(in_flight_resends), 0)), \"listener_waits\""),
 ("event drops case", "app/admitted_graph.py", "{\"waits\": count, \"case\": case, ", "{\"waits\": count, "),
 ("reading not opted in", "app/reading_graph.py", "                                         in_flight_resends=IN_FLIGHT_RESENDS,\n", ""),
 ("reading cap raised", "app/reading_graph.py", "IN_FLIGHT_RESENDS = 2\n", "IN_FLIGHT_RESENDS = 3\n"),
 ("reading state drops in-flight counter", "app/reading_graph.py", "    listener_in_flight_resends: int", "    _unused_listener_in_flight_resends: int"),
]
if not os.environ.get("ORION_ADMISSION_TEST_DSN"):
    print("ORION_ADMISSION_TEST_DSN (a disposable database) is required", file=sys.stderr)
    sys.exit(2)
env = dict(os.environ, PYTHONPATH=WT)
tests = ["tests/test_listener_wait.py", "tests/test_listener_wait_postgres.py"]
survivors = []
for name, rel, old, new in MUTANTS:
    path = os.path.normpath(os.path.join(SVC, rel))
    src = open(path).read()
    assert src.count(old) == 1, (name, src.count(old))
    mutated = src.replace(old, new)
    if name == "awaiting also replayed":
        mutated = mutated.replace("from orion.core.bus.async_service import RpcPublishedToNobody",
                                  "from orion.core.bus.async_service import RpcNoListener, RpcPublishedToNobody")
    open(path, "w").write(mutated)
    try:
        cmd = [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-W", "ignore", *tests]
        if "bus" in name:
            cmd = [PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-W", "ignore",
                   f"{WT}/orion/core/bus/tests/test_rpc_no_listener.py"]
        r = subprocess.run(cmd, cwd=SVC, env=env, capture_output=True, text=True, timeout=600)
        killed = r.returncode == 1   # test failures only: an import/collection crash is not a kill
    finally:
        open(path, "w").write(src)
    print(("KILLED  " if killed else "SURVIVED") + "  " + name, flush=True)
    if not killed:
        survivors.append(name)
print(f"{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} killed")
sys.exit(1 if survivors else 0)

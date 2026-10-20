"""Mutation check for the street attention gate (orion/substrate/street_attention.py).

Each mutation breaks one gate rule; the pure tests must fail for every one. A
surviving mutation means a rule has no test. Restores the file after each run.

Run from repo root:
  python services/orion-substrate-runtime/evals/run_street_gate_mutation_check.py
"""
import os
import subprocess, sys, pathlib

os.chdir(pathlib.Path(__file__).resolve().parents[3])
p = pathlib.Path("orion/substrate/street_attention.py")
orig = p.read_text()
muts = {
 "ring_no_bypass": ('if event.kind == RING_KIND:\n        return GateVerdict(True', 'if False:\n        return GateVerdict(True'),
 "p_le": ('if not p < GATE_P_VALUE:', 'if not p <= GATE_P_VALUE:'),
 "p_dropped": ('if not p < GATE_P_VALUE:', 'if False:'),
 "no_p_passes": ('if p is None:\n        return GateVerdict(False', 'if p is None:\n        return GateVerdict(True'),
 "camera_ignored": ('if stale is not None:', 'if False:'),
 "camera_age_ge": ('if age > max_age_sec:', 'if age > max_age_sec * 10:'),
 "cap_off_by_one": ('if used >= max(0, int(max_per_hour)):', 'if used > max(0, int(max_per_hour)):'),
 "cap_window": ('now_epoch - t < CAP_WINDOW_SEC)\n    if used', 'now_epoch - t < CAP_WINDOW_SEC * 2)\n    if used'),
 "hold_le": ('if now_epoch - h.epoch < hold_sec]', 'if now_epoch - h.epoch <= hold_sec]'),
 "blocked_held": ('            self.held.append(', '            pass\n        if True:\n            self.held.append('),
 "raw_no_floor": ('max(p, P_VALUE_FLOOR)', 'max(p, 1e-300)'),
}
tests = ["orion/substrate/tests/test_street_attention.py"]
bad = []
for name, (a, b) in muts.items():
    assert orig.count(a) == 1, name
    p.write_text(orig.replace(a, b))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-W", "ignore", *tests], capture_output=True, text=True)
    finally:
        p.write_text(orig)
    killed = r.returncode != 0
    print(f"{name}: {'killed' if killed else 'SURVIVED'}")
    if not killed: bad.append(name)
print("survivors:", bad)
sys.exit(1 if bad else 0)

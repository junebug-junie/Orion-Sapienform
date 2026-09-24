"""Eval: metacog capture acceptance checks 4-6 on real stored triggers.

Spec: docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md.

Replays ``orion/metacog/tests/fixtures/metacog_trigger_sample.jsonl`` -- 377
real ``metacog_trigger`` rows exported 2026-09-24 (stratified per kind /
evidence source / timeout count, private free text truncated, IPs scrubbed)
-- plus synthetic rows in the section-A ``transport_baseline`` shape through
``orion.metacog.evidence_map`` and asserts:

4. severity rank-correlates with the raw magnitude proxy for transport and
   telemetry -- judged against the tie-limited ceiling (see
   capture_replay.tie_limited_ceiling), because this fixture is stratified and
   its class balance is not the live one; the live replay script gates on the
   spec's flat rho >= 0.6 -- and no nominal row exceeds the median critical row
   of its kind;
5. causal_density takes > 10 distinct values (fixture pooled as one "day":
   the fixture is too thin per real day) and is 0 only when magnitude is 0;
6. no deterministic summary mentions zen, and the draft prompt no longer
   carries zen_state / pressure.

What this does NOT prove: the check-4 proxies are the same upstream numbers the
mapper bands, so checks 4a/4b pass by construction for a monotone mapper. They
are a REGRESSION GUARD (the mapper still reads and orders these fields), not
independent validation that the severity bands are "right". Check 6 for
LLM-authored summaries needs live rows after deploy: UNVERIFIED here.

Deterministic, no DB, no bus, no LLM.
Run: python orion/metacog/evals/run_capture_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orion.metacog.capture_replay import acceptance_failures, analyze  # noqa: E402

FIXTURE = _REPO_ROOT / "orion" / "metacog" / "tests" / "fixtures" / "metacog_trigger_sample.jsonl"
PROMPT = _REPO_ROOT / "orion" / "cognition" / "prompts" / "log_orion_metacognition_draft.j2"


def _synthetic_transport_baseline_rows() -> list[dict]:
    rows = []
    for i, (cond, over) in enumerate(
        [
            ("spike", {"z": z}) for z in (2.0, 3.1, 3.9, 4.6, 5.2, 7.5, 11.0)
        ]
        + [("saturation", {"saturation_ratio": r}) for r in (1.4, 2.2, 2.8, 3.5, 6.5)]
        + [("timeout", {"timeout_count": t}) for t in (1, 2, 4)]
        + [("zero_success", {"timeout_count": 3}), ("regime_shift", {"saturation_ratio": 2.4})]
    ):
        up = {
            "evidence_source": "transport_baseline",
            "condition": cond,
            "phase": "open",
            "key": "cortex-exec:orion:exec:request:LLMGatewayService",
            "baseline_ms": 9000.0,
            "window_mean_ms": 15000.0,
            "calls_per_min": 6.0,
            "calls_per_min_usual": 6.0,
            "timeout_count": 0,
        }
        up.update(over)
        rows.append({"trigger_kind": "transport", "reason": f"synthetic:{cond}:{i}", "upstream": up, "timestamp": "fixture"})
    return rows


def run() -> list[str]:
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    for r in rows:
        r["timestamp"] = "fixture"  # pool: per-day slices of a 377-row sample are too thin
    rows += _synthetic_transport_baseline_rows()
    res = analyze(rows)
    # Stratified fixture: class balance is not the live one, so check 4a is
    # judged against the tie-limited ceiling here; the live replay script
    # gates on the spec's flat 0.6.
    fails = acceptance_failures(res, min_rows_per_day=1, rho_mode="ceiling")

    template = PROMPT.read_text()
    if "zen_state" in template or "trigger.pressure" in template:
        fails.append("check6: draft prompt still renders zen_state/pressure")

    print(f"rows={res['rows']} distinct_density={res['density_distinct_per_day']['fixture']['new_distinct']}")
    for name, d in sorted(res["spearman_new"].items()):
        rho = "undefined" if d["rho"] is None else f"{d['rho']:.3f}"
        ceiling = "undefined" if d.get("ceiling") is None else f"{d['ceiling']:.3f}"
        print(f"  rho[{name}] = {rho} (ceiling {ceiling}, n={d['n']})")
    print("PASS" if not fails else "FAIL")
    for f in fails:
        print(f"  {f}")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)

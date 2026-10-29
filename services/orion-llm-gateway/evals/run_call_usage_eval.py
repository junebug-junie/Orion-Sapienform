#!/usr/bin/env python3
"""Is the per-call token ledger telling the truth? Reads llm_call_usage (read-only).

Run after deploy (and periodically):

    docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -At -F $'\\t' -c \\
      "select recorder,path,coalesce(correlation_id,''),outcome,coalesce(prompt_tokens::text,''),\\
       coalesce(completion_tokens::text,''),token_source,coalesce(duration_ms::text,''),\\
       coalesce(decode_tps::text,'') from llm_call_usage where recorded_at > now() - interval '24 hours'" \\
      | python services/orion-llm-gateway/evals/run_call_usage_eval.py

Checks, per (recorder, path):
* coverage -- share of SERVED rows with completion tokens. A served llama.cpp call always
  reports them; below 95% means the parser missed a reply shape.
* attribution -- share of rows with a correlation id. bus_rpc should be ~100%; http_anthropic
  is the harness (FCC spawns send X-Orion-Correlation-Id) and should be high; http_openai
  callers (AI Town NPCs, FCC proxy) mostly send none, which is expected.
* plausibility -- completion_tokens / decode_tps must not exceed model time (duration_ms):
  generating N tokens at R tokens/s takes at least N/R seconds. A violation means a count
  or a clock is wrong.
Exit 1 when coverage or plausibility fails; attribution is reported, not gated.
"""
from __future__ import annotations

import sys
from collections import defaultdict


def evaluate(lines) -> dict:
    groups: dict = defaultdict(lambda: {"rows": 0, "served": 0, "served_with_tokens": 0, "with_corr": 0,
                                        "implausible": 0, "completion_sum": 0, "prompt_sum": 0})
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 9:
            continue
        recorder, path, corr, outcome, prompt, completion, _source, duration, tps = parts
        g = groups[(recorder, path)]
        g["rows"] += 1
        g["with_corr"] += bool(corr)
        if outcome == "served":
            g["served"] += 1
            g["served_with_tokens"] += bool(completion)
        if completion:
            g["completion_sum"] += int(completion)
        if prompt:
            g["prompt_sum"] += int(prompt)
        if completion and tps and duration and float(tps) > 0:
            min_ms = int(completion) / float(tps) * 1000.0
            if min_ms > int(duration) * 1.05 + 50:
                g["implausible"] += 1
    report = {}
    ok = True
    for key, g in sorted(groups.items()):
        coverage = g["served_with_tokens"] / g["served"] if g["served"] else None
        row = {**g, "coverage": coverage, "attribution": g["with_corr"] / g["rows"] if g["rows"] else None}
        if coverage is not None and coverage < 0.95:
            ok = False
        if g["implausible"]:
            ok = False
        report["/".join(key)] = row
    return {"ok": ok and bool(groups), "groups": report}


def main() -> int:
    result = evaluate(sys.stdin)
    if not result["groups"]:
        print("llm_call_usage: no rows -- ledger not deployed or not writing (UNVERIFIED)")
        return 1
    for name, g in result["groups"].items():
        cov = "n/a" if g["coverage"] is None else f"{g['coverage']:.0%}"
        print(f"{name:32s} rows={g['rows']:6d} served={g['served']:6d} token_coverage={cov:>5s} "
              f"with_corr={g['attribution']:.0%} implausible={g['implausible']} "
              f"prompt_sum={g['prompt_sum']} completion_sum={g['completion_sum']}")
    print("PASS" if result["ok"] else "FAIL")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

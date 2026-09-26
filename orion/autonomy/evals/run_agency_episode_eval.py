#!/usr/bin/env python3
"""Replay captured live metadata through the actual CLI and grade evidence honesty.

No model calls or live connections. This evaluates the report's claims, not
Orion's learning quality, and cannot certify a causally closed live loop.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "fixtures/agency_episode_metadata.json"


def run(data):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metadata.json"
        path.write_text(json.dumps(data))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/analysis/report_agency_episodes.py"), "--input", str(path)],
            env={**os.environ, "ORION_PG_DSN": "invalid://offline-only"},
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode, json.loads(result.stdout)


def main() -> int:
    source = json.loads(FIXTURE.read_text())
    checks = []

    code, report = run(source)
    asks = [e for e in report["episodes"] if e["lane"] == "contractor_ask"]
    motors = [e for e in report["episodes"] if e["lane"] == "motor"]
    checks.append(("live_sample_coverage", code == 0 and len(asks) == 1 and len(motors) == 2))
    checks.append(("offered_reply_not_learning", asks[0]["links"]["consumption_marker"]["status"] == "observed" and asks[0]["links"]["later_choice"]["status"] == "unverified"))
    checks.append(("recent_execution_without_field_score_visible", any(e["links"]["intervention"]["status"] == "observed" and e["links"]["field_scored_outcome"]["status"] == "missing" for e in motors)))
    checks.append(("late_prediction_not_precommitted", any("expectation_frame_inserted_after_result" in e["issues"] for e in motors) and all(e["links"]["expectation_precommitted"]["status"] == "unverified" for e in motors)))
    checks.append(("no_invented_causal_closure", all(e["verdict"] == "UNVERIFIED" for e in report["episodes"])))
    checks.append(("visual_outcome_kept_separate", any(e["outcome_path"] == "visual" and e["links"]["visual_outcome"]["status"] == "observed" and e["results"][0]["visual_outcome"] == "produced" for e in motors)))

    duplicate = deepcopy(source)
    for value in duplicate.values():
        if isinstance(value, dict):
            value["rows"] = list(reversed(value["rows"] * 2))
    _, replayed = run(duplicate)
    checks.append(("duplicate_reordered_replay_identical", report == replayed))

    unavailable = deepcopy(source)
    unavailable["graph_briefs"] = {"status": "unavailable", "rows": []}
    code, degraded = run(unavailable)
    ask = next(e for e in degraded["episodes"] if e["lane"] == "contractor_ask")
    checks.append(("outage_is_not_absence", code == 2 and ask["links"]["response"]["status"] == "unverified"))

    unrelated = deepcopy(source)
    for b in unrelated["graph_briefs"]["rows"]:
        b["help_id"] = "unrelated"
    _, rejected = run(unrelated)
    ask = next(e for e in rejected["episodes"] if e["lane"] == "contractor_ask")
    checks.append(("no_time_only_reply_attribution", ask["links"]["response"]["status"] == "missing"))

    poisoned = deepcopy(source)
    for value in poisoned.values():
        if isinstance(value, dict):
            for row in value["rows"]:
                row["summary"] = "PRIVATE_SENTINEL"
                row["question"] = "PRIVATE_SENTINEL"
    _, redacted = run(poisoned)
    checks.append(("no_prose_in_report", "PRIVATE_SENTINEL" not in json.dumps(redacted)))
    print(json.dumps({"eval": "agency_episode_evidence_honesty", "checks": dict(checks), "passed": sum(ok for _, ok in checks), "total": len(checks)}, indent=2))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

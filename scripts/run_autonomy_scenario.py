#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from orion.autonomy.verification import DEFAULT_REPORT_DIR, AutonomyVerificationHarness, load_scenarios, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the autonomy replay + verification harness as one scenario workflow.")
    parser.add_argument("--scenario-pack", default=None)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--publish-bus", action="store_true")
    parser.add_argument("--graphdb", action="store_true")
    parser.add_argument("--wait-sec", type=float, default=0.0)
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_DIR / "autonomy_scenario_report.json"))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_DIR / "autonomy_scenario_report.md"))
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_pack) if args.scenario_pack else load_scenarios(only=args.scenario)
    harness = AutonomyVerificationHarness()
    report = harness.run(
        scenarios,
        publish_bus=args.publish_bus,
        verify_graphdb=args.graphdb,
        wait_sec=args.wait_sec,
    )
    write_report(report, json_out=Path(args.json_out), md_out=Path(args.md_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

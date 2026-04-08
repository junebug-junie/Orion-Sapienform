#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from orion.autonomy.verification import DEFAULT_REPORT_DIR, AutonomyVerificationHarness, load_scenarios, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify autonomy artifact graphs locally and optionally in GraphDB.")
    parser.add_argument("--scenario-pack", default=None)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--graphdb", action="store_true")
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT_DIR / "autonomy_verification_report.json"))
    parser.add_argument("--md-out", default=str(DEFAULT_REPORT_DIR / "autonomy_verification_report.md"))
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_pack) if args.scenario_pack else load_scenarios(only=args.scenario)
    harness = AutonomyVerificationHarness()
    report = harness.run(scenarios, publish_bus=False, verify_graphdb=args.graphdb)
    write_report(report, json_out=Path(args.json_out), md_out=Path(args.md_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

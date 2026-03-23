#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = str(Path(__file__).resolve().parent)
SERVICE_DIR = ROOT / "services" / "orion-cortex-exec"
if sys.path and sys.path[0] == SCRIPT_DIR:
    sys.path.pop(0)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from app.self_study_harness import render_self_study_harness, run_self_study_harness


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Orion self-study validation harness.")
    parser.add_argument(
        "--format",
        choices=("human", "json", "both"),
        default="both",
        help="Output mode for stdout.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the machine-readable JSON result.",
    )
    parser.add_argument(
        "--skip-degraded",
        action="store_true",
        help="Skip the developer-only degraded backend scenario.",
    )
    parser.add_argument(
        "--soak-iterations",
        type=int,
        default=1,
        help="Repeat a narrow reflective retrieval flow N times to detect drift.",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_self_study_harness(
            include_degraded=not args.skip_degraded,
            soak_iterations=max(1, args.soak_iterations),
        )
    )
    rendered_json = result.model_dump_json(indent=2)

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(rendered_json + "\n", encoding="utf-8")

    if args.format in {"human", "both"}:
        print(render_self_study_harness(result))
    if args.format in {"json", "both"}:
        if args.format == "both":
            print()
        print(rendered_json)

    return 1 if result.summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

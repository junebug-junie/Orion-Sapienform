from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .orchestrator import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PageIndex-ready chat episode corpus.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--markdown-path", required=True)
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--end-at", default=None)
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=86400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        result = run_pipeline(
            output_dir=Path(args.output_dir),
            markdown_path=Path(args.markdown_path),
            start_at=_parse_dt(args.start_at),
            end_at=_parse_dt(args.end_at),
            max_rows=args.max_rows,
            include_reasoning=args.include_reasoning,
        )
        print(json.dumps({"ok": True, "artifacts": result}))
        if not args.loop:
            break
        time.sleep(max(60, args.interval_seconds))


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


if __name__ == "__main__":
    main()

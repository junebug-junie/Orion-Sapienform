from __future__ import annotations

import argparse
import asyncio

from .main import run_digest_now


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Orion notify digest")
    parser.add_argument("--window-hours", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(run_digest_now(window_hours=args.window_hours, kind="manual"))


if __name__ == "__main__":
    main()

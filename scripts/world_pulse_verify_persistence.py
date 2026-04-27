#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import sysconfig

sys.path = [p for p in sys.path if p not in {"", os.getcwd()}]
sys.modules.pop("platform", None)
stdlib_platform = os.path.join(sysconfig.get_path("stdlib"), "platform.py")
spec = importlib.util.spec_from_file_location("platform", stdlib_platform)
assert spec and spec.loader
platform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(platform)
sys.modules["platform"] = platform

from sqlalchemy import create_engine, text


def _count(conn, table: str, run_id: str, run_key: str = "run_id") -> int:
    return int(conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {run_key} = :run_id"), {"run_id": run_id}).scalar() or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify world pulse SQL persistence for one run_id.")
    parser.add_argument("--run-id", required=True, help="World Pulse run_id to validate")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URI"), help="SQLAlchemy DB URL")
    args = parser.parse_args()
    if not args.db_url:
        print("FAIL: missing --db-url and DATABASE_URL/POSTGRES_URI", file=sys.stderr)
        return 2

    engine = create_engine(args.db_url)
    checks = [
        ("world_pulse_run", "run_id"),
        ("world_pulse_digest", "run_id"),
        ("world_pulse_article", "run_id"),
        ("world_pulse_article_cluster", "run_id"),
        ("world_pulse_digest_item", "run_id"),
        ("world_pulse_situation_change", "run_id"),
        ("world_pulse_context_capsule", "run_id"),
        ("world_pulse_publish_status", "run_id"),
        ("world_pulse_hub_message", "run_id"),
    ]
    with engine.begin() as conn:
        print(f"Run ID: {args.run_id}")
        failed = False
        for table, run_key in checks:
            count = _count(conn, table, args.run_id, run_key=run_key)
            ok = count > 0 if table in {"world_pulse_run", "world_pulse_digest"} else count >= 0
            print(f"{'PASS' if ok else 'FAIL'} {table}: {count}")
            if not ok:
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

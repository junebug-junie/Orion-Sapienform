from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import sysconfig
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SERVICE_APP = ROOT / "services" / "orion-world-pulse"
sys.path = [p for p in sys.path if not p.endswith("/orion/schemas")]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVICE_APP))

stdlib_platform = Path(sysconfig.get_paths()["stdlib"]) / "platform.py"
spec = importlib.util.spec_from_file_location("platform", stdlib_platform)
if spec and spec.loader:
    platform_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(platform_mod)
    sys.modules["platform"] = platform_mod

from app.services.pipeline import run_world_pulse


def main() -> int:
    parser = argparse.ArgumentParser(description="World pulse smoke script")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()
    fixture_items = None
    if args.fixtures:
        fixture_path = ROOT / "services" / "orion-world-pulse" / "tests" / "fixtures" / "rss_items.json"
        fixture_items = json.loads(fixture_path.read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc).isoformat()
        for row in fixture_items:
            row["published_at"] = row.get("published_at") or now
    result = run_world_pulse(requested_by="test", dry_run=args.dry_run, fixture_items=fixture_items)
    payload = result.model_dump(mode="json")
    digest = payload.get("digest") or {}
    digest_items = digest.get("items") or []
    worth_reading = digest.get("things_worth_reading") or []
    worth_watching = digest.get("things_worth_watching") or []
    print(
        json.dumps(
            {
                "run_id": payload.get("run", {}).get("run_id"),
                "status": payload.get("run", {}).get("status"),
                "articles_accepted": payload.get("run", {}).get("articles_accepted"),
                "digest_item_count": len(digest_items),
                "situation_brief_count": payload.get("publish_status", {}).get("briefs_count"),
                "situation_change_count": payload.get("publish_status", {}).get("changes_count"),
                "worth_reading_count": len(worth_reading),
                "worth_watching_count": len(worth_watching),
                "capsule_item_count": len((payload.get("capsule") or {}).get("salient_topics") or []),
                "email_preview": (digest.get("title") or "")[:120],
                "email_dry_run": bool(args.dry_run),
                "hub_message_status": payload.get("run", {}).get("hub_publish_status"),
                "hub_message_dry_run": bool(args.dry_run),
                "graph_dry_run_status": payload.get("run", {}).get("graph_emit_status"),
                "sql_dry_run_status": payload.get("run", {}).get("sql_emit_status"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

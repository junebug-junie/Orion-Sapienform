from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

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

from app.services.emit_graph import build_graph_delta
from app.services.emit_sql import build_sql_envelopes
from app.services.pipeline import run_world_pulse
from app.services.source_registry import load_source_registry
from app.settings import settings
from app.services.renderers import render_email_digest, render_hub_digest
from orion.core.bus.bus_schemas import ServiceRef


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description="World Pulse integration smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--approved-sources", action="store_true")
    args = parser.parse_args()

    fixture_items = None
    if args.fixtures:
        fixture_path = ROOT / "services" / "orion-world-pulse" / "tests" / "fixtures" / "rss_items.json"
        fixture_items = json.loads(fixture_path.read_text(encoding="utf-8"))
        now_iso = datetime.now(timezone.utc).isoformat()
        for row in fixture_items:
            row["published_at"] = row.get("published_at") or now_iso

    checks: list[tuple[str, bool]] = []
    reachability: list[dict[str, str]] = []
    if args.approved_sources:
        registry = load_source_registry(settings.world_pulse_sources_config_path)
        for source in registry.sources:
            if not (source.enabled and source.approved):
                continue
            anchor = source.url or (source.urls[0] if source.urls else "")
            host = urlparse(anchor).hostname or "n/a"
            reachability.append({"source_id": source.source_id, "strategy": source.strategy or "rss", "host": host})
    result = run_world_pulse(
        requested_by="test",
        dry_run=args.dry_run,
        fixture_items=fixture_items,
    )
    checks.append(("fixture_run", result.run.status in {"completed", "partial"}))
    checks.append(("digest_generated", result.digest is not None and len(result.digest.items) > 0 if result.digest else False))
    checks.append(("situation_generated", result.run.situation_briefs_updated > 0 and result.run.situation_changes_created > 0))
    checks.append(("capsule_generated", result.capsule is not None))

    if result.digest is None:
        print("FAIL: digest missing")
        return 1

    email = render_email_digest(
        result.digest,
        subject_prefix="Orion Daily World Pulse",
        to=[],
        from_email=None,
        dry_run=True,
    )
    checks.append(("email_preview_rendered", bool(email.subject and email.plaintext_body)))

    hub = render_hub_digest(result.digest)
    checks.append(("hub_message_payload_generated", bool(hub.message_id and hub.rendered_markdown)))

    source_ref = ServiceRef(name="orion-world-pulse", version="0.1.0", node="athena")
    sql_envs = build_sql_envelopes(
        source_ref=source_ref,
        run_result=result,
        claims=[],
        events=[],
        entities=[],
        briefs=[],
        changes=[],
        learning=[],
    )
    checks.append(("sql_envelopes_generated", len(sql_envs) >= 2))

    graph = build_graph_delta(result.digest, dry_run=True, allowed_item_ids={item.item_id for item in result.digest.items})
    checks.append(("graph_delta_generated", bool(graph.triples or graph.triple_count == 0)))

    ok = all(v for _, v in checks)
    output = {
        "run_id": result.run.run_id,
        "status": result.run.status,
        "coverage_status": (result.digest.coverage_status if result.digest else "empty"),
        "article_clusters": result.run.metrics.get("article_clusters", 0),
        "singleton_cluster_count": result.run.metrics.get("singleton_cluster_count", 0),
        "multi_article_cluster_count": result.run.metrics.get("multi_article_cluster_count", 0),
        "average_articles_per_cluster": result.run.metrics.get("average_articles_per_cluster", 0),
        "capped_situation_changes": result.run.metrics.get("capped_situation_changes", False),
        "covered_sections": result.run.metrics.get("covered_sections", []),
        "missing_sections": {
            "required": result.run.metrics.get("missing_required_sections", []),
            "recommended": result.run.metrics.get("missing_recommended_sections", []),
        },
        "articles_accepted": result.run.articles_accepted,
        "digest_item_count": len(result.digest.items),
        "max_digest_items_total": result.digest.max_digest_items_total,
        "situation_brief_count": result.run.situation_briefs_updated,
        "situation_change_count": result.run.situation_changes_created,
        "worth_reading_count": len(result.digest.things_worth_reading),
        "worth_watching_count": len(result.digest.things_worth_watching),
        "capsule_topic_count": len((result.capsule.salient_topics if result.capsule else [])),
        "sql_envelope_count": len(sql_envs),
        "graph_triple_count": graph.triple_count,
        "checks": [{k: _status(v)} for k, v in checks],
        "approved_source_reachability": reachability,
        "overall": _status(ok),
    }
    print(json.dumps(output, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

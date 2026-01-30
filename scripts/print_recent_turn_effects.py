from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

scripts_dir = Path(__file__).resolve().parent
repo_root = scripts_dir.parent
service_root = repo_root / "services" / "orion-sql-writer"
if str(scripts_dir) in sys.path:
    sys.path.remove(str(scripts_dir))
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(service_root))

from orion.schemas.telemetry.turn_effect import compute_deltas_from_turn_effect

DEFAULT_POSTGRES_URL = "postgresql://postgres:postgres@localhost:5432/conjourney"


def _extract_phi_blocks(spark_meta: Dict[str, Any]) -> Dict[str, Any]:
    metadata = spark_meta.get("metadata") if isinstance(spark_meta, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    rich = metadata.get("spark_meta_rich")
    if not isinstance(rich, dict):
        rich = spark_meta.get("spark_meta_rich") if isinstance(spark_meta, dict) else {}
    if not isinstance(rich, dict):
        rich = {}
    return {
        "phi_before": rich.get("phi_before"),
        "phi_after": rich.get("phi_after"),
        "phi_post_before": rich.get("phi_post_before"),
        "phi_post_after": rich.get("phi_post_after"),
    }


def _format_phi_block(block: Any) -> str:
    if not isinstance(block, dict):
        return "-"
    return (
        f"v={block.get('valence')}, e={block.get('energy')}, "
        f"c={block.get('coherence')}, n={block.get('novelty')}"
    )


def _format_delta(delta: Dict[str, float]) -> str:
    if not isinstance(delta, dict) or not delta:
        return "-"
    return (
        f"v={delta.get('valence')}, e={delta.get('energy')}, "
        f"c={delta.get('coherence')}, n={delta.get('novelty')}"
    )


def _flatten_delta(delta: Dict[str, float], prefix: str) -> Dict[str, Optional[float]]:
    if not isinstance(delta, dict):
        return {
            f"{prefix}_coherence": None,
            f"{prefix}_valence": None,
            f"{prefix}_energy": None,
            f"{prefix}_novelty": None,
        }
    return {
        f"{prefix}_coherence": delta.get("coherence"),
        f"{prefix}_valence": delta.get("valence"),
        f"{prefix}_energy": delta.get("energy"),
        f"{prefix}_novelty": delta.get("novelty"),
    }


def _print_rows(rows: Iterable[Dict[str, Any]], *, wide: bool = False) -> None:
    headers = [
        "corr_id",
        "created_at",
        "phi_before",
        "phi_after",
        "phi_post_before",
        "phi_post_after",
        "turn_effect_summary",
        "delta_user",
        "delta_assistant",
        "delta_turn",
    ]
    if wide:
        headers.extend(
            [
                "delta_user_coherence",
                "delta_user_valence",
                "delta_user_energy",
                "delta_user_novelty",
                "delta_assistant_coherence",
                "delta_assistant_valence",
                "delta_assistant_energy",
                "delta_assistant_novelty",
                "delta_turn_coherence",
                "delta_turn_valence",
                "delta_turn_energy",
                "delta_turn_novelty",
            ]
        )
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row.get(h, "")) for h in headers))


def _resolve_db_url(
    *,
    cli_db_url: Optional[str],
    sqlite_path: Optional[Path],
) -> str:
    if sqlite_path:
        return f"sqlite:///{sqlite_path}"
    if cli_db_url:
        return cli_db_url
    env_url = os.getenv("ORION_SQL_URL")
    if env_url:
        return env_url
    for env_key in ("POSTGRES_URI", "DATABASE_URL"):
        env_val = os.getenv(env_key)
        if env_val:
            return env_val
    return DEFAULT_POSTGRES_URL


def _format_dry_run(url: str, query: str) -> str:
    from sqlalchemy.engine.url import make_url

    parsed = make_url(url)
    user = parsed.username or "-"
    host = parsed.host or "-"
    port = parsed.port or "-"
    database = parsed.database or "-"
    driver = parsed.drivername
    return (
        f"driver={driver} host={host} port={port} db={database} user={user}\n"
        f"query={query}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Print recent turn-effect analytics from chat_history_log.")
    parser.add_argument("--limit", type=int, default=20, help="Max rows to return (default: 20)")
    parser.add_argument("--hours", type=float, default=None, help="Limit to last N hours")
    parser.add_argument("--since-hours", type=float, default=None, help="Alias for --hours")
    parser.add_argument("--format", choices=("wide", "csv"), default=None, help="Output format")
    parser.add_argument("--include-deltas", action="store_true", help="Include flattened delta columns")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--db-url", type=str, default=None, help="Override database URL")
    parser.add_argument("--sqlite-path", type=Path, default=None, help="Use sqlite DB at path")
    parser.add_argument("--dry-run", action="store_true", help="Print connection info and query, no DB call")
    args = parser.parse_args()

    db_url = _resolve_db_url(cli_db_url=args.db_url, sqlite_path=args.sqlite_path)
    base_query = "SELECT correlation_id, created_at, spark_meta FROM chat_history_log"
    where_clause = ""
    params: Dict[str, Any] = {}
    hours = args.hours if args.hours is not None else args.since_hours
    if hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        where_clause = " WHERE created_at >= :cutoff"
        params["cutoff"] = cutoff
    order_limit = " ORDER BY created_at DESC"
    if args.limit:
        order_limit += " LIMIT :limit"
        params["limit"] = args.limit
    query = f"{base_query}{where_clause}{order_limit}"

    if args.dry_run:
        print(_format_dry_run(db_url, query))
        return 0

    from sqlalchemy import create_engine, text

    engine = create_engine(db_url, pool_pre_ping=True)
    rows = []
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.mappings().all()
    except Exception as exc:
        print(f"Query failed: {exc}")
        return 1

    output_rows = []
    for row in rows:
        spark_meta = row.get("spark_meta")
        if isinstance(spark_meta, str):
            try:
                spark_meta = json.loads(spark_meta)
            except Exception:
                spark_meta = {}
        if not isinstance(spark_meta, dict):
            spark_meta = {}
        phi_blocks = _extract_phi_blocks(spark_meta)
        turn_effect = spark_meta.get("turn_effect") if isinstance(spark_meta, dict) else None
        deltas = compute_deltas_from_turn_effect(turn_effect or {})
        delta_user = _flatten_delta(deltas.get("user", {}), "delta_user")
        delta_assistant = _flatten_delta(deltas.get("assistant", {}), "delta_assistant")
        delta_turn = _flatten_delta(deltas.get("turn", {}), "delta_turn")
        output_rows.append(
            {
                "corr_id": row.get("correlation_id"),
                "created_at": row.get("created_at"),
                "phi_before": _format_phi_block(phi_blocks.get("phi_before")),
                "phi_after": _format_phi_block(phi_blocks.get("phi_after")),
                "phi_post_before": _format_phi_block(phi_blocks.get("phi_post_before")),
                "phi_post_after": _format_phi_block(phi_blocks.get("phi_post_after")),
                "turn_effect_summary": spark_meta.get("turn_effect_summary"),
                "delta_user": _format_delta(deltas.get("user", {})),
                "delta_assistant": _format_delta(deltas.get("assistant", {})),
                "delta_turn": _format_delta(deltas.get("turn", {})),
                **delta_user,
                **delta_assistant,
                **delta_turn,
            }
        )

    if not output_rows:
        print("No rows found.")
        return 0

    wide = args.include_deltas or args.format == "wide"
    _print_rows(output_rows, wide=wide)

    if args.format == "csv" and not args.csv:
        args.csv = Path("turn_effects.csv")

    if args.csv:
        csv_path = args.csv
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"Wrote CSV to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

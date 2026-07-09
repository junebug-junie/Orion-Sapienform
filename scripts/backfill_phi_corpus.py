#!/usr/bin/env python3
"""Backfill InnerStateFeaturesV1 JSONL corpus from substrate_self_state (Plan 2 training)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/backfill_phi_corpus.py` puts scripts/ on sys.path[0],
# which shadows stdlib `platform` via scripts/platform/ and breaks sqlalchemy.
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.schemas.self_state import SelfStateV1
from orion.telemetry.corpus_gate import is_corpus_row_healthy

_INNER_STATE_PATH = REPO_ROOT / "services" / "orion-spark-introspector" / "app" / "inner_state.py"
_SPEC = importlib.util.spec_from_file_location("spark_inner_state_backfill", _INNER_STATE_PATH)
assert _SPEC and _SPEC.loader
_inner_state = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_inner_state)

RollingRobustScaler = _inner_state.RollingRobustScaler
build_inner_state_features = _inner_state.build_inner_state_features
COGNITIVE_FEATURE_NAMES = _inner_state.COGNITIVE_FEATURE_NAMES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill phi encoder corpus from Postgres")
    parser.add_argument(
        "--postgres-uri",
        default=os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL"),
        help="Postgres DSN (default: POSTGRES_URI env)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/telemetry/phi/corpus/inner_state.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--sample-every", type=int, default=50, help="Keep every Nth self_state row")
    parser.add_argument("--features-version", default="seed-v2")
    parser.add_argument("--exec-trajectory-max-age-sec", type=int, default=120)
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Write all rows regardless of the corpus health gate (still computes/reports reasons; for debugging)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.postgres_uri:
        raise SystemExit("missing --postgres-uri or POSTGRES_URI")

    engine = create_engine(args.postgres_uri)
    with engine.connect() as conn:
        traj_row = conn.execute(
            text(
                "SELECT projection_json FROM substrate_execution_trajectory_projection "
                "ORDER BY generated_at DESC LIMIT 1"
            )
        ).fetchone()
        trajectory = None
        if traj_row is not None:
            raw = traj_row[0]
            trajectory = raw if isinstance(raw, dict) else json.loads(raw)

        rows = conn.execute(
            text("SELECT self_state_json, generated_at FROM substrate_self_state ORDER BY generated_at ASC")
        ).fetchall()

    scaler = RollingRobustScaler(maxlen=256)
    prev_felt = None
    prev_headline = None
    degenerate_streak = 0
    written = 0
    skipped = 0
    skipped_reasons: Counter = Counter()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for idx, (ss_json, _gen_at) in enumerate(rows):
            if args.sample_every > 1 and idx % args.sample_every != 0:
                continue
            payload = ss_json if isinstance(ss_json, dict) else json.loads(ss_json)
            ss = SelfStateV1.model_validate(payload)
            inner, prev_felt, degenerate_streak = build_inner_state_features(
                ss,
                scaler,
                features_version=args.features_version,
                grammar_degraded=False,
                trajectory_projection=trajectory,
                exec_trajectory_max_age_sec=args.exec_trajectory_max_age_sec,
                prev_felt=prev_felt,
                prev_headline=prev_headline,
                degenerate_streak=degenerate_streak,
            )
            prev_headline = inner.headline
            healthy, reasons = is_corpus_row_healthy(
                inner, cognitive_feature_names=COGNITIVE_FEATURE_NAMES
            )
            if not healthy:
                skipped_reasons.update(reasons)
                if not args.allow_degraded:
                    skipped += 1
                    continue
            fh.write(json.dumps(inner.model_dump(mode="json"), separators=(",", ":")) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "corpus": str(args.out),
                "source_rows": len(rows),
                "sample_every": args.sample_every,
                "written": written,
                "skipped_unhealthy": skipped,
                "skipped_reasons": dict(skipped_reasons),
                "allow_degraded": args.allow_degraded,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

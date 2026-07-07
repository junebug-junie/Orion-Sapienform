"""Salience weight refit — DOCUMENTED STUB. Not run in production this round.

The hybrid seam: it joins attention_salience_trace (feature distributions) with
attention_loop_outcome (human Resolve/Dismiss + implicit decayed_unattended
labels) and would emit candidate combiner weights + a new weights_version. This
round it proves the label table is consumable and returns the SEEDED weights
unchanged. When labels accumulate, replace `candidate_weights_from_labels` with a
real fit (e.g. logistic regression: resolved=1, dismissed/decayed=0).

Usage (read-only, safe):
    python scripts/refit_salience_weights.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from orion.substrate.attention.salience import SEED_WEIGHTS, WEIGHTS_VERSION


def load_labels(limit: int = 5000) -> list[dict[str, Any]]:
    """Read outcome label rows. Best-effort; [] if no DB configured."""
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return []
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(uri, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT verdict, features_at_close FROM attention_loop_outcome "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"limit": limit},
            ).mappings().all()
        out: list[dict[str, Any]] = []
        for r in rows:
            feats = r["features_at_close"]
            if isinstance(feats, str):
                feats = json.loads(feats)
            out.append({"verdict": r["verdict"], "features_at_close": feats or {}})
        return out
    except Exception:
        return []


def candidate_weights_from_labels(labels: list[dict[str, Any]]) -> tuple[dict[str, float], str]:
    """STUB: prove the label table is consumable; return seeded weights.

    A real fit lands here later. We deliberately keep production weights seeded
    (spec non-goal: no training now) but tag the version so a future run diverges.
    """
    n = len([l for l in labels if l.get("verdict")])
    version = f"{WEIGHTS_VERSION}" if n == 0 else f"{WEIGHTS_VERSION}+labels={n}(seeded)"
    return dict(SEED_WEIGHTS), version


def main() -> None:
    parser = argparse.ArgumentParser(description="Salience weight refit (stub)")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.parse_args()
    labels = load_labels()
    weights, version = candidate_weights_from_labels(labels)
    print(json.dumps({"labels_seen": len(labels), "weights_version": version, "weights": weights}, indent=2))


if __name__ == "__main__":
    main()

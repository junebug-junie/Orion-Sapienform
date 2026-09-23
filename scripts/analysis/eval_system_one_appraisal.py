#!/usr/bin/env python3
"""Shadow evaluation for persisted System One appraisal frames.

This script intentionally does not choose behavior thresholds. It answers the
metric-quality gate's first live-data questions: are outputs present, variable,
non-saturated, and inspectably distributed? Calibration against labeled outcomes
is a separate required step before any appraisal can drive behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import create_engine, text


def _safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": _safe_mean(values),
        "stdev": _safe_stdev(values),
        "min": min(values),
        "max": max(values),
    }


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, list[float]] = defaultdict(list)
    confidences: dict[str, list[float]] = defaultdict(list)
    high_level_p: dict[str, list[float]] = defaultdict(list)
    top_levels: dict[str, Counter[str]] = defaultdict(Counter)
    malformed = 0
    providers: Counter[str] = Counter()
    models: Counter[str] = Counter()

    for row in rows:
        payload = row.get("frame_json")
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            malformed += 1
            continue
        providers[str(payload.get("provider") or "")] += 1
        models[str(payload.get("model_id") or "")] += 1
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            malformed += 1
            continue
        for question_id, answer in answers.items():
            if not isinstance(answer, dict):
                malformed += 1
                continue
            if isinstance(answer.get("score"), (int, float)):
                scores[question_id].append(float(answer["score"]))
            if isinstance(answer.get("confidence"), (int, float)):
                confidences[question_id].append(float(answer["confidence"]))
            probabilities = answer.get("probabilities")
            if isinstance(probabilities, dict) and probabilities:
                numeric = {
                    str(k): float(v)
                    for k, v in probabilities.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))
                }
                if numeric:
                    top_levels[question_id][max(numeric, key=numeric.get)] += 1
                    if "2" in numeric:
                        high_level_p[question_id].append(numeric["2"])

    questions = sorted(set(scores) | set(confidences) | set(high_level_p))
    return {
        "rows": len(rows),
        "malformed_rows_or_answers": malformed,
        "providers": dict(providers),
        "models": dict(models),
        "questions": {
            question_id: {
                "score": _summary(scores[question_id]),
                "confidence": _summary(confidences[question_id]),
                "p_level_2": _summary(high_level_p[question_id]),
                "argmax_level_counts": dict(top_levels[question_id]),
                "degenerate_flags": {
                    "score_flat": (
                        len(scores[question_id]) >= 2
                        and max(scores[question_id]) == min(scores[question_id])
                    ),
                    "p_level_2_flat": (
                        len(high_level_p[question_id]) >= 2
                        and max(high_level_p[question_id])
                        == min(high_level_p[question_id])
                    ),
                    "p_level_2_saturated_high": (
                        bool(high_level_p[question_id])
                        and _safe_mean(high_level_p[question_id]) is not None
                        and _safe_mean(high_level_p[question_id]) >= 0.95
                    ),
                    "p_level_2_saturated_low": (
                        bool(high_level_p[question_id])
                        and _safe_mean(high_level_p[question_id]) is not None
                        and _safe_mean(high_level_p[question_id]) <= 0.05
                    ),
                },
            }
            for question_id in questions
        },
        "promotion_status": "shadow_only_unless_separately_labeled_and_calibrated",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--postgres-uri", default=os.getenv("POSTGRES_URI", ""))
    args = parser.parse_args()
    if not args.postgres_uri:
        raise SystemExit("POSTGRES_URI or --postgres-uri is required")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(0.0, args.hours))
    engine = create_engine(args.postgres_uri, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT frame_json
                FROM substrate_system_one_appraisal
                WHERE generated_at >= :cutoff
                ORDER BY generated_at ASC
                LIMIT :limit
                """
            ),
            {"cutoff": cutoff, "limit": max(1, int(args.limit))},
        ).mappings().all()

    print(json.dumps(evaluate([dict(row) for row in rows]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

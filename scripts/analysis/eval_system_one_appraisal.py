#!/usr/bin/env python3
"""Evaluation for persisted System One appraisal frames + curiosity gates.

Shadow questions stay observational. When curiosity admission is live, this
script also reports gate blocked/admitted rates and evaluator outcomes
conditional on System One level — without inventing a reward function.
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
    question_sets: Counter[str] = Counter()
    generated_ats: list[datetime] = []

    for row in rows:
        payload = row.get("frame_json")
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            malformed += 1
            continue
        providers[str(payload.get("provider") or "")] += 1
        models[str(payload.get("model_id") or "")] += 1
        question_sets[str(payload.get("question_set_id") or "")] += 1
        generated = payload.get("generated_at")
        if isinstance(generated, str):
            try:
                generated_ats.append(datetime.fromisoformat(generated.replace("Z", "+00:00")))
            except ValueError:
                pass
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
    window = {
        "rows": len(rows),
        "first_generated_at": min(generated_ats).isoformat() if generated_ats else None,
        "last_generated_at": max(generated_ats).isoformat() if generated_ats else None,
        "hours_span": (
            (max(generated_ats) - min(generated_ats)).total_seconds() / 3600.0
            if len(generated_ats) >= 2
            else None
        ),
    }
    return {
        **window,
        "malformed_rows_or_answers": malformed,
        "providers": dict(providers),
        "models": dict(models),
        "question_sets": dict(question_sets),
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
                    "argmax_single_level_dominated": (
                        bool(top_levels[question_id])
                        and max(top_levels[question_id].values())
                        / max(1, sum(top_levels[question_id].values()))
                        >= 0.95
                    ),
                },
            }
            for question_id in questions
        },
        "promotion_status": {
            "curiosity_pull": "live_endogenous_admission_gate_0_noop_1_and_2_admit",
            "reverie_fit": "observational_degenerate_argmax0_on_initial_window",
            "attention_interrupt": "observational_causal_loop_risk",
            "deliberation_need": "observational_not_task_scoped",
        },
    }


def evaluate_curiosity_gates(gate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Post-live calibration view over candidate-set gate_json rows."""
    by_result: Counter[str] = Counter()
    by_level: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    evaluator_by_level: dict[str, Counter[str]] = defaultdict(Counter)
    seeded = 0

    for row in gate_rows:
        gate = row.get("gate_json")
        if isinstance(gate, str):
            gate = json.loads(gate)
        if not isinstance(gate, dict):
            continue
        seeded += 1
        result = str(gate.get("gate_result") or "")
        level = str(gate.get("selected_level") if gate.get("selected_level") is not None else "")
        by_result[result] += 1
        if level:
            by_level[level] += 1
        if gate.get("fallback_reason"):
            fallback_reasons[str(gate["fallback_reason"])] += 1
        evaluator_outcome = gate.get("evaluator_outcome")
        if level and evaluator_outcome is not None:
            evaluator_by_level[level][str(evaluator_outcome)] += 1
        elif level and result == "system_one_curiosity_noop":
            evaluator_by_level[level]["skipped_by_gate"] += 1

    return {
        "seeded_ticks_with_gate": seeded,
        "gate_results": dict(by_result),
        "selected_level_counts": dict(by_level),
        "fallback_reasons": dict(fallback_reasons),
        "evaluator_outcome_by_level": {
            level: dict(counter) for level, counter in sorted(evaluator_by_level.items())
        },
        "blocked_vs_admitted": {
            "noop": by_result.get("system_one_curiosity_noop", 0),
            "admit": by_result.get("system_one_curiosity_admit", 0),
            "unavailable_fallback": by_result.get("system_one_unavailable_fallback", 0),
        },
        "note": (
            "Compare evaluator invoke/noop rates by level against historical "
            "pre-promotion invoke volume. If level-0 ticks historically produced "
            "valuable investigations, recalibrate the question — do not add "
            "coefficients. Downstream finding/prior-update joins remain future work "
            "when those outcome ids are present on gate_json."
        ),
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
                SELECT frame_json, generated_at
                FROM substrate_system_one_appraisal
                WHERE generated_at >= :cutoff
                ORDER BY generated_at ASC
                LIMIT :limit
                """
            ),
            {"cutoff": cutoff, "limit": max(1, int(args.limit))},
        ).mappings().all()

        gate_rows: list[dict[str, Any]] = []
        try:
            gate_rows = [
                dict(row)
                for row in conn.execute(
                    text(
                        """
                        SELECT candidate_set_id, generated_at, gate_json
                        FROM substrate_endogenous_curiosity_candidates
                        WHERE generated_at >= :cutoff
                          AND gate_json IS NOT NULL
                        ORDER BY generated_at ASC
                        LIMIT :limit
                        """
                    ),
                    {"cutoff": cutoff, "limit": max(1, int(args.limit))},
                ).mappings().all()
            ]
        except Exception as exc:
            gate_rows = []
            gate_error = str(exc)
        else:
            gate_error = None

    report = evaluate([dict(row) for row in rows])
    report["curiosity_gates"] = evaluate_curiosity_gates(gate_rows)
    if gate_error:
        report["curiosity_gates"]["load_error"] = gate_error
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Turn stored human ratings into a belief about what each action produces.

    scripts/score_artifact_ratings.py            # score what is unscored
    scripts/score_artifact_ratings.py --report   # show the beliefs, write nothing

This is the consumer half. Without it the rating path is a contract, a
producer and two empty tables -- which is exactly the "schema with no
consumer" this repo's contract bans, and it was a fair hit in review.

WHAT IT DOES
    chat_response_feedback (artifact-targeted, unscored)
      -> resolve dispatch_id -> (dispatch_kind, target_id)
      -> orion.autonomy.rating.score_rating against the current posterior
      -> substrate_action_ratings + substrate_action_rating_posterior

RUN IT PERIODICALLY, NOT PER TICK. Ratings arrive a few times a day at best;
a per-tick reader would be 30,000 empty scans a day for a handful of rows.

IDEMPOTENT. Scoring is keyed on feedback_id (unique index), and a rating whose
row already exists is skipped rather than re-absorbed. An observation counted
twice corrupts the belief permanently, and unlike the pressure ledger there is
no control arm here to make the corruption visible by contrast.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orion.autonomy.prediction import EffectPosterior  # noqa: E402
from orion.autonomy.rating import (  # noqa: E402
    cold_rating_prior,
    cold_start_surprise_nats,
    score_rating,
)
from orion.schemas.chat_response_feedback import parse_artifact_ref  # noqa: E402

DEFAULT_DSN = os.environ.get(
    "ORION_PG_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney"
)

UNSCORED = """
SELECT f.feedback_id, f.target_artifact_ref, f.feedback_value, f.categories,
       f.free_text, f.user_id, f.source, f.created_at
  FROM chat_response_feedback f
  LEFT JOIN substrate_action_ratings r ON r.feedback_id = f.feedback_id
 WHERE f.target_artifact_ref IS NOT NULL
   AND r.feedback_id IS NULL
 ORDER BY f.created_at
"""

# Resolves an action id to what the action actually was. Only actions that
# declared a signal appear here, which as of 2026-08-21 is 32.3% of dispatch
# volume -- so a rating of an undeclared action's artifact cannot be resolved
# and is reported, never guessed. Widening that means indexing dispatch frames
# by dispatch_id, which is a separate patch.
RESOLVE = """
SELECT dispatch_kind, target_id
  FROM substrate_action_outcomes
 WHERE dispatch_id = :dispatch_id
 ORDER BY observed_at DESC
 LIMIT 1
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--report", action="store_true", help="read-only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from sqlalchemy import create_engine, text

    engine = create_engine(args.dsn)

    if args.report:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT dispatch_kind, target_id, posterior_mean,
                           posterior_variance, posterior_n, unrated_count
                      FROM substrate_action_rating_posterior
                     ORDER BY posterior_n DESC
                    """
                )
            ).all()
        if not rows:
            print("No rated actions yet.")
            return 0
        print(f"{'kind':<12} {'target':<30} {'mean':>8} {'+/-':>7} {'n':>4} {'unrated':>8}")
        print("-" * 74)
        for kind, target, mean, var, n, unrated in rows:
            print(
                f"{kind:<12} {target:<30.30} {mean:>8.3f} {var ** 0.5:>7.3f} "
                f"{n:>4} {unrated:>8}"
            )
        print(
            f"\nmean is on [-1, +1]. Divide surprise by {cold_start_surprise_nats():.4f} "
            "before comparing with the pressure ledger -- same unit, different scale."
        )
        return 0

    skipped: Counter[str] = Counter()
    scored = 0

    with engine.begin() as conn:
        pending = conn.execute(text(UNSCORED)).mappings().all()
        for row in pending:
            try:
                _kind, dispatch_id = parse_artifact_ref(row["target_artifact_ref"])
            except ValueError:
                # Cannot happen through the model, which validates on
                # construction -- but this table predates that validator and a
                # row could have been written by hand.
                skipped["malformed_ref"] += 1
                continue

            resolved = conn.execute(
                text(RESOLVE), {"dispatch_id": dispatch_id}
            ).first()
            if resolved is None:
                # Reported, never guessed. Attributing a human rating to the
                # wrong action is worse than not attributing it.
                skipped["unresolvable_dispatch"] += 1
                continue
            dispatch_kind, target_id = resolved

            prior_row = conn.execute(
                text(
                    """
                    SELECT posterior_mean, posterior_variance, posterior_n
                      FROM substrate_action_rating_posterior
                     WHERE dispatch_kind = :k AND target_id = :t
                    """
                ),
                {"k": dispatch_kind, "t": target_id},
            ).first()
            prior = (
                EffectPosterior(float(prior_row[0]), float(prior_row[1]), int(prior_row[2]))
                if prior_row
                else cold_rating_prior()
            )

            result = score_rating(
                artifact_ref=row["target_artifact_ref"],
                dispatch_id=dispatch_id,
                dispatch_kind=dispatch_kind,
                target_id=target_id,
                feedback_value=row["feedback_value"],
                categories=list(row["categories"] or []),
                free_text=row["free_text"],
                rated_at=row["created_at"],
                prior=prior,
            )

            if args.dry_run:
                print(
                    f"would score {result.feedback_value:<4} "
                    f"{dispatch_kind}/{target_id} -> mean {result.posterior_mean:+.3f} "
                    f"({result.surprise_nats:.4f} nats)"
                )
                scored += 1
                continue

            inserted = conn.execute(
                text(
                    """
                    INSERT INTO substrate_action_ratings (
                        feedback_id, artifact_ref, dispatch_id, dispatch_kind,
                        target_id, feedback_value, rating, categories, free_text,
                        predicted_rating, prediction_error, surprise_nats,
                        posterior_mean, posterior_variance, posterior_n, rated_at
                    ) VALUES (
                        :feedback_id, :artifact_ref, :dispatch_id, :dispatch_kind,
                        :target_id, :feedback_value, :rating, :categories, :free_text,
                        :predicted_rating, :prediction_error, :surprise_nats,
                        :posterior_mean, :posterior_variance, :posterior_n, :rated_at
                    )
                    ON CONFLICT (feedback_id) DO NOTHING
                    RETURNING id
                    """
                ),
                {
                    "feedback_id": row["feedback_id"],
                    "artifact_ref": result.artifact_ref,
                    "dispatch_id": result.dispatch_id,
                    "dispatch_kind": result.dispatch_kind,
                    "target_id": result.target_id,
                    "feedback_value": result.feedback_value,
                    "rating": result.rating,
                    "categories": list(result.categories),
                    "free_text": result.free_text,
                    "predicted_rating": result.predicted_rating,
                    "prediction_error": result.prediction_error,
                    "surprise_nats": result.surprise_nats,
                    "posterior_mean": result.posterior_mean,
                    "posterior_variance": result.posterior_variance,
                    "posterior_n": result.posterior_n,
                    "rated_at": result.rated_at,
                },
            ).first()
            if inserted is None:
                # Raced, or already scored. The belief must not advance.
                skipped["already_scored"] += 1
                continue

            conn.execute(
                text(
                    """
                    INSERT INTO substrate_action_rating_posterior (
                        dispatch_kind, target_id, posterior_mean,
                        posterior_variance, posterior_n, updated_at
                    ) VALUES (:k, :t, :mean, :var, :n, now())
                    ON CONFLICT (dispatch_kind, target_id) DO UPDATE SET
                        posterior_mean = EXCLUDED.posterior_mean,
                        posterior_variance = EXCLUDED.posterior_variance,
                        posterior_n = EXCLUDED.posterior_n,
                        updated_at = now()
                     WHERE substrate_action_rating_posterior.posterior_n
                           < EXCLUDED.posterior_n
                    """
                ),
                {
                    "k": dispatch_kind,
                    "t": target_id,
                    "mean": result.posterior_mean,
                    "var": result.posterior_variance,
                    "n": result.posterior_n,
                },
            )
            scored += 1

    print(f"scored {scored} rating(s) from {len(pending)} unscored")
    for reason, count in sorted(skipped.items()):
        print(f"  skipped {count}: {reason}")
    if skipped["unresolvable_dispatch"]:
        print(
            "  (unresolvable = the action declared no signal, so it has no "
            "ledger row to resolve against. 67.7% of dispatch volume is in "
            "that state.)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

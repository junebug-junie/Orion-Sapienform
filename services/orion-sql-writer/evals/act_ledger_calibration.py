"""Act ledger calibration eval (rung 2).

Question: when acts close through the binary EffectPosterior, does the
``predicted`` stamped on the NEXT act track the real rate of the outcome, or
does the prior / variance choice bias it? Replays an ordered 0/1 outcome
sequence through orion.autonomy.prediction.score_act_outcome exactly as
orion/grammar/act_ledger.py does (predict = current posterior mean, then update)
and reports:

  final_mean   posterior mean after the whole sequence
  empirical    plain outcome rate
  brier        mean (predicted - outcome)^2 of the stamped predictions
  base_brier   the same for an always-0.5 predictor (the cold prior)

Run modes:
  python act_ledger_calibration.py              # bundled fixtures (deterministic)
  python act_ledger_calibration.py --live       # read-only: rebuild say/ask outcomes
                                                # from live chat_history_log (POSTGRES_URI)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.autonomy.prediction import cold_act_posterior, score_act_outcome  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "act_outcome_sequences.json"


def replay(outcomes: list[int]) -> dict:
    belief = cold_act_posterior()
    sq = base_sq = 0.0
    for o in outcomes:
        predicted = belief.mean
        sq += (predicted - o) ** 2
        base_sq += (0.5 - o) ** 2
        belief, _, _ = score_act_outcome(belief, float(o))
    n = len(outcomes)
    return {
        "n": n,
        "empirical": round(sum(outcomes) / n, 4) if n else None,
        "final_mean": round(belief.mean, 4),
        "brier": round(sq / n, 4) if n else None,
        "base_brier": round(base_sq / n, 4) if n else None,
    }


# Same predicates the resolver uses (orion.regulation.juniper_turns), read-only.
_LIVE_SQL = {
    "say:hub.chat_reply (1800s)": """
        SELECT (EXISTS (SELECT 1 FROM chat_history_log c
                         WHERE NULLIF(btrim(c.prompt), '') IS NOT NULL
                           AND COALESCE(c.client_meta->>'unsolicited','false') <> 'true'
                           AND c.created_at > t.created_at
                           AND c.created_at <= t.created_at + interval '1800 seconds'
                           AND coalesce(c.correlation_id,'') <> coalesce(t.correlation_id,'')))::int
          FROM chat_history_log t
         WHERE NULLIF(btrim(t.prompt), '') IS NOT NULL
           AND COALESCE(t.client_meta->>'unsolicited','false') <> 'true'
           AND t.created_at > now() - interval '60 days'
           AND t.created_at < now() - interval '1800 seconds'
         ORDER BY t.created_at""",
    "say/ask:hub.outreach (36000s)": """
        SELECT (EXISTS (SELECT 1 FROM chat_history_log c
                         WHERE NULLIF(btrim(c.prompt), '') IS NOT NULL
                           AND COALESCE(c.client_meta->>'unsolicited','false') <> 'true'
                           AND c.created_at > t.created_at
                           AND c.created_at <= t.created_at + interval '36000 seconds'))::int
          FROM chat_history_log t
         WHERE t.client_meta->>'unsolicited' = 'true'
           AND t.created_at > now() - interval '60 days'
           AND t.created_at < now() - interval '36000 seconds'
         ORDER BY t.created_at""",
}


def live_sequences() -> dict[str, list[int]]:
    import psycopg2

    uri = os.environ["POSTGRES_URI"]
    out: dict[str, list[int]] = {}
    with psycopg2.connect(uri) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            for name, sql in _LIVE_SQL.items():
                cur.execute(sql)
                out[name] = [int(r[0]) for r in cur.fetchall()]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)
    seqs = live_sequences() if args.live else json.loads(FIXTURE.read_text())
    report = {name: replay(seq) for name, seq in seqs.items()}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Kill-criterion eval: does the ask_claude trigger actually discriminate?

The criterion is falsifiable and stated before the data is read, which is the
whole point -- `MIN_TIMES_TESTED` and `MAX_SETTLED_CONFIDENCE` in
`orion/autonomy/ask_claude_trigger.py` are uncalibrated knobs, and a trigger
whose knobs never select anything (or always select everything) is ornamental.
The contested-scarcity spec names that failure directly: "If a week of real
spend never produces a refusal, the allowance is set too high or the action is
too cheap to matter, and wiring it in would ship the same ornamental scarcity
the 2026-07-07 spec was correctly refused for."

Reads the dry-run log this feature produces
(`~/.orion/ask-claude-dry-run.jsonl`, written by the 30-minute systemd timer
around `scripts/report_ask_claude_dry_run.py --json`). No database, no network.

THE TWO SIDES ARE JUDGED SEPARATELY, AND THAT IS DELIBERATE. A run where the
budget refused every time tells you nothing about whether the prior side would
ever have fired -- which is exactly why `decide()` scores priors even on a
refused tick. Collapsing both into one PASS/FAIL would let a week of
`budget_unobserved` (a broken transcript mount) read as "the trigger is dead"
when the trigger was never consulted.

  PRIOR SIDE   PASS when the stuck-count is neither always 0 nor always N,
               judged over the runs that actually saw priors. Both extremes
               mean the knobs carry no information. Outage ticks (worldview
               unreachable) are EXCLUDED rather than counted against either
               mode -- counting them in let one FalkorDB blip disarm the
               criterion entirely.
  BUDGET SIDE  Reported, never failed on. A week of `clear` is a real fact
               about how contended the pool is, not a defect in this trigger.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_LOG = Path.home() / ".orion" / "ask-claude-dry-run.jsonl"
# Below this the answer is "not enough data yet", never PASS or FAIL. At the
# timer's 30-minute cadence this is about a day.
MIN_RUNS = 48


def _load(path: Path) -> tuple[list[dict], int]:
    """Returns (decisions, skipped). A truncated final line is normal for a log
    being appended to right now, so a bad line is skipped and counted rather
    than aborting the eval."""
    decisions: list[dict] = []
    skipped = 0
    try:
        text = path.read_text()
    except OSError:
        return [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            decisions.append(json.loads(line)["decision"])
        except Exception:  # noqa: BLE001
            skipped += 1
    return decisions, skipped


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--min-runs", type=int, default=MIN_RUNS)
    args = ap.parse_args(argv)

    path = Path(args.log)
    print("\n=== ask_claude trigger discrimination eval ===")
    print(f"log: {path}")

    decisions, skipped = _load(path)
    if skipped:
        print(f"unparseable lines skipped: {skipped}")
    if len(decisions) < args.min_runs:
        print(f"runs observed: {len(decisions)} (need >= {args.min_runs})")
        print("RESULT: insufficient data")
        return 0

    print(f"runs observed: {len(decisions)}")

    # -- budget side: reported, never failed on ---------------------------
    refusals = Counter(d.get("refused") for d in decisions if not d.get("would_ask"))
    states = Counter(d.get("limit_state") for d in decisions)
    print("\nbudget side (reported, not a pass/fail criterion)")
    print(f"  limit_state       : {dict(states)}")
    print(f"  refusal reasons   : {dict(refusals)}")
    print(f"  would_ask=True    : {sum(1 for d in decisions if d.get('would_ask'))}")

    # -- prior side: the actual criterion ---------------------------------
    stuck_counts: list[int] = []
    totals: list[int] = []
    subjects: Counter = Counter()
    for d in decisions:
        assessments = d.get("assessments") or []
        totals.append(len(assessments))
        stuck_counts.append(sum(1 for a in assessments if a.get("stuck")))
        if d.get("subject_prior_id"):
            subjects[d["subject_prior_id"]] += 1

    # JUDGE ONLY THE RUNS THAT ACTUALLY SAW PRIORS.
    #
    # An outage tick (worldview unreachable, `t == 0`) is not evidence about
    # the knobs, and mixing it in defeats the criterion outright: the original
    # `all(c == t and t > 0 ...)` treated ONE zero-prior run as evidence
    # against the always-all mode, so 59 runs at 7-of-7 plus a single FalkorDB
    # blip reported PASS. Verified in review. FalkorDB restarts are a
    # documented reality in worldview.py, so one blip in a week would have
    # silently disarmed the whole eval.
    real = [(c, t) for c, t in zip(stuck_counts, totals) if t > 0]
    outages = len(totals) - len(real)
    if outages:
        print(f"\nruns with no readable priors (excluded from the criterion): {outages}")
    if len(real) < args.min_runs:
        # A PASS must not rest on a handful of real data points while most of
        # the window was outage.
        print(f"runs with readable priors: {len(real)} (need >= {args.min_runs})")
        print("RESULT: insufficient data")
        return 0

    real_stuck = [c for c, _ in real]
    real_totals = [t for _, t in real]
    always_none = all(c == 0 for c in real_stuck)
    always_all = all(c == t for c, t in real)
    passed = not (always_none or always_all)

    print("\nprior side (the criterion)")
    print(f"  runs judged         : {len(real)}")
    print(f"  live priors per run : min={min(real_totals)} max={max(real_totals)}")
    print(f"  stuck per run       : min={min(real_stuck)} max={max(real_stuck)}")
    print(f"  distinct subjects   : {dict(subjects)}")
    if always_none:
        print("  FAILURE MODE: the knobs never selected a prior. Too strict, or no prior "
              "ever gets stuck -- either way the trigger cannot fire and must not be armed.")
    if always_all:
        print("  FAILURE MODE: the knobs selected every prior on every run. No "
              "discrimination -- this is 'fire always', which is what the pinned "
              "sub-concept-seed-claude signal was rejected for.")

    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(run())

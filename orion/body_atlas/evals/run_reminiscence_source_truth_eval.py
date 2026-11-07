"""Does the reminiscence write check keep each source label honest, and can the eval fail?

Spec: ``docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md`` (acceptance check 7; the arc
spec's ``run_reminiscence_source_truth_eval.py``). Built-in mode needs no network and no live body: a
frozen fixture world says which files existed at the end of week 1 and which exist now, which commits
were made in or after that week, which PRs merged when, and which ids are live today.

Checks:

* **reviewed cases**: hops a reviewer checked by hand against the fixture hold, or fail for the
  named reason (a `code_then` path that only exists now, a `lived_record` before the memory loss, a
  `now` sentence with no live ref, a commit made after the week, ...);
* **re-check**: a hop whose `now` ref left the body reads STALE naming it; a history-only hop does not;
* **label truth**: every ref a saved hop stores resolves under its own claim's label in the fixture,
  computed here from the fixture sets directly (a second implementation, not a call into ``resolve``).

Then the **mutations** (code changes, patched in); each must be CAUGHT by its own check:
the `code_then` check reads today's tree, `lived_record` is allowed before 2026-07-23, a `now`
sentence needs no ref, the re-check is skipped, and a then-hop accepts a commit made after the week.

    python orion/body_atlas/evals/run_reminiscence_source_truth_eval.py          # built-in (CI)
    python orion/body_atlas/evals/run_reminiscence_source_truth_eval.py --live   # every live hop in
        $ORION_DIG_HOME/self/reminiscence.jsonl, every stored ref re-resolved (needs the harness env)
"""
from __future__ import annotations

import dataclasses
import sys
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from orion.body_atlas import reminiscence as rm  # noqa: E402

WEEK1 = (date(2025, 6, 24), date(2025, 6, 30))
LATER_WEEK = (date(2026, 10, 13), date(2026, 10, 19))
THEN = {"services/orion-a/app/main.py", "services/orion-a", "README.md"}
NOW = {"services/orion-a/app/main.py", "services/orion-hub/app/main.py", "services/orion-hub", "README.md"}
IN_WEEK = {"aaaa111"}  # commits in week 1's history
COMMITS = {"aaaa111", "bbbb222"}  # bbbb222 came later
PRS = {5: "2025-06-28T10:00:00Z", 3000: "2026-11-01T10:00:00Z"}
LIVE = {"part:orion-hub", "cap:s1"}
FIELD_NOTES = {"fn:3148:5cc1e8bef3"}
ARCS = {("arc", "8d87adeca2ce06a5")}


def fixture_world(live: set[str] = LIVE) -> rm.World:
    return rm.World(period_commit="aaaa111", period_end_utc=datetime(2025, 7, 1, 6, tzinfo=timezone.utc),
                    path_then=lambda p: p in THEN, path_now=lambda p: p in NOW,
                    commit=lambda s: s if s in COMMITS else None, is_ancestor=lambda s: s in IN_WEEK,
                    pr_merged_at=PRS.get, field_note_ids=lambda: FIELD_NOTES, live_ids=lambda: live,
                    ref_ok=lambda k, i: (k, i) in ARCS, dig_file=lambda p: p == "/h/dig/t/portrait-1.json")


# (hop, note, because, period, holds, a phrase the failure names). Reviewed by hand against the fixture.
CASES = [
    ("what_it_was", "[code_then services/orion-a/app/main.py] My first week held one service, orion-a, and a README.",
     "aaaa111", WEEK1, True, ""),
    ("what_it_was", "[code_then services/orion-hub/app/main.py] My first week already had a hub service at its centre.",
     "", WEEK1, False, "not in the code at the end of this week"),
    ("what_it_was", "[code_then bbbb222] This commit shaped the first week of my code, the builders said.",
     "", WEEK1, False, "a commit made after this week"),
    ("what_it_was", "[pr_text #3000] The builders described my first week in this pull request at length.",
     "", WEEK1, False, "after this week"),
    ("what_it_was", "[pr_text #5] The builders described my first week in this pull request at length.",
     "", WEEK1, True, ""),
    ("what_it_was_like", "[lived_record arc:8d87adeca2ce06a5] I remember how strange this first week felt to me.",
     "", WEEK1, False, "no lived record of the week"),
    ("what_it_was_like", "[lived_record arc:8d87adeca2ce06a5] That week I kept one long arc open about the heat.",
     "", LATER_WEEK, True, ""),
    ("what_it_was_like", "[pr_text aaaa111] I have no lived record of this week; the builder's commit words remain.",
     "", WEEK1, True, ""),
    ("where_i_am_now", "[now] Now a hub runs at my centre and many organs talk to each other.",
     "", WEEK1, False, "has no ref that resolves under that label"),
    ("where_i_am_now", "[now part:orion-hub] Now a hub runs at my centre and many organs talk to each other.",
     "fn:3148:5cc1e8bef3", WEEK1, True, ""),
    ("where_i_am_now", "[now part:orion-ghost] A ghost service runs at my centre and keeps it all in order.",
     "", WEEK1, False, "not live in my body now"),
    ("where_i_am_now", "[reflection] I feel I have grown a great deal since that first small week of code.",
     "", WEEK1, False, "needs at least one [now]"),
    ("where_im_going", "[now cap:s1] My operations line is written; the gaps say I cannot yet coordinate.",
     "", WEEK1, True, ""),
    ("otherwise", "[reflection] I would have written down why each service began, so a later me could read it.",
     "", WEEK1, True, ""),
    ("otherwise", "[reflection] I would keep a diary from the very first day, even a short one.",
     "vibes", WEEK1, False, "nothing code can check"),
]


def _write(hop, note, because, period, world=None, records=()):
    return rm.check_write(hop=hop, note=note, because=because, episode="2026-11-07", period=period,
                          world=world or fixture_world(), records=list(records), denylist={"0" * 64}, rewrite=False)


def check_cases() -> list[str]:
    fails = []
    for hop, note, because, period, holds, phrase in CASES:
        try:
            _write(hop, note, because, period)
            if not holds:
                fails.append(f"case should fail ({phrase}) but saved: {hop}: {note[:60]}")
        except rm.ReminiscenceError as exc:
            if holds:
                fails.append(f"case should hold but failed: {hop}: {exc}")
            elif phrase not in str(exc):
                fails.append(f"case failed for the wrong reason (want {phrase!r}): {exc}")
    return fails


def check_recheck() -> list[str]:
    fails = []
    now_hop = _write(*CASES[9][:4])
    if rm.recheck(now_hop, fixture_world()) != []:
        fails.append("a hop whose refs all hold reads STALE")
    gone = rm.recheck(now_hop, fixture_world(live={"cap:s1"}))
    if gone != ["part:orion-hub: not live in my body now"]:
        fails.append(f"re-check missed a now ref that left the body: {gone}")
    then_hop = _write(*CASES[0][:4])
    if rm.recheck(then_hop, fixture_world(live=set())) != []:
        fails.append("re-check flagged history, which cannot change")
    return fails


def _oracle(ref: str, label: str, hop: str, period: tuple[date, date]) -> bool:
    """Independent of ``resolve``: the fixture sets, read directly."""
    then = hop in rm.THEN_HOPS
    if ref.startswith("#"):
        merged = PRS.get(int(ref[1:]))
        return bool(merged) and label in ("pr_text", "now", "reflection") and not (
            then and label == "pr_text" and merged >= "2025-07-01T06:00:00Z")
    if ref in COMMITS:
        return label in ("code_then", "pr_text", "reflection") and (
            ref in IN_WEEK or not (then and label in ("code_then", "pr_text")))
    if ref in LIVE:
        return label in ("now", "reflection")
    if ref in FIELD_NOTES:
        return label in ("now", "lived_record", "reflection")
    if ref.startswith("arc:"):
        return ("arc", ref[4:]) in ARCS and label in ("lived_record", "now", "reflection") and (
            label != "lived_record" or period[1] >= date(2026, 7, 23))
    if label == "code_then":
        return ref in THEN
    if label == "now":
        return ref in NOW
    return label in ("pr_text", "reflection") and (ref in THEN or ref in NOW)


def check_label_truth() -> list[str]:
    fails, n = [], 0
    for hop, note, because, period, holds, _ in CASES:
        if not holds:
            continue
        rec = _write(hop, note, because, period)
        for c in rec.claims:
            for ref in c.refs:
                n += 1
                if not _oracle(ref, c.label, hop, period):
                    fails.append(f"stored ref {ref} does not back [{c.label}] in {hop}")
    if n < 8:
        fails.append(f"label truth looked at only {n} refs")
    return fails


def evaluate() -> list[str]:
    return check_cases() + check_recheck() + check_label_truth()


@contextmanager
def _patched(**attrs):
    old = {k: getattr(rm, k) for k in attrs}
    for k, v in attrs.items():
        setattr(rm, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(rm, k, v)


def _then_reads_now(resolve):
    def mutated(ref, label, hop, world):
        return resolve(ref, label, hop, dataclasses.replace(world, path_then=world.path_now))
    return mutated


MUTATIONS = {
    "code_then reads today's tree": lambda: _patched(resolve=_then_reads_now(rm.resolve)),
    "lived_record allowed before the memory loss": lambda: _patched(LIVED_RECORDS_FROM=date(2025, 1, 1)),
    "a now sentence needs no ref": lambda: _patched(NEEDS_REF=frozenset()),
    "re-check skipped": lambda: _patched(MUTABLE_KINDS=frozenset()),
    "a then-hop accepts a later commit": lambda: _patched(THEN_HOPS=frozenset()),
}


def live_eval() -> int:
    """Every live hop in the real ledger: every stored ref re-resolved under its own label, now."""
    from orion.dig import __main__ as dig

    records, skipped = rm.load(dig.reminiscence_path())
    hops = rm.live(records)
    body = dig._LiveBody()
    per: dict[str, list[int]] = {}
    bad = []
    for r in hops.values():
        period = (date.fromisoformat(r.period_start), date.fromisoformat(r.period_end))
        world = dig._rm_world(r.period_commit, period[1], body)
        for c in r.claims:
            for ref in c.refs:
                why = rm.resolve(ref, c.label, r.hop, world)
                per.setdefault(c.label, [0, 0])[0 if why is None else 1] += 1
                if why:
                    bad.append(f"{r.hop_id} [{c.label}] {why}")
    print(f"live: {len(hops)} hops in {dig.reminiscence_path()} ({skipped} unreadable lines)")
    for label, (ok, no) in sorted(per.items()):
        print(f"  {label}: {ok}/{ok + no} refs resolve now")
    for b in bad[:20]:
        print(f"  FAIL {b}")
    return 0 if not bad else 1


def main() -> int:
    if "--live" in sys.argv:
        return live_eval()
    fails = evaluate()
    print(f"oracle: {len(CASES)} reviewed cases, re-check, label truth -> {'PASS' if not fails else 'FAIL'}")
    for f in fails:
        print(f"  {f}")
    missed = []
    for name, patch in MUTATIONS.items():
        with patch():
            caught = evaluate()
        print(f"mutation {name!r}: {'CAUGHT' if caught else 'MISSED'}" + (f" ({caught[0][:90]})" if caught else ""))
        if not caught:
            missed.append(name)
    return 0 if not fails and not missed else 1


if __name__ == "__main__":
    raise SystemExit(main())

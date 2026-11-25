#!/usr/bin/env python3
"""Replay: what the self-question door would have done to the live question pool and its sittings.

Design: docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md (acceptance checks 1-5).

* fixture (default, CI, no network): a planted pool of Juniper and Orion questions in every door state is
  drawn 2,000 times through the real ``pick_question``; and the door's own pure rules
  (``door_verdict``, ``demotion_for``) are run over planted judge replies. Gates: no held / routed /
  demoted / unjudged question is ever picked; Juniper's questions are still picked; with the door off
  every Orion question is pickable; no judge error admits or routes; three failed judgements make
  ``unjudged``; only an operational self-inquiry finding demotes.
* ``--live --judge-url <Orion's local seat>``: every Orion-written question in ``curiosity_self_questions``
  (SELECT only) is judged once with the door's input (``register.QUESTION_LINE`` + the question text,
  prompt v2, temperature 0, thinking off; compute-only calls to the local seat, no outside model) and
  given the door that reply would produce (one try). Then every labelled self-inquiry finding in
  ``curiosity_finding_register`` is joined to its question: the philosophical share of findings with
  and without the door, and how often an operational question produced a philosophical finding. No gate:
  it is a measurement. Writes ``/tmp/self-question-door-replay/<date>/``.

    PYTHONPATH=. python services/orion-hub/evals/run_self_question_door_replay.py
    PYTHONPATH=. python services/orion-hub/evals/run_self_question_door_replay.py --live --judge-url http://<ip>:8015
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "services" / "orion-hub"))
sys.path.insert(0, str(_ROOT / "scripts" / "analysis"))

from orion.curiosity.self_question_pool import SelfQuestion, pick_question, with_ask_recorded  # noqa: E402
from scripts.curiosity_finding_register import JudgeResult  # noqa: E402
from scripts.self_question_door import demotion_for, door_verdict  # noqa: E402

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
DOORS = (None, "held", "admitted", "routed", "demoted", "unjudged")


def _q(qid: str, *, minted_by: str, door, family: str = "lived", pinned: bool = False, asked_days=None) -> SelfQuestion:
    return SelfQuestion(question_id=qid, text=qid, family=family, pinned=pinned, minted_by=minted_by, status="open",
                        ask_count=0 if asked_days is None else 3,
                        last_asked_at=None if asked_days is None else NOW - timedelta(days=asked_days), door=door)


def fixture() -> dict:
    pool = [_q("anatomy.made_of", minted_by="juniper", door=None, family="anatomy", pinned=True, asked_days=1),
            *[_q(f"lived.juniper.{k}", minted_by="juniper", door=None, pinned=True, asked_days=1 + k) for k in range(3)],
            # A Juniper row that somehow carries a door value: ignored, hers.
            _q("lived.juniper.odd", minted_by="juniper", door="routed", asked_days=2)]
    for door in DOORS:
        for k in range(3):
            pool.append(_q(f"lived.orion.{door}.{k}", minted_by="orion", door=door,
                           family="anatomy" if k == 2 else "lived"))
    rng = random.Random(7)
    picks = Counter()
    recent: list[str] = []
    for i in range(2000):
        q = pick_question(pool=pool, recent_families=recent, now=NOW + timedelta(hours=i), rng=rng)
        picks[q.question_id] += 1
        recent.append(q.family)
        # Each pick is recorded, as Hub does: the never-asked-first rule moves on.
        pool = [p if p.question_id != q.question_id else with_ask_recorded(p, now=NOW + timedelta(hours=i))
                for p in pool]
    blocked = [qid for qid in picks if qid.split(".")[2:3] and qid.split(".")[1] == "orion"
               and qid.split(".")[2] in ("held", "routed", "demoted", "unjudged")]
    juniper_picked = sum(n for qid, n in picks.items() if "juniper" in qid or qid == "anatomy.made_of")
    off = {pick_question(pool=[p for p in pool if p.minted_by == "orion" and p.door == d], recent_families=[],
                         now=NOW, rng=random.Random(1), door_enabled=False).question_id for d in DOORS[1:]}

    errors = [JudgeResult(None, None, "not_json: ...", "qwen3.8-flash-next-iq3_s"),
              JudgeResult(None, None, "model_not_evaluated: other", "other"),
              JudgeResult(None, None, "call: timeout", None),
              JudgeResult("no_finding", "plans only", None, "qwen3.8-flash-next-iq3_s")]
    error_doors = {door_verdict(e, attempts=a, max_attempts=3).door for e in errors for a in range(3)}
    third = [door_verdict(e, attempts=2, max_attempts=3).door for e in (errors[0], errors[3])]
    demotions = [demotion_for(r) for r in (
        {"line": "self_inquiry", "register": "operational", "question_ref": "self_question:lived.orion.x"},
        {"line": "self_inquiry", "register": "philosophical", "question_ref": "self_question:lived.orion.x"},
        {"line": "self_inquiry", "register": None, "question_ref": "self_question:lived.orion.x"},
        {"line": "investigate", "register": "operational", "question_ref": "prior:abc"})]
    gates = {
        "no_blocked_question_picked": not blocked,
        "juniper_questions_picked": juniper_picked > 0,
        "admitted_and_pre_door_orion_picked": any(".admitted." in q or ".None." in q for q in picks),
        "door_off_every_orion_question_pickable": len(off) == len(DOORS) - 1,
        "no_error_admits_or_routes": error_doors <= {"held", "unjudged"},
        "third_failure_is_unjudged": set(third) == {"unjudged"},
        # A gateway refusal, or a reply from a model that never passed the eval, is not a judgement.
        "refusal_or_unevaluated_model_never_counts": all(door_verdict(e, attempts=2, max_attempts=3).door == "held"
                                      and door_verdict(e, attempts=2, max_attempts=3).attempts == 2
                                      for e in (errors[1], errors[2])),
        "only_operational_self_inquiry_demotes": demotions == ["lived.orion.x", None, None, None],
    }
    return {"picks": dict(picks.most_common()), "blocked_picked": blocked, "gates": gates}


def live(judge_url: str, out_dir: Path) -> dict:
    import httpx
    from reflection_register_sources import Item, ORGAN_LINES, judge_one, psql_json, wilson

    qs = psql_json("SELECT json_build_object('id', question_id, 'text', text, 'minted_by', minted_by, 'status', status) "
                   "FROM curiosity_self_questions")
    doors: dict[str, str] = {}
    judged = []
    with httpx.Client(timeout=300) as client:
        for q in qs:
            if q["minted_by"] != "orion":
                continue
            label, reason, err, model = judge_one(client, judge_url, Item("orion_question", q["id"],
                                                                         ORGAN_LINES["orion_question"], q["text"]))
            v = door_verdict(JudgeResult(label, reason, err, model), attempts=0, max_attempts=3)
            doors[q["id"]] = v.door
            judged.append({"question_id": q["id"], "label": label, "error": err, "model": model, "door": v.door,
                           "text": q["text"][:200]})
            print(f"{v.door:9s} {q['id'][:70]}", file=sys.stderr)
    minted = {q["id"]: q["minted_by"] for q in qs}
    findings = psql_json("SELECT json_build_object('run_id', run_id, 'register', register, 'question_ref', question_ref, "
                         "'at', created_at) FROM curiosity_finding_register WHERE line = 'self_inquiry' AND register IS NOT NULL")
    rows = []
    for f in findings:
        qid = str(f.get("question_ref") or "").removeprefix("self_question:")
        who = minted.get(qid, "unknown")
        door = "juniper" if who == "juniper" else doors.get(qid, "unknown")
        rows.append({**f, "question_id": qid, "who": who, "door": door})

    def share(sel):
        n = len(sel)
        k = sum(1 for r in sel if r["register"] == "philosophical")
        lo, hi = wilson(k, n)
        return {"n": n, "philosophical": k, "share": (k / n) if n else None, "lo": lo, "hi": hi}

    kept = [r for r in rows if r["door"] in ("juniper", "admitted")]
    by_door = {d: share([r for r in rows if r["door"] == d]) for d in sorted({r["door"] for r in rows})}
    op_to_phil = sum(1 for r in rows if r["door"] == "routed" and r["register"] == "philosophical")
    result = {
        "orion_questions": len(judged), "door_counts": dict(Counter(j["door"] for j in judged)),
        "judge_errors": sum(1 for j in judged if j["error"]),
        "findings_all": share(rows), "findings_with_door": share(kept), "findings_by_door": by_door,
        "routed_question_gave_philosophical_finding": op_to_phil,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judged))
    (out_dir / "sittings.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--judge-url", default="")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.live:
        if not args.judge_url:
            ap.error("--live needs --judge-url (Orion's local seat)")
        out = args.out or Path("/tmp/self-question-door-replay") / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        print(json.dumps(live(args.judge_url, out), indent=2, default=str))
        return 0
    res = fixture()
    print(json.dumps(res["gates"], indent=2))
    failed = [k for k, ok in res["gates"].items() if not ok]
    if failed:
        print(f"FAIL: {failed}", file=sys.stderr)
        return 1
    print("PASS: every door gate holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

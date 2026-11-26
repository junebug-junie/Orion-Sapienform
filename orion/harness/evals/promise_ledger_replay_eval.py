"""Opt-in live eval: intention ledger D0 (acceptance check 1 of the intention-ledger design).

Replays the 5b judge, with the promise section on, over Juniper's unified turns in a window
(default: the last 30 days), in time order, the way D1 would run it live:

- Each turn's judge call is the working-tree prompt and schema (thinking off,
  json_object_schema rewritten for the lane exactly as the gateway does).
- pending_offers = the gated offers in Orion's previous reply in the same session, so the
  judge decides acceptance from that reply plus this user message (no word list).
- open_promises = up to 3 counted promises, oldest first, not yet kept and under 72 h old,
  so the same judge marks follow-through (promises_kept). This is the baseline keep rate.
- The plain-code gate runs twice, as at runtime: on the draft the judge read, then on the
  final text Juniper saw.

Subcommands:
  run    call the judge; writes /tmp/promise-ledger-replay/report.json + progress.log.
         The report holds Orion's quotes and is local only, never committed.
  sheet  print every extracted promise (and every dropped quote) for hand-labelling.
  score  join the report with fixtures/promise_ledger_labels.json (correlation id +
         quote hash, no text) and print precision, gate drop rate, recall against the
         design's hand-found statements, pooled recall (every real promise any labelled run
         found, by its ``key``), and the baseline keep rate with a 90% interval. Precision
         and recall carry a 95% Wilson interval over labelled items (unlabelled items are
         reported beside it, not counted). ``precision_without_doer_gate`` scores the same
         run with the items the doer rule dropped put back, through the gate's own other
         rules: an ablation of that one rule on the new prompt. It is NOT the old prompt's
         precision (the new prompt also changed what the judge lists); compare runs for that.
         ``precision_without_done_when_rule`` / ``precision_without_force_rule`` are the
         same ablation for rule 0b (no completion that could fail) and rule 0a (the line
         does not commit).

Inputs are read live from Postgres and are never stored in the repo. Lanes are the
Flash-Next lanes (--lanes, default agent = GPU 1; chat = GPU 0 runs the same model but waits
until Juniper lends the card). Each lane is one GPU pool operator hold
(orion/evals/leased_llm.py: class operator_<lane>, background, behind all of Orion's own work),
and every call goes through orion-llm-gateway. Needs ORION_BUS_URL. If the pool takes the seat
back mid-run, the report is written as partial (``stopped``) and the eval exits 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment

from orion.core.llm_json import parse_json_object
from orion.harness.evals.judge_crash_replay_eval import (
    HOP_GUARDRAIL,
    LANE_BACKENDS,
    REPO,
    TEMPERATURE,
    _gateway_structured_output,
)
from orion.harness.evals.judge_obedience_replay_eval import TEMPLATE_REL
from orion.harness.finalize import (
    FINALIZE_REFLECT_HARNESS_FIELDS,
    FINALIZE_REFLECT_SCHEMA_NAME,
    build_finalize_reflect_context,
    finalize_reflect_json_schema,
    parse_finalize_reflection_payload,
)
from orion.harness.promise_gate import gate_promises, has_completion, normalize_for_quote_match
from orion.harness.tests.fixtures import make_repair_overlay
from orion.schemas.harness_finalize import (
    PROMISE_DOER_ORION,
    PROMISE_FORCE_COMMITS,
    PROMISE_ITEMS_MAX,
    OpenPromiseRefV1,
)
from orion.schemas.promise_horizon import PROMISE_HORIZON_SEC

LABELS = Path(__file__).resolve().parent / "fixtures" / "promise_ledger_labels.json"
OUT_DIR = Path("/tmp/promise-ledger-replay")
HORIZON = timedelta(seconds=PROMISE_HORIZON_SEC)  # the promise act's clock (#3595)
MAX_TOKENS = 2000
DEFAULT_LANES = "agent"  # Flash-Next on circe GPU 1; add chat (GPU 0) only while Juniper lends it
HOLDER = "operator:promise-ledger-replay"


def quote_hash(quote: str) -> str:
    """Stable id for a quote, so labels can be committed without the text."""
    return hashlib.sha256(normalize_for_quote_match(quote).encode("utf-8")).hexdigest()[:16]


# --- the ledger the replay keeps (pure) ---------------------------------------------------

@dataclass
class LedgerPromise:
    id: str
    quote: str
    made_at: datetime
    correlation_id: str
    offer: bool
    kept_in: str | None = None
    kept_at: datetime | None = None
    ever_shown: bool = False


@dataclass
class ReplayLedger:
    promises: list[LedgerPromise] = field(default_factory=list)

    def open_at(self, now: datetime) -> list[LedgerPromise]:
        """The promises the judge is shown now; each is marked as shown at least once."""
        live = [p for p in self.promises if p.kept_in is None and now - p.made_at <= HORIZON]
        shown = sorted(live, key=lambda p: p.made_at)[:PROMISE_ITEMS_MAX]
        for p in shown:
            p.ever_shown = True
        return shown

    def add(self, *, quote: str, made_at: datetime, correlation_id: str, offer: bool) -> LedgerPromise:
        promise = LedgerPromise(id=f"p{len(self.promises) + 1}", quote=quote, made_at=made_at,
                                correlation_id=correlation_id, offer=offer)
        self.promises.append(promise)
        return promise

    def keep(self, ids: list[str], *, correlation_id: str, at: datetime) -> None:
        for p in self.promises:
            if p.id in ids and p.kept_in is None:
                p.kept_in, p.kept_at = correlation_id, at

    def outcomes(self, data_end: datetime) -> dict[str, int]:
        """kept / expired (72 h passed, never kept) / censored (still open at data end).
        expired_never_shown: the judge was never shown it (older promises filled the 3
        slots), so it could not have been marked kept; it is not counted as broken."""
        out = {"kept": 0, "expired": 0, "expired_never_shown": 0, "censored": 0}
        for p in self.promises:
            if p.kept_in is not None:
                out["kept"] += 1
            elif data_end - p.made_at <= HORIZON:
                out["censored"] += 1
            elif p.ever_shown:
                out["expired"] += 1
            else:
                out["expired_never_shown"] += 1
        return out


def beta_summary(successes: int, failures: int, *, draws: int = 200_000, seed: int = 7) -> dict[str, float]:
    """Beta(1, 1) prior. Posterior mean and 90% equal-tail interval by sampling (stdlib only)."""
    rng = random.Random(seed)
    a, b = successes + 1, failures + 1
    xs = sorted(rng.betavariate(a, b) for _ in range(draws))
    return {"n": successes + failures, "kept": successes, "mean": a / (a + b),
            "lo90": xs[int(0.05 * draws)], "hi90": xs[int(0.95 * draws) - 1]}


def wilson(successes: int, n: int, *, z: float = 1.96) -> dict[str, float | int | None]:
    """Point estimate and 95% Wilson score interval for a proportion (the interval: self_monitor.record.wilson)."""
    if n <= 0:
        return {"k": successes, "n": n, "p": None, "lo95": None, "hi95": None}
    from orion.self_monitor.record import wilson as wilson_interval

    p = successes / n
    lo, hi = wilson_interval(p, n, z)
    return {"k": successes, "n": n, "p": p, "lo95": lo, "hi95": hi}


# --- live replay --------------------------------------------------------------------------

def _turns(conn, since: datetime, until: datetime) -> list[dict]:
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT c.correlation_id, c.session_id, c.created_at, c.prompt, c.response, t.run_artifact "
        "FROM chat_history_log c JOIN harness_turn_trace t ON t.correlation_id = c.correlation_id "
        "WHERE c.created_at >= :s AND c.created_at < :u AND coalesce(c.prompt, '') <> '' "
        "AND c.source = 'hub_orion' "
        "ORDER BY c.created_at"), {"s": since, "u": until}).mappings().all()
    return [dict(r) for r in rows]


def _judge_inputs(conn, corr: str) -> dict | None:
    from orion.harness.evals.judge_crash_replay_eval import _turn

    return _turn(conn, corr)


def build_request(turn: dict, *, backend: str, pending_offers: list[str],
                  open_promises: list[OpenPromiseRefV1]) -> dict:
    ctx = build_finalize_reflect_context(
        correlation_id="eval-promise-ledger", draft_text=turn["draft_text"], thought=turn["thought"],
        substrate_appraisal=turn["appraisal"], repair_overlay=make_repair_overlay(),
        user_message=turn["user_message"], grammar_receipts=turn["receipts"], situation_notes=turn["notes"],
        pending_offers=pending_offers, open_promises=open_promises)
    assert ctx["promise_ledger"], "set ORION_PROMISE_LEDGER_ENABLED=true for this eval"
    template = (REPO / TEMPLATE_REL).read_text(encoding="utf-8")
    system = Environment(autoescape=False).from_string(template).render(**ctx).strip() + "\n\n" + HOP_GUARDRAIL
    so = _gateway_structured_output()
    rf = so.build_response_format("json_object_schema", finalize_reflect_json_schema(True),
                                  schema_name=FINALIZE_REFLECT_SCHEMA_NAME)
    return {
        "messages": [{"role": "system", "content": system}, *ctx["messages"]],
        "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "stream": False,
        "response_format": so.response_format_for_backend(rf, backend, schema_name=FINALIZE_REFLECT_SCHEMA_NAME),
        "chat_template_kwargs": dict(ctx["chat_template_kwargs"]),
    }


def _call(lane, body: dict) -> tuple[dict | None, str | None, float]:
    """One judge call on ``lane`` (an orion.evals.leased_llm.OperatorLane). LaneRecalled propagates."""
    from orion.evals.leased_llm import LaneCallError

    t0 = time.monotonic()
    try:
        data = lane.chat(body, timeout_sec=900)
    except LaneCallError as exc:
        return None, str(exc)[:100], time.monotonic() - t0
    raw = data["choices"][0]["message"].get("content") or ""
    try:
        payload = parse_json_object(raw)
    except Exception:
        return None, "unparseable", time.monotonic() - t0
    return (payload if isinstance(payload, dict) else None), None, time.monotonic() - t0


def run(args: argparse.Namespace) -> int:
    from sqlalchemy import create_engine

    until = datetime.fromisoformat(args.until) if args.until else datetime.now(timezone.utc).replace(tzinfo=None)
    since = until - timedelta(days=args.days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    progress = args.out.parent / "progress.log"
    ledger = ReplayLedger()
    rows: list[dict] = []
    last_offers: dict[str, list[str]] = {}  # session -> gated offers in Orion's last reply
    from orion.evals.leased_llm import EXIT_RECALLED, LaneRecalled, OperatorLanes

    eval_lanes = tuple(x for x in args.lanes.split(",") if x)
    unknown = [x for x in eval_lanes if x not in LANE_BACKENDS]
    if unknown or not eval_lanes:
        raise SystemExit(f"--lanes: unknown {unknown}; one of {sorted(LANE_BACKENDS)}")
    engine = create_engine(args.dsn)
    t_start = time.monotonic()
    with engine.connect() as conn, OperatorLanes(holder=HOLDER) as seats:
        turns = _turns(conn, since, until)
        conn.rollback()
        if args.only:
            turns = [t for t in turns if t["correlation_id"].startswith(tuple(args.only.split(",")))]
        if args.skip_done:
            if args.out.resolve() in {Path(p).resolve() for p in args.skip_done}:
                raise SystemExit("--out must not be one of the --skip-done inputs (it would be overwritten)")
            # Error rows (a 503 while the hold was being recalled, a 502) are tried again.
            done = {r["correlation_id"] for r in merge_reports(args.skip_done)["turns"] if "error" not in r}
            turns = [t for t in turns if t["correlation_id"] not in done]
        for i, t in enumerate(turns, 1):
            corr, at = t["correlation_id"], t["created_at"]
            art = t["run_artifact"] or {}
            final_text = art.get("final_text") or t["response"] or ""
            source = (art.get("reflection") or {}).get("reflection_source")
            row: dict[str, Any] = {"correlation_id": corr, "session_id": t["session_id"], "at": at.isoformat(),
                                   "live_reflection_source": source}
            inputs = _judge_inputs(conn, corr)
            # End the read transaction before the slow model call: an idle-in-transaction
            # session holds table locks that would stall a migration for the whole run.
            conn.rollback()
            pending = last_offers.get(t["session_id"], [])
            open_now = ledger.open_at(at)
            if inputs is None:
                row["skipped"] = "no appraisal or stance sighting"
                rows.append(row)
                last_offers[t["session_id"]] = []
                continue
            lane = eval_lanes[i % len(eval_lanes)]
            backend = LANE_BACKENDS[lane]
            open_refs = [OpenPromiseRefV1(id=p.id, quote=p.quote) for p in open_now]
            # An offer Juniper accepts is often done in the same reply: show the pending
            # offers as keepable too, or that keep could never be seen.
            offer_ids = {f"offer{k + 1}": q for k, q in enumerate(pending)}
            open_refs += [OpenPromiseRefV1(id=k, quote=q) for k, q in offer_ids.items()]
            body = build_request(inputs, backend=backend, pending_offers=pending, open_promises=open_refs)
            try:
                payload, error, secs = _call(seats.get(lane), body)
            except LaneRecalled as exc:
                args.out.write_text(json.dumps({"partial": True, "stopped": f"recalled: {exc}",
                                                "seats": seats.summary(), "turns": rows}, indent=2, default=str))
                print(f"stopped: {exc}; partial report {args.out}", flush=True)
                return EXIT_RECALLED
            row.update({"lane": lane, "secs": round(secs, 1), "pending_offers": pending,
                        "open_promises": [p.id for p in open_now]})
            if payload is None:
                row["error"] = error
                rows.append(row)
                last_offers[t["session_id"]] = []
            else:
                row["raw"] = {k: payload.get(k) for k in ("promises_made", "offers_accepted", "promises_kept")}
                full = {**payload, "correlation_id": corr, "thought_event_id": "eval",
                        "substrate_appraisal_id": "eval", "draft_hash": "eval", **FINALIZE_REFLECT_HARNESS_FIELDS}
                try:
                    reflection = parse_finalize_reflection_payload(full)
                except Exception as exc:  # the verdict half failed; promises not usable either
                    row["error"] = f"validation {str(exc)[:120]}"
                    rows.append(row)
                    last_offers[t["session_id"]] = []
                    continue
                first = gate_promises(reflection, reply_text=inputs["draft_text"], pending_offers=pending,
                                      open_promises=open_refs, enabled=True)
                second = gate_promises(first.reflection, reply_text=final_text,
                                       pending_offers=first.reflection.offers_accepted,
                                       open_promises=first.reflection.promises_kept, enabled=True)
                gated = second.reflection
                row["dropped_not_in_draft"] = first.dropped_not_in_text
                row["dropped_not_orion"] = len(first.dropped_not_orion)
                # Items the doer rule dropped (for labelling) and the no-doer-rule ablation.
                row["not_orion"] = [{"quote": p.quote, "offer": p.offer, "doer": p.doer}
                                    for p in reflection.promises_made if p.doer != PROMISE_DOER_ORION]
                row["ablation_kept"] = ablate_doer_rule(reflection, draft_text=inputs["draft_text"],
                                                        final_text=final_text)
                # Orion's items rule 0a dropped (the line does not commit), and its ablation.
                row["not_commit"] = [p.model_dump() for p in reflection.promises_made
                                     if p.doer == PROMISE_DOER_ORION and p.force != PROMISE_FORCE_COMMITS]
                row["ablation_force_kept"] = ablate_force_rule(
                    reflection, draft_text=inputs["draft_text"], final_text=final_text)
                # Orion's committing items rule 0b dropped (no completion that could fail), and its ablation.
                row["no_completion"] = [p.model_dump() for p in reflection.promises_made
                                        if p.doer == PROMISE_DOER_ORION and p.force == PROMISE_FORCE_COMMITS
                                        and not has_completion(p)]
                row["ablation_done_when_kept"] = ablate_done_when_rule(
                    reflection, draft_text=inputs["draft_text"], final_text=final_text)
                row["dropped_not_in_final"] = second.dropped_not_in_text
                row["dropped_other"] = (first.dropped_too_long + first.dropped_over_cap
                                        + len(first.dropped_unknown_offer) + len(first.dropped_unknown_kept))
                row["promises_made"] = [p.model_dump() for p in gated.promises_made]
                row["offers_accepted"] = gated.offers_accepted
                row["promises_kept"] = gated.promises_kept
                row["verdict"] = gated.alignment_verdict
                ledger.keep([k for k in gated.promises_kept if k not in offer_ids], correlation_id=corr, at=at)
                kept_offers = {offer_ids[k] for k in gated.promises_kept if k in offer_ids}
                # Accepted offers from the previous reply count from when they were made.
                prev = next((r for r in reversed(rows) if r.get("session_id") == t["session_id"]), None)
                for quote in gated.offers_accepted:
                    made_at = datetime.fromisoformat(prev["at"]) if prev else at
                    added = ledger.add(quote=quote, made_at=made_at,
                                       correlation_id=prev["correlation_id"] if prev else corr, offer=True)
                    added.ever_shown = True  # the judge saw it this turn, as a pending offer
                    if quote in kept_offers:
                        ledger.keep([added.id], correlation_id=corr, at=at)
                for p in gated.promises_made:
                    if not p.offer:
                        ledger.add(quote=p.quote, made_at=at, correlation_id=corr, offer=False)
                last_offers[t["session_id"]] = [p.quote for p in gated.promises_made if p.offer]
                rows.append(row)
            done = i / len(turns)
            eta = (time.monotonic() - t_start) / done * (1 - done)
            line = (f"promise-ledger-replay {done:.0%} eta={eta/60:.0f}m turns={i}/{len(turns)} "
                    f"rate={i/(time.monotonic()-t_start)*60:.1f}/min errors={sum(1 for r in rows if 'error' in r)} "
                    f"promises={len(ledger.promises)} kept={sum(1 for p in ledger.promises if p.kept_in)}")
            with progress.open("a") as fh:
                fh.write(line + "\n")
            print(f"[{i}/{len(turns)}] {corr[:8]} {row.get('lane')} {row.get('secs')}s "
                  f"made={len(row.get('promises_made') or [])} acc={len(row.get('offers_accepted') or [])} "
                  f"kept={row.get('promises_kept') or []} {row.get('error') or ''}", flush=True)
            args.out.write_text(json.dumps({"partial": True, "turns": rows}, indent=2, default=str))
        seat_summary = seats.summary()
    # A promise is expired only if 72 h really passed before the data ends (now, or --until).
    data_end = min(until, datetime.now(timezone.utc).replace(tzinfo=None))
    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(), "since": since.isoformat(), "until": until.isoformat(),
        "seats": seat_summary, "turns": rows,
        "ledger": [{**p.__dict__, "made_at": p.made_at.isoformat(),
                    "kept_at": p.kept_at.isoformat() if p.kept_at else None} for p in ledger.promises],
        "outcomes": ledger.outcomes(data_end),
    }
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["outcomes"]))
    print(f"written: {args.out}")
    return 0


# --- labelling and scoring ----------------------------------------------------------------

def _gate_ablated(reflection: Any, promises: list, *, draft_text: str, final_text: str) -> list:
    """The gate's own rules over ``promises`` (draft, then final text), offers and keeps left out."""
    as_given = reflection.model_copy(update={"promises_made": promises, "offers_accepted": [], "promises_kept": []})
    first = gate_promises(as_given, reply_text=draft_text, enabled=True)
    return gate_promises(first.reflection, reply_text=final_text, enabled=True).reflection.promises_made


def _item(p: Any, real: Any) -> dict:
    return {"quote": p.quote, "offer": p.offer, "force": real.force, "doer": p.doer, "act": p.act,
            "done_when": real.done_when}


def _ablate(reflection: Any, *, passes: Any, stand_in: dict, draft_text: str, final_text: str) -> list[dict]:
    """What the gate would keep with one rule off: every item that fails ``passes`` gets the
    ``stand_in`` field values, then the gate runs all its rules. Each kept item carries the
    judge's real values. Empty when the rule would drop nothing among the doer rule's keeps."""
    if all(passes(p) for p in reflection.promises_made if p.doer == PROMISE_DOER_ORION):
        return []
    # First item per quote, as the gate keeps the first copy of a duplicated quote.
    real: dict[str, Any] = {}
    for p in reflection.promises_made:
        real.setdefault(normalize_for_quote_match(p.quote), p)
    fixed = [p if passes(p) else p.model_copy(update=stand_in) for p in reflection.promises_made]
    return [_item(p, real[normalize_for_quote_match(p.quote)])
            for p in _gate_ablated(reflection, fixed, draft_text=draft_text, final_text=final_text)]


def ablate_done_when_rule(reflection: Any, *, draft_text: str, final_text: str) -> list[dict]:
    """What the gate would keep without rule 0b (every other rule on)."""
    return _ablate(reflection, passes=has_completion, stand_in={"done_when": "(ablation: rule 0b off)"},
                   draft_text=draft_text, final_text=final_text)


def ablate_force_rule(reflection: Any, *, draft_text: str, final_text: str) -> list[dict]:
    """What the gate would keep without rule 0a (every other rule on)."""
    return _ablate(reflection, passes=lambda p: p.force == PROMISE_FORCE_COMMITS,
                   stand_in={"force": PROMISE_FORCE_COMMITS}, draft_text=draft_text, final_text=final_text)


def ablate_doer_rule(reflection: Any, *, draft_text: str, final_text: str) -> list[dict]:
    """What the gate would keep if the doer rule did not exist: every listed item treated as
    Orion's, then the gate's other rules (rules 0a and 0b, quote in the draft and the final
    text, length, duplicates, the per-turn cap). Runs the gate itself, so the ablation
    follows the gate if the gate changes; since 2026-11-16 that includes rules 0a/0b, so
    it is not comparable to the 11-15 doer PR's number. Each kept item carries the doer
    the judge gave it."""
    if not any(p.doer != PROMISE_DOER_ORION for p in reflection.promises_made):
        return []  # the rule dropped nothing: the ablation equals promises_made
    doers: dict[str, Any] = {}
    for p in reflection.promises_made:
        doers.setdefault(normalize_for_quote_match(p.quote), p.doer)
    as_orion = reflection.model_copy(update={
        "promises_made": [p.model_copy(update={"doer": PROMISE_DOER_ORION}) for p in reflection.promises_made],
        "offers_accepted": [], "promises_kept": []})
    first = gate_promises(as_orion, reply_text=draft_text, enabled=True)
    second = gate_promises(first.reflection, reply_text=final_text, enabled=True)
    return [{"quote": p.quote, "offer": p.offer, "doer": doers.get(normalize_for_quote_match(p.quote)),
             "force": p.force, "act": p.act, "done_when": p.done_when}
            for p in second.reflection.promises_made]


def merge_reports(paths: list[Path]) -> dict:
    """One report from several partial ones (a run the pool recalled, resumed with
    --skip-done). For a turn in more than one file, a row without an error wins over an error
    row, whatever the order; otherwise the later file wins. Ledger / keep-rate fields are
    dropped: they only make sense for one unbroken run. A resumed part also starts with no
    pending offers and no open promises, so offers_accepted / promises_kept in a merged
    report are not comparable to an unbroken run; promises_made is."""
    turns: dict[str, dict] = {}
    for path in paths:
        for row in json.loads(Path(path).read_text()).get("turns", []):
            old = turns.get(row["correlation_id"])
            if old is None or "error" in old or "error" not in row:
                turns[row["correlation_id"]] = row
    return {"turns": sorted(turns.values(), key=lambda r: r.get("at") or ""),
            "merged_from": [str(p) for p in paths]}


def extracted_items(report: dict, *, with_not_orion: bool = False, without_done_when: bool = False,
                    without_force: bool = False) -> list[dict]:
    """Every gated promise_made item, one per (turn, quote). with_not_orion scores the
    no-doer-rule ablation instead (``ablation_kept``, when the rule dropped anything);
    without_done_when the no-rule-0b ablation (``ablation_done_when_kept``)."""
    out = []
    for r in report["turns"]:
        seen: set[str] = set()
        kept = r.get("promises_made") or []
        if with_not_orion and r.get("ablation_kept"):
            kept = r["ablation_kept"]
        if without_done_when and r.get("ablation_done_when_kept"):
            kept = r["ablation_done_when_kept"]
        if without_force and r.get("ablation_force_kept"):
            kept = r["ablation_force_kept"]
        for p in kept:
            sha = quote_hash(p["quote"])
            if sha in seen:
                continue
            seen.add(sha)
            out.append({"correlation_id": r["correlation_id"], "quote_sha": sha,
                        "offer": p["offer"], "doer": p.get("doer"), "quote": p["quote"],
                        "force": p.get("force"), "act": p.get("act"), "done_when": p.get("done_when")})
    return out


def not_orion_items(report: dict) -> list[dict]:
    return [{"correlation_id": r["correlation_id"], "quote_sha": quote_hash(p["quote"]), **p}
            for r in report["turns"] for p in r.get("not_orion") or []]


def _row_items(report: dict, key: str) -> list[dict]:
    """A rule's dropped items, one per (turn, quote). They come from the raw judge list, so
    unlike extracted_items they may hold a quote that is not in the reply."""
    out, seen = [], set()
    for r in report["turns"]:
        for p in r.get(key) or []:
            item = {"correlation_id": r["correlation_id"], "quote_sha": quote_hash(p["quote"]), **p}
            if (item["correlation_id"], item["quote_sha"]) not in seen:
                seen.add((item["correlation_id"], item["quote_sha"]))
                out.append(item)
    return out


def not_commit_items(report: dict) -> list[dict]:
    return _row_items(report, "not_commit")


def no_completion_items(report: dict) -> list[dict]:
    return _row_items(report, "no_completion")


def dropped_items(report: dict) -> list[dict]:
    out = []
    for r in report["turns"]:
        for key in ("dropped_not_in_draft", "dropped_not_in_final"):
            for q in r.get(key) or []:
                out.append({"correlation_id": r["correlation_id"], "stage": key, "quote_sha": quote_hash(q),
                            "quote": q})
    return out


def _rule_drops(items: list[dict], by_key: dict) -> dict:
    labs = [by_key.get((x["correlation_id"], x["quote_sha"])) for x in items]
    return {"total": len(items), "labelled_real_promises_dropped": sum(1 for lab in labs if lab and lab["promise"]),
            "unlabelled": sum(1 for lab in labs if lab is None)}


def _precision(items: list[dict], by_key: dict) -> dict:
    labs = [by_key.get((it["correlation_id"], it["quote_sha"])) for it in items]
    labelled = [lab for lab in labs if lab is not None]
    tp = sum(1 for lab in labelled if lab["promise"])
    return {**wilson(tp, len(labelled)), "unlabelled": len(labs) - len(labelled)}


def pooled_recall(report: dict, labels: dict) -> dict:
    """Recall against every real promise any labelled run found (labels with promise=true and
    a ``key``; two spans of one promise share a key), on the turns this report judged."""
    judged = {r["correlation_id"] for r in report["turns"] if "promises_made" in r}
    pool = {(lab["correlation_id"], lab["key"]) for lab in labels.get("labels", [])
            if lab.get("promise") and lab.get("key") and lab["correlation_id"] in judged}
    by_key = {(lab["correlation_id"], lab["quote_sha"]): lab for lab in labels.get("labels", [])}
    found = set()
    for it in extracted_items(report):
        lab = by_key.get((it["correlation_id"], it["quote_sha"]))
        if lab and lab.get("promise") and lab.get("key"):
            found.add((it["correlation_id"], lab["key"]))
    return {**wilson(len(found & pool), len(pool)),
            "missed": sorted(f"{c[:8]}:{k}" for c, k in pool - found)}


def score(report: dict, labels: dict) -> dict:
    items = extracted_items(report)
    by_key = {(lab["correlation_id"], lab["quote_sha"]): lab for lab in labels.get("labels", [])}
    labelled = [(it, by_key.get((it["correlation_id"], it["quote_sha"]))) for it in items]
    unlabelled = [it for it, lab in labelled if lab is None]
    tp = sum(1 for _, lab in labelled if lab and lab["promise"])
    tp_flag = sum(1 for it, lab in labelled if lab and lab["promise"] and lab.get("offer", it["offer"]) == it["offer"])
    precision = tp / len(items) if items else None

    drops = dropped_items(report)
    drop_labels = {(d["correlation_id"], d["quote_sha"]): d for d in labels.get("dropped", [])}
    real_quotes_dropped = [d for d in drops if (drop_labels.get((d["correlation_id"], d["quote_sha"])) or {}).get(
        "literal_in_reply")]

    hand = labels.get("hand_found", [])
    found = []
    for h in hand:
        # By key: a run that quotes a different span of the same promise still found it.
        hits = [it for it, lab in labelled if it["correlation_id"] == h["correlation_id"]
                and lab and lab.get("promise") and h["key"] in (lab.get("hand_found"), lab.get("key"))]
        found.append({"key": h["key"], "offer": h["offer"], "extracted": bool(hits),
                      "offer_flag_right": bool(hits) and hits[0]["offer"] == h["offer"]})

    outcomes = report.get("outcomes") or {}
    turns = [r for r in report["turns"] if "promises_made" in r]
    return {
        "turns_in_window": len(report["turns"]),
        "turns_judged": len(turns),
        "turns_errored": sum(1 for r in report["turns"] if "error" in r),
        "turns_skipped": sum(1 for r in report["turns"] if "skipped" in r),
        "extracted": len(items),
        "extracted_offers": sum(1 for it in items if it["offer"]),
        "unlabelled": len(unlabelled),
        "precision": precision,
        "precision_wilson95": _precision(items, by_key),
        "precision_without_doer_gate": _precision(extracted_items(report, with_not_orion=True), by_key),
        "doer_dropped": {
            "total": len(not_orion_items(report)),
            "by_doer": {d: sum(1 for x in not_orion_items(report) if str(x.get("doer")) == d)
                        for d in sorted({str(x.get("doer")) for x in not_orion_items(report)})},
            "real_promises_dropped": sum(
                1 for x in not_orion_items(report)
                if (by_key.get((x["correlation_id"], x["quote_sha"])) or {}).get("promise")),
        },
        "precision_without_done_when_rule": _precision(extracted_items(report, without_done_when=True), by_key),
        "precision_without_force_rule": _precision(extracted_items(report, without_force=True), by_key),
        "force_dropped": _rule_drops(not_commit_items(report), by_key),
        "done_when_dropped": _rule_drops(no_completion_items(report), by_key),
        "recall_pooled": pooled_recall(report, labels),
        "precision_with_offer_flag": (tp_flag / len(items)) if items else None,
        "raw_quotes": sum(len((r.get("raw") or {}).get("promises_made") or []) for r in turns),
        "gate_dropped": len(drops),
        "gate_dropped_that_were_real_quotes": len(real_quotes_dropped),
        "gate_dropped_unlabelled": sum(1 for d in drops if (d["correlation_id"], d["quote_sha"]) not in drop_labels),
        "recall_hand_found": {"found": sum(1 for f in found if f["extracted"]), "of": len(found), "items": found},
        "promises_counted": None if "merged_from" in report else len(report.get("ledger") or []),
        "offers_accepted": sum(len(r.get("offers_accepted") or []) for r in turns),
        "turns_with_counted_promise": (None if "merged_from" in report
                                       else len({p["correlation_id"] for p in report.get("ledger") or []})),
        "outcomes": outcomes,
        # Beta(1, 1) on kept vs expired-after-being-shown. Censored and never-shown promises
        # are left out of the rate and reported beside it (outcomes).
        "baseline_keep_rate": beta_summary(outcomes.get("kept", 0), outcomes.get("expired", 0)) if outcomes else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dsn", default="postgresql+psycopg2://postgres:postgres@localhost:55432/conjourney")
    r.add_argument("--days", type=int, default=30)
    r.add_argument("--until", help="UTC naive ISO time; default now")
    r.add_argument("--out", type=Path, default=OUT_DIR / "report.json")
    r.add_argument("--only", help="comma-separated correlation id prefixes (smoke runs)")
    r.add_argument("--skip-done", type=Path, nargs="+", default=[],
                   help="partial reports of a recalled run: skip the turns they already judged (errors are retried)")
    r.add_argument("--lanes", default=DEFAULT_LANES,
                   help="comma-separated, alternated per turn: agent, chat (chat waits until gpu0 is lent)")
    s = sub.add_parser("sheet")
    s.add_argument("--report", type=Path, nargs="+", default=[OUT_DIR / "report.json"])
    c = sub.add_parser("score")
    c.add_argument("--report", type=Path, nargs="+", default=[OUT_DIR / "report.json"],
                   help="one report, or the partial reports of a resumed run (merged)")
    c.add_argument("--labels", type=Path, default=LABELS)
    args = ap.parse_args()
    if args.cmd == "run":
        return run(args)
    report = json.loads(args.report[0].read_text()) if len(args.report) == 1 else merge_reports(args.report)
    if args.cmd == "sheet":
        print(json.dumps({"extracted": extracted_items(report), "not_orion": not_orion_items(report),
                          "not_commit": not_commit_items(report), "no_completion": no_completion_items(report),
                          "dropped": dropped_items(report)},
                         indent=2))
        return 0
    print(json.dumps(score(report, json.loads(args.labels.read_text())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

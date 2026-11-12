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
         design's hand-found statements, and the baseline keep rate with a 90% interval.

Inputs are read live from Postgres and are never stored in the repo. Lanes are the
Flash-Next chat/agent lanes (GPU 0/1), not Bonsai. Bypasses the GPU pool like the other
replay evals: run it when the lanes are quiet.
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

import httpx
from jinja2 import Environment

from orion.core.llm_json import parse_json_object
from orion.harness.evals.judge_crash_replay_eval import (
    HOP_GUARDRAIL,
    LANES,
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
from orion.harness.promise_gate import gate_promises, normalize_for_quote_match
from orion.harness.tests.fixtures import make_repair_overlay
from orion.schemas.harness_finalize import PROMISE_ITEMS_MAX, OpenPromiseRefV1

LABELS = Path(__file__).resolve().parent / "fixtures" / "promise_ledger_labels.json"
OUT_DIR = Path("/tmp/promise-ledger-replay")
HORIZON = timedelta(hours=72)
MAX_TOKENS = 2000
EVAL_LANES = ("chat", "agent")  # Flash-Next on circe GPU 0 / GPU 1


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


@dataclass
class ReplayLedger:
    promises: list[LedgerPromise] = field(default_factory=list)

    def open_at(self, now: datetime) -> list[LedgerPromise]:
        live = [p for p in self.promises if p.kept_in is None and now - p.made_at <= HORIZON]
        return sorted(live, key=lambda p: p.made_at)[:PROMISE_ITEMS_MAX]

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
        """kept / expired (72 h passed, never kept) / censored (still open at data end)."""
        out = {"kept": 0, "expired": 0, "censored": 0}
        for p in self.promises:
            if p.kept_in is not None:
                out["kept"] += 1
            elif data_end - p.made_at > HORIZON:
                out["expired"] += 1
            else:
                out["censored"] += 1
        return out


def beta_summary(successes: int, failures: int, *, draws: int = 200_000, seed: int = 7) -> dict[str, float]:
    """Beta(1, 1) prior. Posterior mean and 90% equal-tail interval by sampling (stdlib only)."""
    rng = random.Random(seed)
    a, b = successes + 1, failures + 1
    xs = sorted(rng.betavariate(a, b) for _ in range(draws))
    return {"n": successes + failures, "kept": successes, "mean": a / (a + b),
            "lo90": xs[int(0.05 * draws)], "hi90": xs[int(0.95 * draws) - 1]}


# --- live replay --------------------------------------------------------------------------

def _turns(conn, since: datetime, until: datetime) -> list[dict]:
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT c.correlation_id, c.session_id, c.created_at, c.prompt, c.response, t.run_artifact "
        "FROM chat_history_log c JOIN harness_turn_trace t ON t.correlation_id = c.correlation_id "
        "WHERE c.created_at >= :s AND c.created_at < :u AND coalesce(c.prompt, '') <> '' "
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


def _call(url: str, body: dict) -> tuple[dict | None, str | None, float]:
    t0 = time.monotonic()
    try:
        resp = httpx.post(f"{url}/v1/chat/completions", timeout=900, json=body)
    except httpx.HTTPError as exc:
        return None, f"transport {type(exc).__name__}", time.monotonic() - t0
    if resp.status_code != 200:
        return None, f"http {resp.status_code} {resp.text[:80]}", time.monotonic() - t0
    raw = resp.json()["choices"][0]["message"].get("content") or ""
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
    engine = create_engine(args.dsn)
    t_start = time.monotonic()
    with engine.connect() as conn:
        turns = _turns(conn, since, until)
        conn.rollback()
        if args.only:
            turns = [t for t in turns if t["correlation_id"].startswith(tuple(args.only.split(",")))]
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
            lane = EVAL_LANES[i % len(EVAL_LANES)]
            url, backend = LANES[lane]
            open_refs = [OpenPromiseRefV1(id=p.id, quote=p.quote) for p in open_now]
            body = build_request(inputs, backend=backend, pending_offers=pending, open_promises=open_refs)
            payload, error, secs = _call(url, body)
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
                row["dropped_not_in_final"] = second.dropped_not_in_text
                row["dropped_other"] = (first.dropped_too_long + first.dropped_over_cap
                                        + len(first.dropped_unknown_offer) + len(first.dropped_unknown_kept))
                row["promises_made"] = [p.model_dump() for p in gated.promises_made]
                row["offers_accepted"] = gated.offers_accepted
                row["promises_kept"] = gated.promises_kept
                row["verdict"] = gated.alignment_verdict
                ledger.keep(gated.promises_kept, correlation_id=corr, at=at)
                # Accepted offers from the previous reply count from when they were made.
                prev = next((r for r in reversed(rows) if r.get("session_id") == t["session_id"]), None)
                for quote in gated.offers_accepted:
                    made_at = datetime.fromisoformat(prev["at"]) if prev else at
                    ledger.add(quote=quote, made_at=made_at, correlation_id=prev["correlation_id"] if prev else corr,
                               offer=True)
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
    # A promise is expired only if 72 h really passed before the data ends (now, or --until).
    data_end = min(until, datetime.now(timezone.utc).replace(tzinfo=None))
    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(), "since": since.isoformat(), "until": until.isoformat(),
        "turns": rows,
        "ledger": [{**p.__dict__, "made_at": p.made_at.isoformat(),
                    "kept_at": p.kept_at.isoformat() if p.kept_at else None} for p in ledger.promises],
        "outcomes": ledger.outcomes(data_end),
    }
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["outcomes"]))
    print(f"written: {args.out}")
    return 0


# --- labelling and scoring ----------------------------------------------------------------

def extracted_items(report: dict) -> list[dict]:
    """Every gated promise_made item, one per (turn, quote)."""
    out = []
    for r in report["turns"]:
        for p in r.get("promises_made") or []:
            out.append({"correlation_id": r["correlation_id"], "quote_sha": quote_hash(p["quote"]),
                        "offer": p["offer"], "quote": p["quote"]})
    return out


def dropped_items(report: dict) -> list[dict]:
    out = []
    for r in report["turns"]:
        for key in ("dropped_not_in_draft", "dropped_not_in_final"):
            for q in r.get(key) or []:
                out.append({"correlation_id": r["correlation_id"], "stage": key, "quote_sha": quote_hash(q),
                            "quote": q})
    return out


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
        hits = [it for it, lab in labelled if it["correlation_id"] == h["correlation_id"]
                and lab and lab.get("hand_found") == h["key"]]
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
        "precision_with_offer_flag": (tp_flag / len(items)) if items else None,
        "raw_quotes": sum(len((r.get("raw") or {}).get("promises_made") or []) for r in turns),
        "gate_dropped": len(drops),
        "gate_dropped_that_were_real_quotes": len(real_quotes_dropped),
        "gate_dropped_unlabelled": sum(1 for d in drops if (d["correlation_id"], d["quote_sha"]) not in drop_labels),
        "recall_hand_found": {"found": sum(1 for f in found if f["extracted"]), "of": len(found), "items": found},
        "promises_counted": len(report.get("ledger") or []),
        "offers_accepted": sum(len(r.get("offers_accepted") or []) for r in turns),
        "turns_with_counted_promise": len({p["correlation_id"] for p in report.get("ledger") or []}),
        "outcomes": outcomes,
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
    s = sub.add_parser("sheet")
    s.add_argument("--report", type=Path, default=OUT_DIR / "report.json")
    c = sub.add_parser("score")
    c.add_argument("--report", type=Path, default=OUT_DIR / "report.json")
    c.add_argument("--labels", type=Path, default=LABELS)
    args = ap.parse_args()
    if args.cmd == "run":
        return run(args)
    report = json.loads(args.report.read_text())
    if args.cmd == "sheet":
        print(json.dumps({"extracted": extracted_items(report), "dropped": dropped_items(report)}, indent=2))
        return 0
    print(json.dumps(score(report, json.loads(args.labels.read_text())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

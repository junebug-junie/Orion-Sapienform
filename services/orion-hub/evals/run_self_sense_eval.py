#!/usr/bin/env python3
"""Self-sense eval: ask the live Hub three fixed identity questions, score
each answer deterministically, persist one row per question via the bus.

Patch A of docs/superpowers/specs/2026-09-08-orion-sense-of-self-design.md.
Turns "Orion stopped sounding like a chatbot" (after the curiosity
self-inquiry line, PR #2158) into two numbers per answer that can be tracked
over time -- scorers in orion/evals/self_sense.py, row shape in
orion/schemas/self_sense.py, table `self_sense_eval_log` (orion-sql-writer).

    make eval-self-sense
    python services/orion-hub/evals/run_self_sense_eval.py [--no-publish] [--json]

Env: HUB_BASE_URL (default http://127.0.0.1:8080), ORION_BUS_URL (required
unless --no-publish), SUBSTRATE_FELT_STATE_DATABASE_URL or
ENDOGENOUS_RUNTIME_SQL_DATABASE_URL (Postgres; optional -- without it the
HTTP body is the only answer source and self_definition_version is null).

Costs three real chat turns (`no_write: true`, so nothing lands in chat
history), several minutes each. Exit 0 = three answers scored and (unless
--no-publish) published. Exit 1 = at least one answer came back empty from
both sources, or a publish failed -- rows are still written for the record,
but an empty answer is not a measurement and is never reported as one.
Exit 2 = could not run at all (Hub unreachable, bad args).

Why Postgres and not just the HTTP body: the body's `text` comes back empty
when the voice/TTS lane fails at delivery even though the turn produced text
(seen live 2026-09-08); `harness_turn_trace.run_artifact->>'final_text'`
keyed by the returned correlation_id is the reliable source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests  # noqa: E402

from orion.evals.self_sense import (  # noqa: E402
    SELF_DEFINITION_CONCEPT_ID,
    SELF_DEFINITION_PRODUCED_BY,
    SELF_DEFINITION_VERSION_SQL,
    grounded_records,
    self_definition_version_from_row,
    self_label_hits,
    self_label_score,
)
from orion.schemas.self_sense import (  # noqa: E402
    CHANNEL_SELF_SENSE_EVAL_WRITE,
    KIND_SELF_SENSE_EVAL_WRITE,
    SELF_SENSE_QUESTIONS,
    SelfSenseEvalV1,
    build_entry_id,
)

DEFAULT_HUB_BASE_URL = "http://127.0.0.1:8080"
SESSION_ID = "self-sense-eval"
PRODUCER_SERVICE = "orion-hub"
PRODUCER_VERSION = "self-sense-eval/0.1.0"

FINAL_TEXT_SQL = (
    "SELECT run_artifact->>'final_text' FROM harness_turn_trace WHERE correlation_id = :corr"
)


def _database_url() -> str | None:
    # Same precedent as orion/substrate/felt_state_reader.py `_database_url`,
    # minus its container-hostname default: this runs on the host.
    return os.getenv("SUBSTRATE_FELT_STATE_DATABASE_URL") or os.getenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL")


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


# --- live calls ----------------------------------------------------------------

def ask_hub(base_url: str, question: str, *, timeout_sec: float) -> dict[str, Any]:
    resp = requests.post(
        base_url.rstrip("/") + "/api/chat",
        json={"messages": [{"role": "user", "content": question}], "mode": "orion", "no_write": True},
        headers={"X-Orion-Session-Id": SESSION_ID},
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"unexpected /api/chat body type: {type(body).__name__}")
    return body


def read_final_text(engine, correlation_id: str, *, wait_sec: float, poll_sec: float = 5.0) -> str | None:
    """Poll harness_turn_trace for the turn's final_text. The row is written by
    sql-writer off the bus, so it can land a few seconds after the HTTP reply."""
    from sqlalchemy import text

    deadline = time.monotonic() + max(0.0, wait_sec)
    while True:
        with engine.connect() as conn:
            row = conn.execute(text(FINAL_TEXT_SQL), {"corr": correlation_id}).fetchone()
        if row and isinstance(row[0], str) and row[0].strip():
            return row[0].strip()
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_sec)


def read_self_definition_version(engine) -> int | None:
    from sqlalchemy import text

    with engine.connect() as conn:
        row = conn.execute(
            text(SELF_DEFINITION_VERSION_SQL),
            {"concept_id": SELF_DEFINITION_CONCEPT_ID, "produced_by": SELF_DEFINITION_PRODUCED_BY},
        ).fetchone()
    return self_definition_version_from_row(tuple(row) if row else None)


# --- pure assembly (unit-tested) ----------------------------------------------

def build_row(
    *,
    run_id: str,
    question_key: str,
    question: str,
    http_text: str | None,
    trace_text: str | None,
    correlation_id: str | None,
    self_definition_version: int | None,
) -> SelfSenseEvalV1:
    """Pick the answer source (trace beats HTTP; empty is 'none'), score it,
    and build the row. Pure: no network, no database."""
    if trace_text and trace_text.strip():
        answer, source = trace_text.strip(), "harness_trace"
    elif http_text and http_text.strip():
        answer, source = http_text.strip(), "http"
    else:
        answer, source = "", "none"

    labels = self_label_hits(answer)
    grounded = grounded_records(answer)
    notes: list[str] = []
    if source == "none":
        notes.append("answer_empty_from_both_sources; scores are not a measurement")
    if labels:
        notes.append("labels=" + ",".join(labels))
    if grounded.records:
        notes.append("records=" + ",".join(grounded.records))

    return SelfSenseEvalV1(
        entry_id=build_entry_id(run_id, question_key),
        run_id=run_id,
        question_key=question_key,  # type: ignore[arg-type]
        question=question,
        answer_text=answer,
        answer_source=source,  # type: ignore[arg-type]
        correlation_id=correlation_id,
        self_label_score=self_label_score(answer),
        grounded_record_score=grounded.score,
        self_definition_version=self_definition_version,
        notes="; ".join(notes) or None,
    )


def envelope_correlation_id(row: SelfSenseEvalV1) -> uuid.UUID:
    """BaseEnvelope requires a UUID. Reuse the chat turn's correlation_id when
    it is one (it is -- Hub mints a uuid4), otherwise derive a deterministic
    uuid5 from the entry_id so a replay carries the same id."""
    if row.correlation_id:
        try:
            return uuid.UUID(row.correlation_id)
        except ValueError:
            pass
    return uuid.uuid5(uuid.NAMESPACE_URL, "orion:self_sense:" + row.entry_id)


def build_envelope(row: SelfSenseEvalV1, *, node: str | None):
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    return BaseEnvelope(
        kind=KIND_SELF_SENSE_EVAL_WRITE,
        source=ServiceRef(name=PRODUCER_SERVICE, version=PRODUCER_VERSION, node=node),
        correlation_id=envelope_correlation_id(row),
        payload=row.model_dump(mode="json"),
    )


# --- publish -------------------------------------------------------------------

def publish_rows(rows: list[SelfSenseEvalV1], *, bus_url: str, node: str | None) -> tuple[int, int]:
    """One short-lived bus connection for the batch (the
    self_atlas_cluster_history.py idiom). Returns (published, failed)."""
    import anyio

    from orion.core.bus.async_service import OrionBusAsync

    published = failed = 0

    async def _run() -> None:
        nonlocal published, failed
        bus = OrionBusAsync(bus_url)
        await bus.connect()
        try:
            for row in rows:
                try:
                    await bus.publish(CHANNEL_SELF_SENSE_EVAL_WRITE, build_envelope(row, node=node))
                    published += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"publish_failed entry_id={row.entry_id} error={exc}", file=sys.stderr)
                    failed += 1
        finally:
            await bus.close()

    anyio.run(_run)
    return published, failed


# --- main ------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hub-url", default=os.getenv("HUB_BASE_URL", DEFAULT_HUB_BASE_URL))
    ap.add_argument("--bus-url", default=os.getenv("ORION_BUS_URL"))
    ap.add_argument("--dsn", default=_database_url())
    ap.add_argument("--timeout-sec", type=float, default=900.0, help="per-question HTTP timeout")
    ap.add_argument("--trace-wait-sec", type=float, default=120.0, help="how long to wait for harness_turn_trace.final_text")
    ap.add_argument("--no-publish", action="store_true", help="score and print only; write nothing")
    ap.add_argument("--json", action="store_true", help="print rows as JSON")
    ap.add_argument("--node", default=os.getenv("NODE_NAME") or os.uname().nodename)
    args = ap.parse_args(argv)

    if not args.no_publish and not args.bus_url:
        print("ORION_BUS_URL is required unless --no-publish (use the tailscale redis://<ip>:6379/0)", file=sys.stderr)
        return 2

    try:
        health = requests.get(args.hub_url.rstrip("/") + "/health", timeout=10)
        health.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"hub_unreachable url={args.hub_url} error={exc}", file=sys.stderr)
        return 2

    engine = None
    if args.dsn:
        from sqlalchemy import create_engine

        engine = create_engine(args.dsn, pool_pre_ping=True)
    else:
        print("no Postgres DSN: answer_source will be 'http' only and self_definition_version null", file=sys.stderr)

    self_definition_version = None
    if engine is not None:
        try:
            self_definition_version = read_self_definition_version(engine)
        except Exception as exc:  # noqa: BLE001
            print(f"self_definition_read_failed error={exc}", file=sys.stderr)

    run_id = _new_run_id()
    print(f"self_sense_eval run_id={run_id} hub={args.hub_url} self_definition_version={self_definition_version}")

    rows: list[SelfSenseEvalV1] = []
    for question_key, question in SELF_SENSE_QUESTIONS:
        started = time.monotonic()
        http_text: str | None = None
        trace_text: str | None = None
        correlation_id: str | None = None
        try:
            body = ask_hub(args.hub_url, question, timeout_sec=args.timeout_sec)
            http_text = body.get("text") if isinstance(body.get("text"), str) else None
            correlation_id = body.get("correlation_id") if isinstance(body.get("correlation_id"), str) else None
            if body.get("error"):
                print(f"hub_error question={question_key} error={body.get('error')}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"chat_failed question={question_key} error={exc}", file=sys.stderr)
        if engine is not None and correlation_id:
            try:
                trace_text = read_final_text(engine, correlation_id, wait_sec=args.trace_wait_sec)
            except Exception as exc:  # noqa: BLE001
                print(f"trace_read_failed corr={correlation_id} error={exc}", file=sys.stderr)
        row = build_row(
            run_id=run_id,
            question_key=question_key,
            question=question,
            http_text=http_text,
            trace_text=trace_text,
            correlation_id=correlation_id,
            self_definition_version=self_definition_version,
        )
        rows.append(row)
        print(
            f"[{question_key}] {time.monotonic() - started:.0f}s corr={correlation_id} source={row.answer_source} "
            f"self_label_score={row.self_label_score} grounded_record_score={row.grounded_record_score}"
        )
        print("    " + (row.answer_text.replace("\n", "\n    ") if row.answer_text else "<empty>"))

    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in rows], indent=2, ensure_ascii=False))

    exit_code = 0
    if any(r.answer_source == "none" for r in rows):
        exit_code = 1

    if args.no_publish:
        print("no_publish: rows not written")
        return exit_code

    published, failed = publish_rows(rows, bus_url=args.bus_url, node=args.node)
    print(f"published={published} failed={failed} channel={CHANNEL_SELF_SENSE_EVAL_WRITE}")
    if failed:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

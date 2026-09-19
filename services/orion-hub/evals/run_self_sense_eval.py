#!/usr/bin/env python3
"""Self-sense eval: ask the live Hub four fixed identity questions, score
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

Costs four real chat turns (`no_write: true`, so nothing lands in chat
history), several minutes each. Exit 0 = four answers scored and (unless
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
    lived_answers_from_history_rows,
    pinned_lived_concept_ids,
    self_definition_version_from_row,
)
# Row assembly, scoring, notes and envelope are SHARED with Hub's in-process
# scheduler line (curiosity_investigation.tick_self_sense_eval) so the two
# paths cannot drift. Re-exported under the same names this script always had.
from orion.evals.self_sense_runner import (  # noqa: E402,F401
    PRODUCER_SERVICE,
    PRODUCER_VERSION,
    SESSION_ID,
    build_envelope,
    build_row,
    envelope_correlation_id,
    lived_answers_sql_named as _lived_answers_sql,
    new_run_id as _new_run_id,
)
from orion.schemas.self_sense import (  # noqa: E402
    CHANNEL_SELF_SENSE_EVAL_WRITE,
    SELF_SENSE_QUESTIONS,
    SelfSenseEvalV1,
)

DEFAULT_HUB_BASE_URL = "http://127.0.0.1:8080"
# When the HTTP body already carries the text, the trace read is a cross-check,
# not the only source -- do not spend the full wait on it.
TRACE_WAIT_WHEN_HTTP_HAS_TEXT_SEC = 20.0

FINAL_TEXT_SQL = (
    "SELECT run_artifact->>'final_text' FROM harness_turn_trace WHERE correlation_id = :corr"
)


def _database_url() -> str | None:
    # Same precedent as orion/substrate/felt_state_reader.py `_database_url`,
    # minus its container-hostname default: this runs on the host.
    return os.getenv("SUBSTRATE_FELT_STATE_DATABASE_URL") or os.getenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL")


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


def read_lived_answers(engine) -> list[dict[str, Any]]:
    """Latest pinned lived ledger rows — same concept_ids chat hydrates."""
    from sqlalchemy import text

    concept_ids = pinned_lived_concept_ids()
    if not concept_ids:
        return []
    sql, params = _lived_answers_sql(concept_ids)
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return lived_answers_from_history_rows(rows)


# --- pure assembly: `build_row`, `build_envelope`, `envelope_correlation_id`
# live in orion/evals/self_sense_runner.py (shared with Hub's scheduler line).

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
    # Progress lines must reach a redirected log as each turn finishes, not at exit.
    sys.stdout.reconfigure(line_buffering=True)

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

        # A dead Postgres must fail in seconds, not hang on the TCP default.
        engine = create_engine(args.dsn, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    else:
        print("no Postgres DSN: answer_source will be 'http' only and self_definition_version null", file=sys.stderr)

    self_definition_version = None
    lived_answers: list[dict[str, Any]] = []
    if engine is not None:
        try:
            self_definition_version = read_self_definition_version(engine)
        except Exception as exc:  # noqa: BLE001
            print(f"self_definition_read_failed error={exc}", file=sys.stderr)
        try:
            lived_answers = read_lived_answers(engine)
        except Exception as exc:  # noqa: BLE001
            print(f"lived_answers_read_failed error={exc}", file=sys.stderr)

    run_id = _new_run_id()
    print(
        f"self_sense_eval run_id={run_id} hub={args.hub_url} "
        f"self_definition_version={self_definition_version} lived_answers={len(lived_answers)}"
    )

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
        trace_wait: float | None = None
        if engine is not None and correlation_id:
            trace_wait = args.trace_wait_sec
            if http_text and http_text.strip():
                trace_wait = min(trace_wait, TRACE_WAIT_WHEN_HTTP_HAS_TEXT_SEC)
            try:
                trace_text = read_final_text(engine, correlation_id, wait_sec=trace_wait)
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
            trace_missing_after_sec=trace_wait,
            lived_answers=lived_answers,
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

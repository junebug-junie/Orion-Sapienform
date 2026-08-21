#!/usr/bin/env python3
"""Rate something Orion made, from the terminal.

    scripts/rate_artifact.py --list
    scripts/rate_artifact.py --dispatch-id dispatch:proposal:... --kind journal up
    scripts/rate_artifact.py -d dispatch:proposal:... -k report down --why "all preamble"

WHY A CLI AND NOT MORE UI
-------------------------
The Hub already has a rating UI for chat responses, wired end to end through
the bus into `chat_response_feedback`. It has been used **twice in three
weeks**. The plumbing was never the problem; the friction was. This exists
because Juniper lives in a terminal.

It is NOT a new channel. It POSTs to the same Hub endpoint the UI uses,
validated by the same registered schema, published to the same bus channel,
consumed by the same service. Nothing bypasses a contract.

WHY IT BUILDS THE REF INSTEAD OF TAKING ONE
-------------------------------------------
An artifact ref is `artifact:<kind>:<dispatch_id>`, and real dispatch ids look
like `dispatch:proposal:prune_stopped_containers:tick_fc7585176059:none:
execution_dispatch_policy.v1`. Nobody is retyping that. A ref typed by hand is
a ref with a typo, and a typo silently creates a rating attributed to no
action -- which teaches nothing about any action, which is the whole point of
the pipeline this feeds. So `--list` finds them and this builds the ref.

ATTESTATION, NOT AUTHENTICATION
-------------------------------
`--as` (default `$ORION_OPERATOR` or `$USER`) is recorded as `user_id`, and the
schema refuses an artifact rating without one. That makes a rating Orion filed
for itself *detectable*, not impossible: the Hub route has no auth and the Hub
holds the docker socket, so no software boundary on this host is enforceable
(resolved 2026-08-14 -- "detect, do not pretend to prevent"). Do not read a
stored `user_id` as proof of a human.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orion.schemas.chat_response_feedback import (  # noqa: E402
    THUMBS_DOWN_CATEGORY_LABELS,
    THUMBS_UP_CATEGORY_LABELS,
    ChatResponseFeedbackV1,
    build_artifact_ref,
)

# 8080, verified against the running container and services/orion-hub/
# .env_example's HUB_PORT. The first version of this script defaulted to 8081,
# which is not open -- so the one-line tool whose entire justification is low
# friction failed on first use. It had never been run against the live Hub.
DEFAULT_HUB = os.environ.get("ORION_HUB_URL", "http://127.0.0.1:8080")
DEFAULT_DSN = os.environ.get(
    "ORION_PG_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney"
)


# Orion has been writing these all along and nobody has ever read one.
# `substrate_dispatch_results` holds the actual prose a summarize/inspect/
# observe verb produced -- ~200 an hour, ~161 characters each, status=success.
# The artifact this whole rating path exists to grade was already there; what
# was missing was any surface that shows it to a human.
RECENT_ARTIFACTS = """
SELECT r.dispatch_id, r.result_json->>'observation' AS observation,
       r.raw_len, r.created_at
  FROM substrate_dispatch_results r
 WHERE r.status = 'success'
   AND coalesce(r.result_json->>'observation', '') <> ''
   AND NOT EXISTS (
       SELECT 1 FROM chat_response_feedback f
        WHERE f.target_artifact_ref LIKE '%' || r.dispatch_id
   )
 ORDER BY r.created_at DESC
 LIMIT :limit
"""

ONE_ARTIFACT = """
SELECT dispatch_id, result_json->>'observation' AS observation,
       result_json->>'salient_facts' AS salient_facts,
       result_json->>'confidence' AS confidence, created_at
  FROM substrate_dispatch_results
 WHERE dispatch_id = :dispatch_id AND status = 'success'
 ORDER BY created_at DESC LIMIT 1
"""


def _recent_artifacts(dsn: str, limit: int):
    from sqlalchemy import create_engine, text

    engine = create_engine(dsn)
    with engine.connect() as conn:
        return conn.execute(text(RECENT_ARTIFACTS), {"limit": limit}).mappings().all()


def _one_artifact(dsn: str, dispatch_id: str):
    from sqlalchemy import create_engine, text

    engine = create_engine(dsn)
    with engine.connect() as conn:
        return conn.execute(
            text(ONE_ARTIFACT), {"dispatch_id": dispatch_id}
        ).mappings().first()


def _verify_landed(dsn: str, feedback_id: str) -> bool:
    """Did the rating actually reach storage?

    The Hub route returns `{"ok": true}` unconditionally -- it awaits a publish
    helper that catches and logs its own exceptions, so a Redis failure or
    PUBLISH_CHAT_HISTORY_LOG=false both produce a 200 with ok:true and nothing
    stored. An earlier version of this script claimed to "report the real
    status" on the strength of reading that body, which was inert. This reads
    the table.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(dsn)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM chat_response_feedback WHERE feedback_id = :fid"),
            {"fid": feedback_id},
        ).first()
    return row is not None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("verdict", nargs="?", choices=["up", "down"])
    parser.add_argument("-d", "--dispatch-id", help="the action that produced it")
    parser.add_argument("-k", "--kind", default="artifact", help="journal, report, ...")
    parser.add_argument("-c", "--category", action="append", default=[], dest="categories")
    parser.add_argument("--why", dest="free_text", default=None)
    parser.add_argument("--as", dest="rater", default=None, help="recorded as user_id")
    parser.add_argument("--hub", default=DEFAULT_HUB)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--list", action="store_true", help="unrated artifacts, newest first")
    parser.add_argument("--show", metavar="DISPATCH_ID", help="read one in full")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the storage read-back (do not use; see _verify_landed)",
    )
    args = parser.parse_args()

    if args.list_categories:
        for verdict, labels in (
            ("up", THUMBS_UP_CATEGORY_LABELS),
            ("down", THUMBS_DOWN_CATEGORY_LABELS),
        ):
            print(f"\n{verdict}:")
            for key, label in labels.items():
                print(f"  {key:<45} {label}")
        print(
            "\nCategories are recorded and NOT scored: five of them is not five "
            "times the verdict. They say why. --why says it better."
        )
        return 0

    if args.show:
        row = _one_artifact(args.dsn, args.show)
        if row is None:
            print(f"no successful result stored for {args.show}", file=sys.stderr)
            return 1
        print(f"{row['created_at']}  ({row['dispatch_id']})\n")
        print(row["observation"])
        if row["salient_facts"]:
            print(f"\nsalient facts: {row['salient_facts']}")
        if row["confidence"]:
            print(f"confidence (Orion's own): {row['confidence']}")
        print(f"\nrate it:  {sys.argv[0]} -d {row['dispatch_id']} -k observation up|down")
        return 0

    if args.list:
        rows = _recent_artifacts(args.dsn, args.limit)
        if not rows:
            print("nothing unrated with real content. (--show <dispatch-id> to reread one)")
            return 0
        for row in rows:
            text_preview = " ".join(str(row["observation"]).split())
            print(f"\n{row['created_at']:%H:%M}  {row['raw_len']:>4}ch  {row['dispatch_id']}")
            print(f"  {text_preview[:160]}{'...' if len(text_preview) > 160 else ''}")
        print(
            f"\nrate one:  {sys.argv[0]} -d <dispatch-id> -k observation up --why '...'"
        )
        return 0

    if not args.verdict:
        parser.error("a verdict (up|down) is required unless --list/--show/--list-categories")
    if not args.dispatch_id:
        parser.error("--dispatch-id is required; find one with --list")

    rater = args.rater or os.environ.get("ORION_OPERATOR") or getpass.getuser()

    try:
        feedback = ChatResponseFeedbackV1(
            feedback_id=f"artifact-rating-{uuid4()}",
            target_artifact_ref=build_artifact_ref(args.kind, args.dispatch_id),
            feedback_value=args.verdict,
            categories=args.categories,
            free_text=args.free_text,
            user_id=rater,
            source="rate_artifact_cli",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        print("try --list-categories, or --list for dispatch ids", file=sys.stderr)
        return 2

    payload = feedback.model_dump(mode="json")
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    url = f"{args.hub.rstrip('/')}/api/chat/response-feedback"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
            body = response.read().decode("utf-8")[:200]
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8')[:300]}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"could not reach the hub at {url}: {exc.reason}", file=sys.stderr)
        print("is orion-hub up? override with --hub or $ORION_HUB_URL", file=sys.stderr)
        return 1

    if args.no_verify:
        print(f"{status} {body}  (UNVERIFIED -- a 200 here does not mean stored)")
        return 0

    if _verify_landed(args.dsn, feedback.feedback_id):
        print(f"stored: {feedback.feedback_id}")
        return 0
    print(
        f"{status} {body}\nNOT STORED. The hub accepted it and nothing reached "
        "chat_response_feedback -- check the bus and orion-sql-writer.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

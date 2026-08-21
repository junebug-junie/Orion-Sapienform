#!/usr/bin/env python3
"""Rate something Orion made, from the terminal, in one line.

    scripts/rate_artifact.py artifact:journal:2026-08-21 up
    scripts/rate_artifact.py artifact:journal:2026-08-21 down --why "all preamble"
    scripts/rate_artifact.py artifact:report:affect:1234 up -c well_grounded -c right_depth

WHY A CLI AND NOT MORE UI
-------------------------
The Hub already has a rating UI for chat responses, wired end to end through
the bus and into `chat_response_feedback`. It has been used **twice in three
weeks**. The plumbing was never the problem; the friction was. This exists
because Juniper lives in a terminal, and a rating that costs one line has a
different chance of happening than one that costs opening a page and finding
the thing.

This is deliberately NOT a new channel. It POSTs to the same Hub endpoint the
UI uses (`/api/chat/response-feedback`), which validates with the same
registered schema and publishes to the same bus channel with the same
consumers. Nothing here bypasses a contract.

WHAT AN ARTIFACT REF IS
-----------------------
`artifact:<kind>:<id>` naming a thing Orion produced, carrying enough to find
the dispatch that produced it. A rating that cannot be attributed to an action
teaches nothing about any action, which is what the whole pipeline is for.
"""

from __future__ import annotations

import argparse
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
)

DEFAULT_HUB = os.environ.get("ORION_HUB_URL", "http://localhost:8081")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("artifact_ref", help="e.g. artifact:journal:2026-08-21")
    parser.add_argument("verdict", choices=["up", "down"])
    parser.add_argument(
        "-c",
        "--category",
        action="append",
        default=[],
        dest="categories",
        help="repeatable. --list-categories to see them.",
    )
    parser.add_argument("--why", dest="free_text", default=None, help="freetext")
    parser.add_argument("--hub", default=DEFAULT_HUB)
    parser.add_argument(
        "--list-categories", action="store_true", help="print the vocabulary and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the payload; send nothing",
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

    # Validated locally BEFORE the network call, against the same registered
    # model the Hub will use, so a typo'd category is a clear message here
    # rather than a 422 from a service.
    try:
        feedback = ChatResponseFeedbackV1(
            feedback_id=f"artifact-rating-{uuid4()}",
            target_artifact_ref=args.artifact_ref,
            feedback_value=args.verdict,
            categories=args.categories,
            free_text=args.free_text,
            source="rate_artifact_cli",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        print("try --list-categories", file=sys.stderr)
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
            body = response.read().decode("utf-8")
            # Report the real status, not just "no exception". A 200 with a
            # body saying the bus was unavailable is not a delivered rating,
            # and this repo has already been bitten once by an outreach path
            # returning ok while silently falling back.
            print(f"{response.status} {body[:200]}")
            return 0 if 200 <= response.status < 300 else 1
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8')[:300]}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"could not reach the hub at {url}: {exc.reason}", file=sys.stderr)
        print("is orion-hub up? override with --hub or $ORION_HUB_URL", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

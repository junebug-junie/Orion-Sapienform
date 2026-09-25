#!/usr/bin/env python3
"""Replay real field ticks through the real proposal builder, with and without
some templates, and print what each template would have won.

Read-only. Built for the 2026-09-25 deletion of three transport templates
(chore/transport-lattice-semantics) and kept because the question -- "if we
kill template X, who takes its arena slots?" -- recurs every time a template
is retired.

Input is a JSONL file of `substrate_field_state.field_json` rows. Export a
sample of *warranted* ticks (the only ticks where candidates survive the
action-warrant gate) with, for example:

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "copy (
      select f.field_json from substrate_proposal_frames p
      join substrate_field_state f on f.tick_id = p.source_field_tick_id
       and f.generated_at > now() - interval '25 hours'
      where p.generated_at > now() - interval '24 hours'
        and p.proposal_frame_json->>'action_warrant_gate' = 'warranted'
        and random() < 0.15) to stdout" > warranted_fields.jsonl

Then:

    python scripts/analysis/replay_proposal_template_removal.py \\
        warranted_fields.jsonl --drop inspect_bus_channel_catalog ...

To reproduce the 2026-09-25 numbers after that deletion merged, point
--policy at the pre-deletion file, e.g.
`--policy <(git show origin/main~1:config/proposals/proposal_policy.v1.yaml)`
with a commit from before chore/transport-lattice-semantics; against the
current policy the dropped keys no longer exist and before == after.

"top5" is arena rank (the first five candidates by priority), not dispatch
admission, which also applies reserved slots, aging and the allocator.

Attention is replayed as None, so inspect_attended_target's numbers are the
no-attention case; external producers (reverie, cognitive hop) are not
replayed. Both limits apply equally to the before and after columns.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from orion.proposals.builder import build_proposal_frame  # noqa: E402
from orion.proposals.policy import load_proposal_policy  # noqa: E402
from orion.schemas.field_state import FieldStateV1  # noqa: E402

TOP_N = 5  # max_dispatches_per_tick in execution_dispatch_policy.v1.yaml


def _rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # psql `copy ... to stdout` (text format) doubles backslashes.
        yield json.loads(line.replace("\\\\", "\\"))


def replay(rows, *, policy, drop: set[str]):
    kept = policy.model_copy(
        update={"proposal_templates": {k: v for k, v in policy.proposal_templates.items() if k not in drop}}
    )
    counts = {name: (collections.Counter(), collections.Counter()) for name in ("before", "after")}
    ticks = skipped = 0
    for raw in rows:
        try:
            field = FieldStateV1.model_validate(raw)
        except Exception:  # older rows can predate the current schema
            skipped += 1
            continue
        ticks += 1
        for name, pol in (("before", policy), ("after", kept)):
            cand, top = counts[name]
            frame = build_proposal_frame(field=field, attention=None, policy=pol)
            for i, c in enumerate(frame.candidates):
                key = c.proposal_id.split(":")[1]
                cand[key] += 1
                if i < TOP_N:
                    top[key] += 1
    return ticks, skipped, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fields_jsonl", type=Path)
    ap.add_argument("--drop", nargs="+", required=True, help="template keys to remove")
    ap.add_argument(
        "--policy", type=Path, default=REPO / "config" / "proposals" / "proposal_policy.v1.yaml"
    )
    args = ap.parse_args(argv)
    policy = load_proposal_policy(args.policy)
    unknown = set(args.drop) - set(policy.proposal_templates)
    if unknown:
        print(f"not in policy (already deleted?): {sorted(unknown)}", file=sys.stderr)
    ticks, skipped, counts = replay(_rows(args.fields_jsonl), policy=policy, drop=set(args.drop))
    cb, tb = counts["before"]
    ca, ta = counts["after"]
    print(f"ticks={ticks} skipped={skipped}")
    print(f"{'template':40s} {'cand_before':>11s} {'cand_after':>10s} {'top5_before':>11s} {'top5_after':>10s}")
    for key in sorted(set(cb) | set(ca), key=lambda k: (-cb[k], k)):
        print(f"{key:40s} {cb[key]:11d} {ca[key]:10d} {tb[key]:11d} {ta[key]:10d}")
    print(f"{'TOTAL':40s} {sum(cb.values()):11d} {sum(ca.values()):10d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

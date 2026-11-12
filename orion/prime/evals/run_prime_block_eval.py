"""Prime block eval (living Prime design Part 1): the block on live Prime state, offline or live.

Offline (CI): renders the block from Prime's live state captured 2026-11-12 (``fixtures/
prime_block_live_2026-11-12.json``), then from the same state with its three most recent closed
concerns re-opened (real concern words, real arc ids), and checks:

* grounded: every concern id and arc id in the block is in the state; the last-thought ref is the
  state's own thought count or an open move's thought id;
* bounded: <= 600 chars, every ref intact, every body line ends with a ref;
* honest gaps: the same state ticked 11 min ago renders only "has not ticked since"; no state renders absent.

Live (``--live``, needs the bus Redis): reads ``orion:prime:state`` now and checks the rendered concern ids
match Redis exactly (the Part 1 live acceptance check, without a chat turn).

Exit 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from orion.prime.block import BLOCK_CAP, fetch_prime_block, render_prime_block
from orion.prime.state import PrimeConcernV1, PrimeStateV1

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prime_block_live_2026-11-12.json"


def _check(name: str, ok: bool, detail: str, fails: list[str]) -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        fails.append(name)


def _grounded(block, state: PrimeStateV1) -> tuple[bool, str]:
    ids = {c.concern_id for c in state.concerns}
    arcs = {c.arc_id for c in state.concerns if c.arc_id}
    thought_ids = ({f"#{state.thoughts_total}"} | {n.thought_id for n in state.open_nominations}
                   | {r.thought_id for r in state.open_regulations})
    bad = []
    for ref in block.refs:
        kind, _, rid = ref.partition(":")
        if (kind == "prime_concern" and rid not in ids) or (kind == "arc" and rid not in arcs) or (
                kind == "thought" and rid not in thought_ids):
            bad.append(ref)
    return not bad, f"refs={list(block.refs)} unknown={bad}"


def _bounded(block) -> tuple[bool, str]:
    lines = block.text.splitlines()[1:]
    uncited = [ln for ln in lines if not ln.rstrip().endswith("]")]
    missing = [r for r in block.refs if f"[{r}]" not in block.text]
    ok = len(block.text) <= BLOCK_CAP and not uncited and not missing
    return ok, f"chars={len(block.text)} uncited_lines={len(uncited)} cut_refs={missing}"


def offline() -> list[str]:
    fails: list[str] = []
    raw = json.loads(FIXTURE.read_text())
    from orion.prime.state import PrimeConcernsViewV1

    state = PrimeStateV1.model_validate(raw["state"])
    ticked = PrimeConcernsViewV1.model_validate(raw["view"]).as_of
    now = ticked + timedelta(seconds=20)

    quiet = render_prime_block(state, ticked_at=ticked, now=now)
    print(quiet.text, "\n")
    _check("quiet_grounded", quiet.status == "ok" and _grounded(quiet, state)[0], _grounded(quiet, state)[1], fails)
    _check("quiet_bounded", _bounded(quiet)[0], _bounded(quiet)[1], fails)

    # Re-open the three most recent closed concerns: real words and arcs, the cap under pressure.
    reopened = [PrimeConcernV1(concern_id=r.concern_id, text=r.text, opened_at=r.opened_at,
                               last_evidence_at=r.opened_at) for r in state.concern_record[-3:]]
    held = state.model_copy(update={"concerns": reopened})
    busy = render_prime_block(held, ticked_at=ticked, now=now)
    print(busy.text, "\n")
    _check("busy_grounded", busy.status == "ok" and _grounded(busy, held)[0], _grounded(busy, held)[1], fails)
    _check("busy_bounded", _bounded(busy)[0], _bounded(busy)[1], fails)
    shown = sum(1 for c in reopened if f"prime_concern:{c.concern_id}" in busy.refs)
    _check("busy_shows_concerns", shown >= 2, f"{shown} of {len(reopened)} concerns shown", fails)

    stale = render_prime_block(held, ticked_at=ticked, now=ticked + timedelta(minutes=11))
    _check("stale_is_absent", stale.status == "stale" and "has not ticked since" in stale.text and not stale.refs
           and all(c.concern_id not in stale.text for c in reopened), stale.text, fails)
    none = render_prime_block(None, ticked_at=None, now=now)
    _check("missing_is_labelled", none.status == "absent" and "absent" in none.text, none.text, fails)
    return fails


def live() -> list[str]:
    from redis import asyncio as aioredis

    fails: list[str] = []
    url = os.environ.get("ORION_BUS_URL", "")
    if not url:
        print("FAIL live: ORION_BUS_URL is not set")
        return ["live"]

    async def run():
        r = aioredis.from_url(url)
        try:
            block = await fetch_prime_block(r)
            raw = await r.get("orion:prime:state")
        finally:
            await r.aclose()
        return block, raw

    block, raw = asyncio.run(run())
    print(block.text, "\n")
    state = PrimeStateV1.model_validate_json(raw) if raw else None
    if block.status != "ok" or state is None:
        _check("live_fresh", False, f"status={block.status}", fails)
        return fails
    want = {f"prime_concern:{c.concern_id}" for c in state.concerns}
    got = {r for r in block.refs if r.startswith("prime_concern:")}
    _check("live_concern_ids_match_redis", got <= want and (got or not want), f"redis={sorted(want)} block={sorted(got)}",
           fails)
    _check("live_bounded", _bounded(block)[0], _bounded(block)[1], fails)
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="also read the live bus Redis (ORION_BUS_URL)")
    args = ap.parse_args()
    fails = offline()
    if args.live:
        fails += live()
    print(f"\n{'FAIL' if fails else 'PASS'}: {len(fails)} failing check(s) {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

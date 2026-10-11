#!/usr/bin/env python3
"""Orion's Day reread eval: does the `orion_day` responder quote the right words and flag the
unsupported claims?

Runs the real Hub responder (scripts/orion_day_introspect_listener.py) over a fixture letter
(orion/orion_day/tests/fixtures.py: the fixture day plus planted facts) with 10 planted
references, each a ref Juniper could point at ("2026-09-29 ¶2") and a concrete claim or citation
in that part. Six are backed by that day's records, four are not. For each it checks:

- the returned item's id is the ref and its text is exactly the stored part (quote fidelity);
- the claim check / citation for the planted token says found / resolved, or not found /
  unresolved, as planted (flagging);
- the record named for a backed claim is the planted record.

Deterministic and offline: no DB, LLM, bus or network. It measures the tool's evidence, not a
model's reply (acceptance check 6's reply half needs a live turn and is not covered here).

    python services/orion-hub/evals/run_orion_day_reread_eval.py

Exit 0 = every check passed, 1 = at least one failed.
"""
from __future__ import annotations

import asyncio
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

_HUB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _HUB_ROOT.parents[1]
warnings.filterwarnings("ignore", message=r"Field \"model_", category=UserWarning)


def _paths() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(_REPO_ROOT), str(_HUB_ROOT)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_HUB_ROOT))


class Case(NamedTuple):
    ref: str            # what Juniper would point at
    token: str          # the planted claim token, or the cited ref for a carry item
    backed: bool        # planted as supported by that day's records
    record: str | None  # the record that should be named when backed
    excerpt_has: str | None = None


RUN_REF = "curiosity:e7d03d2ecdbb"
CASES = [
    Case("2026-09-29 ¶1", "260.6", True, RUN_REF),
    Case("2026-09-29 ¶1", "06:35:27Z", True, RUN_REF),
    Case("2026-09-29 ¶1", "01:02:09Z", True, RUN_REF),
    Case("2026-09-29 ¶2", "0.72", True, RUN_REF),
    Case("2026-09-29 ¶2", "999.4", False, None),
    Case("2026-09-29 ¶2", "2557", False, None),
    Case("2026-09-29 ¶4", "4,059", False, None),
    Case("2026-09-29 carry 1", RUN_REF, True, RUN_REF, excerpt_has="silent for 260.6 hours"),
    Case("2026-09-29 carry 2", "reading_journal:nope-not-here", False, None),
    Case("2026-09-29 carry 3", "dream:20", True, "dream:20", excerpt_has="A library of recent work"),
]


class _Conn:
    def __init__(self, letter: Any, store: Any):
        self.letter, self.store = letter, store

    @asynccontextmanager
    async def transaction(self, readonly: bool = False):
        assert readonly, "the responder must read inside a READ ONLY transaction"
        yield

    async def fetchrow(self, sql: str, *args: Any):
        if sql == self.store.SELECT_LETTER_SQL and args[0] == self.letter.letter_date:
            return self.letter.model_dump(mode="json") | {"letter_date": self.letter.letter_date}
        if sql == self.store.SELECT_LATEST_LETTER_SQL:
            return self.letter.model_dump(mode="json") | {"letter_date": self.letter.letter_date}
        return None


class _Pool:
    def __init__(self, conn: _Conn):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


class _Bus:
    def __init__(self):
        self.published: list = []

    async def publish(self, channel: str, envelope: Any) -> None:
        self.published.append((channel, envelope))


def _ask(listener: Any, args: dict[str, Any]) -> Any:
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
    from orion.introspect.transport import REQUEST_KIND, RESULT_PREFIX
    from orion.schemas.introspect import IntrospectResultV1, IntrospectToolBindingV1

    corr = uuid4()
    binding = IntrospectToolBindingV1(invocation_context="unified_chat", parent_run_id="eval",
                                      parent_trace_id="eval", memory_allowed=True)
    env = BaseEnvelope(kind=REQUEST_KIND, correlation_id=corr, reply_to=f"{RESULT_PREFIX}{corr}",
                       source=ServiceRef(name="eval"),
                       payload={"operation": "orion_day", "binding": binding.model_dump(mode="json"), "args": args})
    asyncio.run(listener.handle(env))
    return IntrospectResultV1.model_validate(listener.bus.published[-1][1].payload)


def main(argv: list[str] | None = None) -> int:
    _paths()
    from orion.core.bus.bus_schemas import ServiceRef
    from orion.orion_day import store
    from orion.orion_day.letter_parts import find_part, parse_ref, split_carry, split_note
    from orion.orion_day.tests.fixtures import reread_letter
    from scripts.orion_day_introspect_listener import OrionDayIntrospectListener

    letter = reread_letter()
    pool = _Pool(_Conn(letter, store))
    listener = OrionDayIntrospectListener(pool_provider=lambda: pool, source_ref=ServiceRef(name="eval"), search=None)
    listener.bus = _Bus()

    failures: list[str] = []
    quoted = flagged = named = 0
    for case in CASES:
        day, kind, index = parse_ref(case.ref)
        part_arg = "note" if kind == "paragraph" else "carry_forward"
        result = _ask(listener, {"letter_date": day, "part": part_arg, "index": index})
        if not result.ok or len(result.items) != 1:
            failures.append(f"{case.ref}: tool answered {result.error or len(result.items)}")
            continue
        item = result.items[0]
        parts = split_note(letter.note_md) if kind == "paragraph" else split_carry(letter.carry_forward_md)
        stored = find_part(parts, kind, index)
        if item.id == case.ref and item.text == stored.text and item.epistemic_status == "unsettled":
            quoted += 1
        else:
            failures.append(f"{case.ref}: quoted text or id differs from the stored part")
        if kind == "paragraph":
            entry = next((c for c in item.extra["claim_check"] if c["token"] == case.token), None)
            supported = bool(entry and entry["found_in"])
            names = entry["found_in"] if entry else []
        else:
            entry = next((c for c in item.extra["citations"] if c["ref"] == case.token), None)
            supported = bool(entry and entry["resolved"])
            names = [entry["ref"]] if supported else []
            if supported and case.excerpt_has and case.excerpt_has not in (entry["excerpt"] or ""):
                failures.append(f"{case.ref}: excerpt for {case.token} lacks {case.excerpt_has!r}")
        if entry is None:
            failures.append(f"{case.ref}: {case.token!r} not checked at all")
            continue
        if supported == case.backed:
            flagged += 1
        else:
            failures.append(f"{case.ref}: {case.token!r} reported {'supported' if supported else 'unsupported'}, "
                            f"planted {'supported' if case.backed else 'unsupported'}")
        if case.backed and case.record in names:
            named += 1
        elif case.backed:
            failures.append(f"{case.ref}: {case.token!r} names {names}, expected {case.record}")
    unknown = _ask(listener, {"letter_date": "2030-01-01"})
    if unknown.ok or "not found" not in (unknown.error or ""):
        failures.append("unknown letter_date did not come back as a not-found tool error")

    backed = sum(1 for c in CASES if c.backed)
    print(f"quoted exactly: {quoted}/{len(CASES)}")
    print(f"flagged as planted (supported / unsupported): {flagged}/{len(CASES)}")
    print(f"backed claims naming the planted record: {named}/{backed}")
    print(f"unknown letter is a not-found error: {'yes' if not unknown.ok else 'no'}")
    for f in failures:
        print(f"FAIL {f}")
    print("PASS" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

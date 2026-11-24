"""The day dream's grounding judge reads a reminiscence hop's note where the cited turn holds only its receipt (#3739).

A reminiscence hop turn's reply (``harness_turn_trace.run_artifact.final_text``) is its save receipt and nothing else:
``reminisce what_it_was saved rm:2026-11-21:what_it_was:ffc6f3f855``. The hop's words live only in the reminiscence
ledger (``$REMINISCENCE_LEDGER_DIR/reminiscence.jsonl``, the governor's dig writes it, compose mounts it here). A dream
memory that cites that turn was judged against the receipt alone, so it could never pass (live 2026-11-17 and
11-21). This module hands the judge the saved note instead.

It is the THIRD named exception to the repo-wide self-ledger partition test under tests/, for the
reminiscence ledger only, and only for the grounding judge: ``judge_hop_notes`` is called from one place,
app/dream_day_store.py ``resolve_refs`` with ``full=True`` (the judge's read). The short text (a memory's stored
evidence quote, the validator's names check) never carries a note, and nothing here writes. The judge stores its
verdict (claims taken from the memory's statement, not from the note) on the ``dream_day`` row.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from orion.body_atlas import reminiscence as rm

# The receipt `dig reminisce <hop>` prints, the whole reply or its first clause: "reminisce <hop> saved <hop id>".
RECEIPT_RE = re.compile(
    r"^\s*(?:reply:\s*)?reminisce\s+\S+\s+saved\s+(rm:[0-9]{4}-[0-9]{2}-[0-9]{2}:[a-z_]+:[0-9a-f]+)")
NOTE_CHARS = 2400
# How the store marks a swapped-in note in the judge's text: the gate then keeps no raw judge answer for that memory,
# so no hop words reach the dream_day row (app/dream_day_graph.py ``ground``).
NOTE_MARKER = " -- the reminiscence note it saved ("
logger = logging.getLogger("orion-durable-runs.dream_day")


def receipt_hop_id(reply: str) -> str | None:
    """The hop id a reminiscence save receipt names, or None when the reply is not one."""
    m = RECEIPT_RE.match(str(reply or ""))
    return m.group(1) if m else None


def judge_hop_notes(self_dir: str | Path, hop_ids: list[str]) -> dict[str, str]:
    """{hop id: "the week <start>..<end>: <note>"} for every asked hop id the ledger holds (that exact line, the one
    the receipt names). Read-only; a missing or unreadable ledger gives {} (the judge then reads the receipt)."""
    wanted = set(hop_ids)
    if not wanted:
        return {}
    try:
        hops, _ = rm.load(Path(self_dir) / rm.LEDGER_FILE)
    except (OSError, ValueError):   # unreadable or not UTF-8: the judge reads the receipt, the night goes on
        logger.warning("dream_day_hop_notes_unreadable dir=%s", self_dir, exc_info=True)
        return {}
    out: dict[str, str] = {}
    for h in hops:
        if h.hop_id in wanted:
            out[h.hop_id] = f"the week {h.period_start}..{h.period_end}: {' '.join(h.note.split())}"[:NOTE_CHARS]
    return out

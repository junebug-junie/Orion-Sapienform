"""The ONE durable-runs module that touches Orion's reminiscence ledger (design
docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md, section 6, Patch 2).

The ledger is partitioned from recall: only ``orion/body_atlas`` and ``orion/dig`` read it (the repo-wide
partition test under tests/). This file is the second named exception there (the first is the field-note
receipt checker), for three jobs the daily run cannot do any other way:

* pick the week the tick names in its brief (the same rule ``dig reminisce`` uses);
* check a hop turn's save receipt against the ledger, and skip a hop already written today;
* list today's live hop ids (the run's checkpoint and its finish check).

It never writes, and what it hands back is identifiers, dates, counts and booleans -- a hop id, "this
hop is written", a week's first and last day -- never a hop's text, refs or claims, so nothing here can
carry Orion's words toward recall, memory or a prompt.

The ledger lives at ``$REMINISCENCE_LEDGER_DIR/reminiscence.jsonl`` (compose mounts the host's
``/mnt/orion-fcc/self`` there: the same directory the harness governor's dig writes to).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from orion.body_atlas import reminiscence as rm


@dataclass(frozen=True)
class WeekPick:
    """The week an episode is (or would be) about, without any hop text."""

    period_start: date
    period_end: date
    fixed_by_hop: bool       # True: a hop written today already fixed the week
    weeks_complete: int
    weeks_total: int


class ReminiscenceLedger:
    def __init__(self, self_dir: str | Path):
        self.path = Path(self_dir) / rm.LEDGER_FILE

    def readable(self) -> bool:
        """The ledger's directory is mounted. (The file itself appears with the first hop.)"""
        return self.path.parent.is_dir()

    def _records(self) -> list[rm.ReminiscenceHopV1]:
        if not self.readable():
            # An unmounted ledger must never read as "no hop written": the run would re-write all five.
            raise FileNotFoundError(f"reminiscence ledger directory not mounted: {self.path.parent}")
        records, _skipped = rm.load(self.path)
        return records

    def week_for(self, episode: date) -> WeekPick | None:
        """The week of ``episode``: the one its first hop fixed, else the oldest week that ended before it
        with no complete episode (``rm.pick_period``). None: every such week is complete."""
        records = self._records()
        done = len(rm.complete_weeks(records))
        total = len(rm.weeks_ended_before(episode))
        hops = rm.episode_hops(records, episode.isoformat())
        if hops:
            first = min(hops.values(), key=lambda r: r.written_at)
            return WeekPick(date.fromisoformat(first.period_start), date.fromisoformat(first.period_end), True,
                            done, total)
        pick = rm.pick_period(records, episode)
        if pick is None:
            return None
        return WeekPick(pick[0], pick[1], False, done, total)

    def hop_ids(self, episode: date) -> dict[str, str]:
        """hop -> its live hop id for ``episode`` (hops not written today are absent)."""
        return {hop: r.hop_id for hop, r in rm.episode_hops(self._records(), episode.isoformat()).items()}

    def receipt_ok(self, episode: date, hop: str, hop_id: str) -> bool:
        """True when ``hop_id`` is a line of this episode and this hop in the ledger. A later rewrite
        superseding it does not undo the fact that this turn saved it."""
        return any(r.hop_id == hop_id and r.episode == episode.isoformat() and r.hop == hop
                   for r in self._records())

    def rewrite_ok(self, episode: date, hop_id: str) -> bool:
        """True when ``hop_id`` is a rewrite (it supersedes an earlier line) in this episode."""
        return any(r.hop_id == hop_id and r.episode == episode.isoformat() and r.supersedes is not None
                   for r in self._records())

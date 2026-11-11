"""The ONE durable-runs module that touches Orion's reminiscence ledger (design
docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md, section 6, Patch 2).

The ledger is partitioned from recall: only ``orion/body_atlas`` and ``orion/dig`` read it (the repo-wide
partition test under tests/). This file is the second named exception there (the first is the field-note
receipt checker), for three jobs the daily run cannot do any other way:

* pick the period the tick names in its brief (the same rule ``dig reminisce`` uses: the oldest open
  week, and a run of weeks with no commit on main joined onto the next week that has one, counted in the
  read-only dig clone at ``$REMINISCENCE_REPO_DIR``);
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


def _no_history(start: date, end: date) -> int:
    raise rm.ReminiscenceError("no repo_dir configured: main's history cannot be read")


class HistoryUnreadable(RuntimeError):
    """Main's history could not be read, so the period cannot be picked (no guessed week)."""


class PeriodRefused(RuntimeError):
    """The ledger is out of order: the next period would overlap a visited week (no overlapping visit)."""


@dataclass(frozen=True)
class WeekPick:
    """The period an episode is (or would be) about, without any hop text: one week, or a run of quiet
    weeks joined onto the next week with a commit (``quiet_weeks`` of them)."""

    period_start: date
    period_end: date
    fixed_by_hop: bool       # True: a hop written today already fixed the period
    weeks_complete: int
    weeks_total: int
    quiet_weeks: int = 0


class ReminiscenceLedger:
    def __init__(self, self_dir: str | Path, repo_dir: str | Path | None = None,
                 commits_in=None):
        """``commits_in(start, end)`` counts main's commits (tests pass a fake); else read from ``repo_dir``."""
        self.path = Path(self_dir) / rm.LEDGER_FILE
        if commits_in is None:
            commits_in = rm.git_commits_in(repo_dir) if repo_dir is not None else _no_history
        self.commits_in = commits_in

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
        """The period of ``episode``: the one its first hop fixed, else ``rm.pick_period`` (the oldest open
        week, quiet weeks joined onto the next week with a commit). None: every week is complete, or the
        open weeks are quiet and wait for a week with a commit (``quiet_wait_from``). Raises
        ``HistoryUnreadable`` when main's history cannot be read."""
        records = self._records()
        done = len(rm.complete_weeks(records))
        total = len(rm.weeks_ended_before(episode))
        hops = rm.episode_hops(records, episode.isoformat())
        if hops:
            first = min(hops.values(), key=lambda r: r.written_at)
            start, end = date.fromisoformat(first.period_start), date.fromisoformat(first.period_end)
            return WeekPick(start, end, True, done, total, rm.quiet_weeks(start, end))
        try:
            pick = rm.pick_period(records, episode, self.commits_in)
        except rm.PeriodOverlap as exc:
            raise PeriodRefused(str(exc)) from exc
        except rm.ReminiscenceError as exc:
            raise HistoryUnreadable(str(exc)) from exc
        if pick is None:
            return None
        return WeekPick(pick[0], pick[1], False, done, total, rm.quiet_weeks(*pick))

    def quiet_wait_from(self, episode: date) -> date | None:
        """With no period to pick: the first open week (all open weeks are quiet and wait), else None."""
        first = rm.first_open_week(self._records(), episode)
        return first[0] if first else None

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

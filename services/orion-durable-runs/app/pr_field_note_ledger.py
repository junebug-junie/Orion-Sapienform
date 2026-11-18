"""The ONE durable-runs module that touches Orion's PR field-notes ledger (spec
docs/superpowers/specs/2026-11-04-pr-field-notes-design.md, Patch 2).

The ledger is partitioned from recall: only ``orion/body_atlas`` and ``orion/dig`` read it
(the repo-wide partition test under tests/). This file is the one named exception, for two jobs the
automatic half cannot do any other way:

* write the code-only record of a dependency-only or tiny docs-only PR (Juniper's decision 2: no
  model turn for those, so no dig command runs for them);
* check a note turn's save receipt against the ledger (``verify_receipt``), and skip PRs that
  already have a record (a note Juniper asked for in chat, or an earlier run's).

What it hands back is identifiers and booleans only -- a note id, "has a note", a PR number --
never a note's text, so nothing here can carry a note toward recall, memory or a prompt.

The ledger lives at ``$PR_FIELD_NOTES_LEDGER_DIR/pr_field_notes.jsonl`` (compose mounts the host's
``/mnt/orion-fcc/self`` there: the same directory the harness governor's dig writes to).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orion.body_atlas import field_notes as fn


def is_code_record_pr(pr: dict[str, Any]) -> bool:
    """Juniper's decision 2: a dependency-only or tiny docs-only PR gets a code record, no note turn.
    The same rule ``orion.dig field-note`` applies (``field_notes.flags_for`` + ``code_record_only``),
    on the same gh-shaped dict, so the poller and the dig CLI cannot disagree about a PR."""
    return fn.code_record_only(fn.flags_for(pr))


@dataclass(frozen=True)
class RecordRef:
    """A live record of one PR, without its text."""

    pr_number: int
    note_id: str
    has_note: bool          # False: a code record ("code record, not a note")
    written_at: str


class FieldNoteLedger:
    def __init__(self, self_dir: str | Path):
        self.path = fn.ledger_path(self_dir)

    def readable(self) -> bool:
        """The ledger's directory is mounted. (The file itself appears with the first record.)"""
        return self.path.parent.is_dir()

    def live_records(self) -> dict[int, RecordRef]:
        """PR number -> its live record. Raises FileNotFoundError when the directory is not mounted:
        an unmounted ledger must never read as "no PR has a record"."""
        if not self.readable():
            raise FileNotFoundError(f"field-notes ledger directory not mounted: {self.path.parent}")
        records, _skipped = fn.load(self.path)
        return {n: RecordRef(n, r.note_id, r.note is not None, r.written_at) for n, r in fn.live(records).items()}

    def last_written_at(self) -> str | None:
        """The newest record's write time (the poller's gap check), or None for an empty ledger."""
        if not self.readable():
            return None
        records, _ = fn.load(self.path)
        return max((r.written_at for r in records), default=None)

    def receipt_ok(self, pr_number: int, note_id: str) -> bool:
        """True when ``note_id`` is a NOTE (not a code record) for this PR in the ledger. A rewrite
        later superseding it does not undo the fact that this turn saved it."""
        if not self.readable():
            raise FileNotFoundError(f"field-notes ledger directory not mounted: {self.path.parent}")
        records, _ = fn.load(self.path)
        return any(r.note_id == note_id and r.pr_number == int(pr_number) and r.note is not None for r in records)

    def write_code_record(self, *, pr: dict[str, Any], run_id: str | None = None) -> str | None:
        """Append the code record for a dependency-only / tiny docs-only PR; its note id. None when the
        PR already has a record (another writer got there first). ``pr`` is the gh-shaped dict
        ``app.pr_field_note_poll.gh_shape`` builds from the public API.

        The record's ``body_map`` is empty on purpose: durable-runs has neither the repo (for the
        importer scan) nor the live atlas, a code record carries no claim for a body map to check, and
        a map with every part marked "not live" would be false. Raises FieldNoteError when the PR is not
        a code-record PR (the caller's classification and this check disagree: never silently written).
        """
        if not self.readable():
            raise FileNotFoundError(f"field-notes ledger directory not mounted: {self.path.parent}")
        flags = fn.flags_for(pr)
        with fn.ledger_lock(self.path):
            records, _ = fn.load(self.path)
            if int(pr["number"]) in fn.live(records):
                return None
            rec = fn.code_record(pr_number=int(pr["number"]), merge_sha=str(pr.get("mergeCommit", {}).get("oid") or ""),
                                 merged_at=str(pr["mergedAt"]), title=str(pr.get("title") or "")[:300], flags=flags,
                                 bmap=[], records=records, dig="durable-runs:pr_field_note_poll", run_id=run_id)
            fn.append(self.path, rec)
        return rec.note_id

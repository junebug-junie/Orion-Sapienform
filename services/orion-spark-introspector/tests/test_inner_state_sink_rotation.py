"""Unit tests for InnerStateCorpusSink's size-based rotation + retention pruning.

Exercises orion/telemetry/corpus_sink.py directly (no bus/handle_self_state mocking
needed) -- these tests protect the already-live production corpus file at
/mnt/telemetry/phi/corpus/inner_state.jsonl from unbounded growth.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel

from orion.telemetry.corpus_sink import InnerStateCorpusSink

ROTATED_SUFFIX_RE = re.compile(r"^\d{8}T\d{6}\.\d{6}Z(\.\d+)?$")


class _TestRow(BaseModel):
    n: int


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip("\n").splitlines()
    return [json.loads(line) for line in lines if line]


def test_rotation_triggers_when_max_bytes_exceeded(tmp_path) -> None:
    corpus_path = tmp_path / "inner_state.jsonl"
    sink = InnerStateCorpusSink(str(corpus_path), max_bytes=50, max_rotated_files=5)

    for i in range(20):
        sink.append(_TestRow(n=i))

    rotated = list(tmp_path.glob("inner_state.jsonl.*"))
    assert len(rotated) >= 1

    # Active file exists, is small, and is not one of the rotated files.
    assert corpus_path.exists()
    assert corpus_path not in rotated
    assert corpus_path.stat().st_size < 50 or len(_read_jsonl(corpus_path)) < 20


def test_rotated_file_naming_pattern(tmp_path) -> None:
    corpus_path = tmp_path / "inner_state.jsonl"
    sink = InnerStateCorpusSink(str(corpus_path), max_bytes=10, max_rotated_files=5)

    # Each ~8-byte row: 1st append writes without checking (file doesn't
    # exist yet), 2nd append checks size=8 (<10, no rotate) and writes
    # (size=16), 3rd append checks size=16 (>=10) and rotates.
    sink.append(_TestRow(n=1))
    sink.append(_TestRow(n=2))
    sink.append(_TestRow(n=3))  # forces the rotation

    rotated = list(tmp_path.glob("inner_state.jsonl.*"))
    assert len(rotated) == 1
    rotated_path = rotated[0]

    # Sibling of the original path (same directory).
    assert rotated_path.parent == corpus_path.parent

    # Filename is "{original_name}.{timestamp}" with timestamp matching
    # the exact %Y%m%dT%H%M%S.%fZ (microsecond-precision) format used by
    # _rotate_if_needed, optionally followed by a ".N" collision counter.
    prefix = f"{corpus_path.name}."
    assert rotated_path.name.startswith(prefix)
    suffix = rotated_path.name[len(prefix):]
    assert ROTATED_SUFFIX_RE.match(suffix), f"unexpected suffix: {suffix!r}"


def test_old_rotations_pruned_beyond_max_rotated_files(tmp_path, monkeypatch) -> None:
    corpus_path = tmp_path / "inner_state.jsonl"
    sink = InnerStateCorpusSink(str(corpus_path), max_bytes=10, max_rotated_files=2)

    # _rotate_if_needed's collision-counter fallback means forcing several
    # rotations back-to-back can no longer produce colliding *filenames*
    # (production code handles that itself now). But _prune_old_rotations
    # sorts survivors by st_mtime, which has its own coarse resolution on
    # some filesystems -- a tight test loop could still get a nondeterministic
    # prune order. Monkeypatch the sink module's clock so each rotation gets
    # a distinct, monotonically increasing timestamp, keeping the test fast
    # and deterministic regardless of filesystem mtime granularity.
    import datetime as _dt

    import orion.telemetry.corpus_sink as sink_module

    class _FakeDateTime(_dt.datetime):
        _tick = 0

        @classmethod
        def now(cls, tz=None):
            cls._tick += 1
            return _dt.datetime(2026, 7, 13, 0, 0, 0, tzinfo=tz) + _dt.timedelta(seconds=cls._tick)

    monkeypatch.setattr(sink_module, "datetime", _FakeDateTime)

    # Force 4 separate rotation events. With max_bytes=10 and ~8-byte rows,
    # a rotation happens every other append (after two rows accumulate to
    # 16 bytes): 8 appends -> 4 rotations.
    for i in range(8):
        sink.append(_TestRow(n=i))

    rotated = sorted(p.name for p in tmp_path.glob("inner_state.jsonl.*"))
    assert len(rotated) == 2

    # The survivors should be the two lexically-latest (== chronologically
    # latest, since the timestamp format sorts correctly) rotated names.
    all_possible_prefix = f"{corpus_path.name}."
    assert all(name.startswith(all_possible_prefix) for name in rotated)


def test_append_continues_working_after_rotation(tmp_path) -> None:
    corpus_path = tmp_path / "inner_state.jsonl"
    sink = InnerStateCorpusSink(str(corpus_path), max_bytes=50, max_rotated_files=5)

    for i in range(30):
        sink.append(_TestRow(n=i))

    # Active file is valid, readable JSONL.
    rows = _read_jsonl(corpus_path)
    assert len(rows) >= 1
    for row in rows:
        assert "n" in row

    # One more append after all the rotation churn still works cleanly.
    sink.append(_TestRow(n=999))
    rows_after = _read_jsonl(corpus_path)
    assert rows_after[-1]["n"] == 999


def test_disabled_sink_never_rotates(tmp_path) -> None:
    sink = InnerStateCorpusSink("", max_bytes=1, max_rotated_files=1)

    assert sink.enabled is False
    assert sink._path is None

    # append() is a safe no-op.
    sink.append(_TestRow(n=1))
    sink.append(_TestRow(n=2))

    # No directory or files created anywhere under tmp_path (the disabled
    # sink never touches the filesystem at all).
    assert list(tmp_path.iterdir()) == []


def test_rotation_respects_default_thresholds(tmp_path) -> None:
    corpus_path = tmp_path / "inner_state.jsonl"
    sink = InnerStateCorpusSink(str(corpus_path))

    assert sink._max_bytes == 200_000_000
    assert sink._max_rotated_files == 5


def test_rotation_collision_counter_avoids_overwriting_existing_rotated_file(tmp_path, monkeypatch) -> None:
    # Path.rename() onto an existing path silently overwrites it -- for
    # rotation specifically, that means two rotations landing on the exact
    # same timestamp would silently destroy the FIRST rotated backup, the
    # opposite of what this whole mechanism exists to prevent. Force a
    # frozen clock (same instant on every call) across two separate
    # rotation events and confirm both rotated files survive with distinct
    # names (the ".1" collision-counter suffix), not one overwriting the
    # other.
    import datetime as _dt

    import orion.telemetry.corpus_sink as sink_module

    frozen = _dt.datetime(2026, 7, 13, 0, 0, 0, tzinfo=_dt.timezone.utc)

    class _FrozenDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(sink_module, "datetime", _FrozenDateTime)

    corpus_path = tmp_path / "inner_state.jsonl"
    sink = InnerStateCorpusSink(str(corpus_path), max_bytes=10, max_rotated_files=5)

    # Each ~8-byte row (`{"n":N}\n`): rotation #1 fires on the 3rd append
    # (rows 1+2 = 16 bytes >= 10), carrying rows {1, 2} into the first
    # rotated file. Rotation #2 fires on the 5th append (rows 3+4 = 16
    # bytes >= 10), carrying rows {3, 4}. Row 5 stays in the active file.
    for n in (1, 2, 3, 4, 5):
        sink.append(_TestRow(n=n))

    rotated = sorted(p.name for p in tmp_path.glob("inner_state.jsonl.*"))
    assert len(rotated) == 2, (
        f"expected 2 distinct rotated files surviving a timestamp collision, got {rotated}"
    )
    # Exactly one of the two must carry the ".1" collision-counter suffix --
    # confirms the collision was actually hit and handled, not avoided by
    # accident (e.g. a test that doesn't really force same-instant rotations
    # would show 0, not 1, here).
    assert sum(name.endswith(".1") for name in rotated) == 1

    # Union of both rotated files' content must be exactly {1,2,3,4} -- every
    # row that was ever rotated out survived, none lost, none duplicated
    # across the two files (the failure mode a silent rename-overwrite would
    # produce: file 1 disappears, its two rows gone).
    all_rotated_ns: set[int] = set()
    for name in rotated:
        for row in _read_jsonl(tmp_path / name):
            all_rotated_ns.add(row["n"])
    assert all_rotated_ns == {1, 2, 3, 4}
    assert _read_jsonl(corpus_path) == [{"n": 5}]


def test_prune_never_deletes_a_file_not_matching_the_rotation_pattern(tmp_path) -> None:
    # 2026-07-13, found by code review (2 independent angles): the original
    # prune glob was `{name}.*`, which matches ANY sibling sharing the
    # corpus basename prefix -- a manually-placed backup, a .gz archive, a
    # stray editor temp file. Fixed via _ROTATED_SUFFIX_RE filtering. This
    # test plants exactly that kind of stray file and confirms it survives
    # even when max_rotated_files is small enough that real rotations would
    # otherwise prune something.
    corpus_path = tmp_path / "inner_state.jsonl"
    stray_backup = tmp_path / "inner_state.jsonl.manual-backup-20260701"
    stray_backup.write_text('{"n":-1}\n', encoding="utf-8")
    stray_gzip = tmp_path / "inner_state.jsonl.gz"
    stray_gzip.write_bytes(b"not a real gzip, just needs to exist")

    sink = InnerStateCorpusSink(str(corpus_path), max_bytes=10, max_rotated_files=1)
    for n in (1, 2, 3, 4, 5):
        sink.append(_TestRow(n=n))

    assert stray_backup.exists(), "a file not matching the rotation pattern must never be pruned"
    assert stray_gzip.exists(), "a file not matching the rotation pattern must never be pruned"

    # Confirm real rotation + pruning still happened around the stray files
    # (i.e. this isn't passing merely because nothing rotated at all).
    real_rotated = [
        p for p in tmp_path.glob("inner_state.jsonl.*")
        if p not in (stray_backup, stray_gzip)
    ]
    assert len(real_rotated) == 1, f"expected exactly 1 real rotated file kept, got {real_rotated}"


def test_negative_max_rotated_files_is_clamped_not_inverted(tmp_path) -> None:
    # 2026-07-13, found by code review: rotated[-1:] on a negative
    # max_rotated_files keeps only the OLDEST rotated file and deletes
    # every newer one -- the exact inverse of the setting's intent.
    # Clamped to 0 (a legitimate "no retention" value) instead.
    corpus_path = tmp_path / "inner_state.jsonl"
    sink = InnerStateCorpusSink(str(corpus_path), max_bytes=10, max_rotated_files=-3)
    assert sink._max_rotated_files == 0

    for n in (1, 2, 3, 4, 5):
        sink.append(_TestRow(n=n))

    # max_rotated_files clamped to 0 -> every real rotated file gets
    # pruned immediately, none survive.
    rotated = list(tmp_path.glob("inner_state.jsonl.*"))
    assert rotated == []


def test_rotation_failure_does_not_prevent_the_row_from_being_written(tmp_path, monkeypatch) -> None:
    # 2026-07-13, found by code review: Path.exists()/stat() can raise
    # OSError (not just return False/a value) for errno values pathlib
    # doesn't treat as "doesn't exist" -- ESTALE/EACCES on a degraded
    # network mount, which /mnt/telemetry genuinely is in production.
    # _rotate_if_needed() wraps the whole rotation attempt in one
    # try/except OSError specifically so a filesystem hiccup during
    # housekeeping skips rotation for this tick rather than raising past
    # append() and losing the row entirely.
    import orion.telemetry.corpus_sink as sink_module

    corpus_path = tmp_path / "inner_state.jsonl"
    sink = InnerStateCorpusSink(str(corpus_path), max_bytes=10, max_rotated_files=5)

    sink.append(_TestRow(n=1))
    sink.append(_TestRow(n=2))  # size >= 10 on the next append's check

    def _always_raises(self):
        raise OSError("simulated ESTALE on a degraded network mount")

    monkeypatch.setattr(sink_module.Path, "exists", _always_raises)

    # Must not raise, even though every exists() check inside rotation
    # fails -- append() itself doesn't call exists() on the write path.
    sink.append(_TestRow(n=3))

    # Restore the real Path.exists before making assertions -- _read_jsonl
    # (this test file's own helper) calls path.exists() too, and would
    # otherwise hit the same simulated failure.
    monkeypatch.undo()

    # Rotation was skipped this tick (its own exists() check failed), so
    # the row lands in the still-active, now-oversized file rather than a
    # fresh rotated one -- that's the correct, safe degraded-mode outcome:
    # no data lost, rotation deferred to a healthier tick.
    assert list(tmp_path.glob("inner_state.jsonl.*")) == []
    assert _read_jsonl(corpus_path) == [{"n": 1}, {"n": 2}, {"n": 3}]

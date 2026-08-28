"""Windowing: one document per utterance, with the speaker preserved.

Regression cover for the 2026-08-28 rebuild
(docs/superpowers/specs/2026-08-28-concept-induction-topic-model-rebuild-design.md).

The bug these lock down: `chat_history_log` stores one FULL EXCHANGE per row
(`prompt` = Juniper, `response` = Orion). The live windowing spec used
`block_mode="turn_pairs"`, which paired two *consecutive rows* -- i.e. two
complete exchanges -- and stamped one "User:" and the other "Assistant:".
Both labels were false, both went into the text that gets embedded, and the
speaker (a recorded fact: it is the column) was destroyed at windowing time.

Fixtures are hand-computed: every expected block text/speaker list below is
written out literally rather than derived from the code under test.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest

_SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# windowing.py -> boundary_judge -> llm_client imports `orion.core.bus`, so the
# repo root has to be importable too, not just the service package.
_REPO_ROOT = os.path.dirname(os.path.dirname(_SERVICE_DIR))
for _path in (_SERVICE_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.models import WindowingSpec  # noqa: E402
from app.services.windowing import build_blocks_for_conversation  # noqa: E402

TEXT_COLUMNS = ["prompt", "response"]
SPEAKERS = {"prompt": "juniper", "response": "orion"}


def _rows():
    """Two chat_history_log-shaped rows: each holds a whole exchange."""
    return [
        {
            "correlation_id": "c1",
            "created_at": datetime(2026, 8, 28, 1, 0, tzinfo=timezone.utc),
            "prompt": "why is the topic model bullshit",
            "response": "because it is clustering 228 rows",
        },
        {
            "correlation_id": "c2",
            "created_at": datetime(2026, 8, 28, 2, 0, tzinfo=timezone.utc),
            "prompt": "and the labels",
            "response": "hidden by god-node gating",
        },
    ]


def _build(spec: WindowingSpec):
    return build_blocks_for_conversation(
        _rows(),
        spec=spec,
        text_columns=TEXT_COLUMNS,
        time_column="created_at",
        id_column="correlation_id",
    )


def _split_spec(**overrides) -> WindowingSpec:
    base = {"block_mode": "rows", "split_text_columns": True, "column_speakers": SPEAKERS}
    base.update(overrides)
    return WindowingSpec(**base)


def test_library_defaults_never_rewrite_an_already_frozen_spec():
    """A model row freezes its windowing_spec at creation and runs.py rehydrates
    it with WindowingSpec(**row). Rows written before these fields existed have
    no key for them, so a True/non-empty default would silently change how every
    pre-existing model builds documents -- no name change, no fingerprint change,
    no warning. Both new fields must therefore default to the OLD behavior.
    """
    spec = WindowingSpec()
    assert spec.split_text_columns is False
    assert spec.column_speakers == {}
    # include_roles must fail open now that the filter can actually fire.
    assert spec.include_roles == []
    # block_mode is safe to flip: every pre-existing row stores it explicitly.
    assert spec.block_mode == "rows"


def test_split_emits_one_block_per_utterance_with_its_speaker():
    blocks = _build(_split_spec())

    # 2 rows x 2 text columns = 4 utterances, hand-counted.
    assert len(blocks) == 4
    assert [b.text for b in blocks] == [
        "why is the topic model bullshit",
        "because it is clustering 228 rows",
        "and the labels",
        "hidden by god-node gating",
    ]
    assert [b.speakers for b in blocks] == [["juniper"], ["orion"], ["juniper"], ["orion"]]
    # Both utterances of one row keep pointing at that row.
    assert [b.row_ids for b in blocks] == [["c1"], ["c1"], ["c2"], ["c2"]]


@pytest.mark.parametrize(
    "mode,expected_blocks",
    # Hand-counted over the 4 utterances _rows() expands to: rows -> 4 blocks;
    # turn_pairs -> 2; triads -> 1 (the trailing 4th unit cannot fill a triad).
    [("rows", 4), ("turn_pairs", 2), ("triads", 1)],
)
def test_no_speaker_label_of_any_kind_reaches_the_embedded_text(mode, expected_blocks):
    """No role label in the vectorized text -- fabricated OR true.

    The original defect was hardcoded "User:"/"Assistant:" prefixes. The first
    fix for it replaced them with the REAL speaker, which is the same defect:
    on a corpus split roughly in half by speaker, a leading "juniper: "/"orion: "
    is a near-perfect high-IDF discriminator, so HDBSCAN can cluster on speaker
    instead of topic and tf-idf can hand the speakers a top-keyword slot. The
    speaker belongs on RowBlock.speakers, not in the text.

    The block count is asserted first on purpose. An earlier version of this
    test only looped over the blocks, and under turn_pairs the then-default
    include_roles filtered every block away -- so the loop body never ran and
    the test passed vacuously against a deliberately reintroduced bug.
    """
    blocks = _build(_split_spec(block_mode=mode, include_roles=[]))
    assert len(blocks) == expected_blocks
    for block in blocks:
        for forbidden in ("User:", "Assistant:", "juniper:", "orion:"):
            assert forbidden not in block.text
        # ...while the speaker is still recorded structurally.
        assert block.speakers


def test_unknown_speaker_is_recorded_as_unknown_not_guessed():
    # column_speakers empty -> split still happens, but nothing is asserted
    # about who spoke.
    blocks = _build(_split_spec(column_speakers={}))
    assert len(blocks) == 4
    assert blocks[0].text == "why is the topic model bullshit"
    assert blocks[0].speakers == []


def test_fused_mode_records_no_speakers_and_no_prefixes():
    """Opting out of the split means the speaker is genuinely unknown --
    the block must say so (empty list), not invent one."""
    blocks = _build(_split_spec(split_text_columns=False, block_mode="rows"))
    assert len(blocks) == 2
    assert blocks[0].text == "why is the topic model bullshit\nbecause it is clustering 228 rows"
    assert blocks[0].speakers == []
    assert blocks[0].row_ids == ["c1"]


def test_fused_mode_still_reads_a_real_per_row_role_column():
    """_role_of must not be dead code. A source that genuinely carries a
    `role`/`speaker` column keeps working -- otherwise include_roles filtering
    that used to function for such a source would go permanently inert, and its
    blocks would record no speaker despite the row stating one."""
    rows = [
        {
            "correlation_id": "r1",
            "created_at": datetime(2026, 8, 28, 5, 0, tzinfo=timezone.utc),
            "role": "Assistant",
            "prompt": "",
            "response": "a turn from a role-carrying source",
        }
    ]
    blocks = build_blocks_for_conversation(
        rows,
        spec=WindowingSpec(block_mode="rows", split_text_columns=False),
        text_columns=TEXT_COLUMNS,
        time_column="created_at",
        id_column="correlation_id",
    )
    assert len(blocks) == 1
    assert blocks[0].speakers == ["assistant"]  # normalized lowercase by _role_of
    assert "assistant" not in blocks[0].text


def test_turn_pairs_never_straddles_two_rows_when_split():
    """A row with a NULL column contributes one unit. A flat idx += 2 walk over
    units would then pair THIS row's prompt with the NEXT row's prompt and stay
    misaligned for the rest of the conversation -- the same class of defect this
    whole patch exists to fix, just relocated."""
    rows = [
        {
            "correlation_id": "a",
            "created_at": datetime(2026, 8, 28, 6, 0, tzinfo=timezone.utc),
            "prompt": "pa",
            "response": "ra",
        },
        {
            "correlation_id": "b",
            "created_at": datetime(2026, 8, 28, 7, 0, tzinfo=timezone.utc),
            "prompt": "pb",
            "response": None,  # the misaligning row
        },
        {
            "correlation_id": "c",
            "created_at": datetime(2026, 8, 28, 8, 0, tzinfo=timezone.utc),
            "prompt": "pc",
            "response": "rc",
        },
    ]
    blocks = build_blocks_for_conversation(
        rows,
        spec=_split_spec(block_mode="turn_pairs"),
        text_columns=TEXT_COLUMNS,
        time_column="created_at",
        id_column="correlation_id",
    )
    # Hand-computed: units are [pa, ra, pb, pc, rc]. Only (pa, ra) and (pc, rc)
    # are same-row pairs; the lone pb is dropped rather than mispaired.
    assert [b.row_ids for b in blocks] == [["a"], ["c"]]
    assert [b.text for b in blocks] == ["pa\nra", "pc\nrc"]
    for block in blocks:
        assert len(block.row_ids) == 1


def test_turn_pairs_over_split_units_is_one_real_exchange():
    """turn_pairs was pairing two whole exchanges. Over split units it pairs
    a prompt with its own response -- and de-duplicates the row id, so a
    2-unit block from 1 row does not claim to cover 2 rows."""
    blocks = _build(_split_spec(block_mode="turn_pairs", include_roles=["juniper", "orion"]))

    assert len(blocks) == 2
    assert blocks[0].text == (
        "why is the topic model bullshit\nbecause it is clustering 228 rows"
    )
    assert blocks[0].speakers == ["juniper", "orion"]
    assert blocks[0].row_ids == ["c1"]
    assert blocks[0].timestamps == ["2026-08-28T01:00:00+00:00"]


def test_include_roles_actually_filters_now():
    """It never did before: _role_of read `role`/`speaker` columns that do not
    exist on this source, so both roles were None and the guard
    short-circuited on every run."""
    blocks = _build(_split_spec(block_mode="turn_pairs", include_roles=["juniper"]))
    assert blocks == []


def test_include_roles_still_inert_when_speakers_unknown():
    """Fail open, not closed: an unknown speaker must not silently delete the
    corpus (AI Town passes no column_speakers)."""
    blocks = _build(
        _split_spec(block_mode="turn_pairs", column_speakers={}, include_roles=["juniper", "orion"])
    )
    assert len(blocks) == 2
    assert blocks[0].speakers == []


def test_truncation_applies_per_block():
    long_row = [
        {
            "correlation_id": "c9",
            "created_at": datetime(2026, 8, 28, 3, 0, tzinfo=timezone.utc),
            "prompt": "x" * 50,
            "response": "y" * 50,
        }
    ]
    blocks = build_blocks_for_conversation(
        long_row,
        spec=_split_spec(max_chars=20),
        text_columns=TEXT_COLUMNS,
        time_column="created_at",
        id_column="correlation_id",
    )
    assert len(blocks) == 2
    for block in blocks:
        assert len(block.text) <= 20


def test_empty_column_is_skipped_not_emitted_as_a_blank_document():
    rows = [
        {
            "correlation_id": "c3",
            "created_at": datetime(2026, 8, 28, 4, 0, tzinfo=timezone.utc),
            "prompt": "   ",
            "response": "only this one is real",
        }
    ]
    blocks = build_blocks_for_conversation(
        rows,
        spec=_split_spec(),
        text_columns=TEXT_COLUMNS,
        time_column="created_at",
        id_column="correlation_id",
    )
    assert len(blocks) == 1
    assert blocks[0].speakers == ["orion"]
    assert blocks[0].text == "only this one is real"


@pytest.mark.parametrize("mode", ["rows", "turn_pairs", "triads"])
def test_speakers_never_fabricated_for_a_column_without_a_mapping(mode):
    partial = {"prompt": "juniper"}  # response deliberately unmapped
    for block in _build(_split_spec(block_mode=mode, column_speakers=partial, include_roles=[])):
        assert set(block.speakers) <= {"juniper"}


def test_merged_blocks_do_not_overcount_source_rows():
    """min_blocks_per_segment > 1 merges blocks. Two split blocks from ONE row
    each carry that row_id, so without de-duplication provenance.row_ids -- and
    SegmentRecord.size/row_ids_count derived from it -- would claim the segment
    covers two source rows when it covers one."""
    from app.services.windowing import _chunk_blocks

    spec = _split_spec(block_mode="rows", min_blocks_per_segment=2)
    # build_blocks_for_conversation does not chunk -- _build_segments_internal
    # applies _chunk_blocks afterwards, so exercise that step directly.
    blocks = _chunk_blocks(_build(spec), spec)
    # 4 units -> 4 blocks -> chunks of 2, and each chunk is one row's own
    # prompt+response.
    assert len(blocks) == 2
    assert blocks[0].row_ids == ["c1"]
    assert blocks[0].timestamps == ["2026-08-28T01:00:00+00:00"]
    assert blocks[0].speakers == ["juniper", "orion"]
    assert blocks[1].row_ids == ["c2"]


def test_dedup_extend_survives_a_materialized_sequence():
    """The de-dup must not depend on list.extend consuming a generator lazily."""
    from app.services.windowing_provenance import dedup_extend, dedup_row_provenance

    target = ["a"]
    dedup_extend(target, ["a", "b", "b", "c", "a"])
    assert target == ["a", "b", "c"]
    dedup_extend(target, (x for x in ["c", "d"]))
    assert target == ["a", "b", "c", "d"]

    ids, ts = dedup_row_provenance(["r1", "r1", "r2"], ["t1", "t1b", "t2"])
    assert ids == ["r1", "r2"]
    assert ts == ["t1", "t2"]

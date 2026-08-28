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


def test_defaults_are_split_rows_not_turn_pairs():
    """The shipped default must be the fixed shape, not the old one."""
    spec = WindowingSpec()
    assert spec.block_mode == "rows"
    assert spec.split_text_columns is True
    assert spec.column_speakers == {}


def test_split_emits_one_block_per_utterance_with_its_speaker():
    blocks = _build(_split_spec())

    # 2 rows x 2 text columns = 4 utterances, hand-counted.
    assert len(blocks) == 4
    assert [b.text for b in blocks] == [
        "juniper: why is the topic model bullshit",
        "orion: because it is clustering 228 rows",
        "juniper: and the labels",
        "orion: hidden by god-node gating",
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
def test_no_fabricated_user_assistant_labels_anywhere(mode, expected_blocks):
    """The exact old defect: a role label the data never supported.

    The block count is asserted first on purpose. An earlier version of this
    test only looped over the blocks, and under turn_pairs the default
    include_roles filtered every block away -- so the loop body never ran and
    the test passed vacuously against a deliberately reintroduced bug.
    """
    blocks = _build(_split_spec(block_mode=mode, include_roles=[]))
    assert len(blocks) == expected_blocks
    for block in blocks:
        assert "User:" not in block.text
        assert "Assistant:" not in block.text


def test_unknown_speaker_gets_no_prefix_rather_than_a_guess():
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


def test_turn_pairs_over_split_units_is_one_real_exchange():
    """turn_pairs was pairing two whole exchanges. Over split units it pairs
    a prompt with its own response -- and de-duplicates the row id, so a
    2-unit block from 1 row does not claim to cover 2 rows."""
    blocks = _build(_split_spec(block_mode="turn_pairs", include_roles=["juniper", "orion"]))

    assert len(blocks) == 2
    assert blocks[0].text == (
        "juniper: why is the topic model bullshit\norion: because it is clustering 228 rows"
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
    assert blocks[0].text == "orion: only this one is real"


@pytest.mark.parametrize("mode", ["rows", "turn_pairs", "triads"])
def test_speakers_never_fabricated_for_a_column_without_a_mapping(mode):
    partial = {"prompt": "juniper"}  # response deliberately unmapped
    for block in _build(_split_spec(block_mode=mode, column_speakers=partial, include_roles=[])):
        assert set(block.speakers) <= {"juniper"}

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for module_name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
    sys.modules.pop(module_name)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVICE_ROOT))

from app.town_continuity import select_town_continuity, speaker_on_thread, utterances_from_row


def _row(
    *,
    thread_id,
    speaker_id,
    speaker_name,
    prompt,
    response,
    created_at,
    tags=None,
    recall_safe=True,
):
    return {
        "prompt": prompt,
        "response": response,
        "tags": tags if tags is not None else ["aitown"],
        "redaction": {"recall_safe": recall_safe},
        "client_meta": {
            "external_room": {
                "platform": "aitown",
                "room_id": "aitown-town",
                "thread_id": thread_id,
            },
            "external_participant": {
                "participant_id": speaker_id,
                "participant_name": speaker_name,
            },
        },
        "created_at": created_at,
    }


def _select(rows, *, thread_id="juniper-feld--nico-sable", speaker_id="nico-sable"):
    return select_town_continuity(
        platform="aitown",
        room_id="aitown-town",
        thread_id=thread_id,
        speaker_id=speaker_id,
        rows=rows,
    )


def _texts(turns) -> list[str]:
    return [turn.text for turn in turns]


def test_nico_juniper_pair_includes_juniper_prompt_line() -> None:
    rows = [
        _row(
            thread_id="juniper-feld--nico-sable",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt="spill the tea",
            response="the pie sat out and the crumbs were sugar",
            created_at="2026-08-29T20:00:00+00:00",
        )
    ]
    body = _select(rows)
    assert [turn.speaker for turn in body.pair_turns] == ["Juniper Feld", "Nico Sable"]
    assert [turn.other for turn in body.pair_turns] == ["Nico Sable", "Juniper Feld"]
    assert _texts(body.pair_turns) == [
        "spill the tea",
        "the pie sat out and the crumbs were sugar",
    ]
    assert body.town_turns == []


def test_nico_sofia_rows_land_in_town_turns() -> None:
    rows = [
        _row(
            thread_id="nico-sable--sofia-bell",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt="you still owe me for the trivia night",
            response="I'll settle it Friday",
            created_at="2026-08-29T19:00:00+00:00",
        )
    ]
    body = _select(rows)
    assert body.pair_turns == []
    # Same created_at: sort key is speaker, so Nico before Sofia.
    assert [turn.speaker for turn in body.town_turns] == ["Nico Sable", "Sofia Bell"]
    assert set(_texts(body.town_turns)) == {
        "you still owe me for the trivia night",
        "I'll settle it Friday",
    }


def test_sofia_cam_rows_do_not_appear_for_nico() -> None:
    rows = [
        _row(
            thread_id="juniper-feld--nico-sable",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt="hi",
            response="hey",
            created_at="2026-08-29T20:00:00+00:00",
        ),
        _row(
            thread_id="cam-lin--sofia-bell",
            speaker_id="sofia-bell",
            speaker_name="Sofia Bell",
            prompt="jailbreak",
            response="not for nico",
            created_at="2026-08-29T18:00:00+00:00",
        ),
    ]
    body = _select(rows)
    blob = " ".join(_texts(body.pair_turns) + _texts(body.town_turns))
    assert "jailbreak" not in blob
    assert "not for nico" not in blob
    assert body.town_turns == []


def test_recall_safe_false_is_skipped() -> None:
    rows = [
        _row(
            thread_id="juniper-feld--nico-sable",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt="keep me",
            response="kept",
            created_at="2026-08-29T20:00:00+00:00",
        ),
        _row(
            thread_id="juniper-feld--nico-sable",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt="drop prompt",
            response="drop response",
            created_at="2026-08-29T20:01:00+00:00",
            recall_safe=False,
        ),
    ]
    body = _select(rows)
    assert _texts(body.pair_turns) == ["keep me", "kept"]


def test_missing_aitown_tag_is_skipped() -> None:
    rows = [
        _row(
            thread_id="juniper-feld--nico-sable",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt="tagged",
            response="ok",
            created_at="2026-08-29T20:00:00+00:00",
        ),
        _row(
            thread_id="juniper-feld--nico-sable",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt="no tag prompt",
            response="no tag response",
            created_at="2026-08-29T20:01:00+00:00",
            tags=["social_room"],
        ),
    ]
    body = _select(rows)
    assert _texts(body.pair_turns) == ["tagged", "ok"]


def test_pair_and_town_caps_are_oldest_first_after_flatten() -> None:
    pair_rows = [
        _row(
            thread_id="juniper-feld--nico-sable",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt=f"j-{index}",
            response=f"n-{index}",
            created_at=f"2026-08-29T10:0{index}:00+00:00",
        )
        for index in range(9)
    ]
    town_rows = [
        _row(
            thread_id="nico-sable--sofia-bell",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt=f"sofia-{index}",
            response=f"nico-town-{index}",
            created_at=f"2026-08-29T08:0{index}:00+00:00",
        )
        for index in range(5)
    ]
    body = _select(pair_rows + town_rows)
    assert _texts(body.pair_turns) == [
        "j-0",
        "n-0",
        "j-1",
        "n-1",
        "j-2",
        "n-2",
        "j-3",
        "n-3",
    ]
    # Same created_at sorts by speaker: "Nico Sable" before "Sofia Bell".
    assert _texts(body.town_turns) == [
        "nico-town-0",
        "sofia-0",
        "nico-town-1",
        "sofia-1",
    ]


def test_speaker_on_thread_requires_exactly_two_parts() -> None:
    assert speaker_on_thread("juniper-feld--nico-sable", "nico-sable") is True
    assert speaker_on_thread("juniper-feld--nico-sable", "sofia-bell") is False
    assert speaker_on_thread("nico-sable--sofia-bell--extra", "nico-sable") is False
    assert speaker_on_thread("nico-sable", "nico-sable") is False


def test_empty_rows_yield_empty_lists() -> None:
    body = _select([])
    assert body.thread_id == "juniper-feld--nico-sable"
    assert body.speaker_id == "nico-sable"
    assert body.pair_turns == []
    assert body.town_turns == []


def test_utterances_from_row_skips_empty_text_and_clamps() -> None:
    long_line = "x" * 200
    turns = utterances_from_row(
        _row(
            thread_id="juniper-feld--nico-sable",
            speaker_id="nico-sable",
            speaker_name="Nico Sable",
            prompt="",
            response=long_line,
            created_at="2026-08-29T20:00:00+00:00",
        )
    )
    assert len(turns) == 1
    assert turns[0].speaker == "Nico Sable"
    assert turns[0].text == "x" * 160


def test_missing_json_paths_and_mismatch_are_skipped() -> None:
    bare = {
        "prompt": "ghost",
        "response": "nope",
        "tags": ["aitown"],
        "redaction": {"recall_safe": True},
        "created_at": "2026-08-29T20:00:00+00:00",
    }
    wrong_room = _row(
        thread_id="juniper-feld--nico-sable",
        speaker_id="nico-sable",
        speaker_name="Nico Sable",
        prompt="wrong",
        response="room",
        created_at="2026-08-29T20:00:00+00:00",
    )
    wrong_room["client_meta"]["external_room"]["room_id"] = "other-room"
    encoded = _row(
        thread_id="juniper-feld--nico-sable",
        speaker_id="nico-sable",
        speaker_name="Nico Sable",
        prompt="json prompt",
        response="json response",
        created_at=datetime(2026, 8, 29, 21, 0, tzinfo=timezone.utc),
    )
    encoded["tags"] = json.dumps(["aitown"])
    encoded["redaction"] = json.dumps({"recall_safe": True})
    encoded["client_meta"] = json.dumps(encoded["client_meta"])
    body = _select([bare, wrong_room, encoded])
    assert _texts(body.pair_turns) == ["json prompt", "json response"]
    assert body.pair_turns[0].created_at.endswith("+00:00")

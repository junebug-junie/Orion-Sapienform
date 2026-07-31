"""Chat-history fragments must distinguish ai-town-sourced rows from real
Juniper/hub conversation, not label every row "User: ...".

Root cause (found live 2026-07-31): fetch_sql_fragments never selected
client_meta at all, tagged every chat_history_log row with the identical
["dialogue"], and rendered every row as f"User: {prompt}\\nOrion: {response}"
regardless of source -- for an ai-town row, `prompt` is an NPC's line, not
Juniper's, so labeling it "User:" is a false claim baked directly into what
reaches the model. Confirmed live: ai-town dialogue bled into hub-mode
responses as if it were real conversation with Juniper (and vice versa).
See docs/superpowers/specs/2026-07-31-recall-aitown-source-tagging-design.md.
"""

from __future__ import annotations

from app.storage import sql_adapter


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *args, **kwargs):
        pass

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        pass


def _fetch_chat_only(monkeypatch, rows):
    monkeypatch.setattr(sql_adapter, "_connect", lambda: _FakeConn(rows))
    return sql_adapter.fetch_sql_fragments(include_chat=True, include_mirrors=False)


def test_aitown_row_gets_aitown_tag_and_correct_speaker_label(monkeypatch):
    rows = [
        (
            "trace-1",
            "You're still avoiding the sensor bus.",
            "I'm not avoiding anything.",
            "2026-07-31T00:00:00+00:00",
            {
                "external_room": {"platform": "aitown", "room_id": "aitown-town"},
                "external_participant": {
                    "participant_id": "p:6",
                    "participant_name": "Juno Park",
                    "participant_kind": "npc",
                },
            },
        )
    ]
    frags = _fetch_chat_only(monkeypatch, rows)
    assert len(frags) == 1
    frag = frags[0]
    assert "aitown" in frag.tags
    assert frag.text.startswith("[ai-town] Juno Park:")
    assert "You're still avoiding the sensor bus." in frag.text
    assert "Orion (in ai-town):" in frag.text
    # Must never claim Juniper said the NPC's line.
    assert "User:" not in frag.text
    assert frag.meta.get("platform") == "aitown"
    assert frag.meta.get("participant_name") == "Juno Park"


def test_hub_row_is_unchanged(monkeypatch):
    rows = [
        (
            "trace-2",
            "hey, Orion!",
            "Hey, Juniper.",
            "2026-07-31T00:00:00+00:00",
            None,
        )
    ]
    frags = _fetch_chat_only(monkeypatch, rows)
    assert len(frags) == 1
    frag = frags[0]
    assert frag.tags == ["dialogue"]
    assert frag.text == "User: hey, Orion!\nOrion: Hey, Juniper."
    assert frag.meta == {}


def test_row_with_no_client_meta_defaults_to_hub_labeling(monkeypatch):
    """Rows that predate the client_meta.external_room field (or any other
    platform) must default to the original "not aitown" behavior, not crash
    or silently mislabel something else as ai-town."""
    rows = [
        ("trace-3", "some old prompt", "some old response", "2026-01-01T00:00:00+00:00", {}),
    ]
    frags = _fetch_chat_only(monkeypatch, rows)
    assert len(frags) == 1
    frag = frags[0]
    assert frag.tags == ["dialogue"]
    assert frag.text == "User: some old prompt\nOrion: some old response"


def test_client_meta_as_json_string_is_parsed(monkeypatch):
    """psycopg2 returns jsonb as dict, but handle a string defensively too --
    mirrors _safe_tags's existing pattern for the tags column."""
    import json

    rows = [
        (
            "trace-4",
            "npc line",
            "orion reply",
            "2026-07-31T00:00:00+00:00",
            json.dumps(
                {
                    "external_room": {"platform": "aitown"},
                    "external_participant": {"participant_name": "Sofia Bell"},
                }
            ),
        )
    ]
    frags = _fetch_chat_only(monkeypatch, rows)
    assert len(frags) == 1
    assert "aitown" in frags[0].tags
    assert frags[0].text.startswith("[ai-town] Sofia Bell:")

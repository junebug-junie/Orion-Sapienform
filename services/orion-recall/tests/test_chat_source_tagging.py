"""Shared ai-town source detection/labeling used by every chat_history_log
reader in this service. See app/chat_source_tagging.py's module docstring
for why this is centralized (found live 2026-07-31: eight separate call
sites -- counting fetch_chat_turns_by_id's three independent downstream
consumers separately -- independently built "User: ..."-style text with no
client_meta awareness. This module exists so a fix doesn't have to be
reapplied eight times, or re-diverge across them)."""

from __future__ import annotations

import json

from app import chat_source_tagging as tagging


def _aitown_client_meta(name: str = "Nico Sable") -> dict:
    return {
        "external_room": {"platform": "aitown", "room_id": "aitown-town"},
        "external_participant": {
            "participant_id": "p:2",
            "participant_name": name,
            "participant_kind": "npc",
        },
    }


class TestChatSourcePlatform:
    def test_extracts_platform_and_name_from_dict(self):
        platform, name = tagging.chat_source_platform(_aitown_client_meta())
        assert platform == "aitown"
        assert name == "Nico Sable"

    def test_parses_json_encoded_string(self):
        platform, name = tagging.chat_source_platform(json.dumps(_aitown_client_meta("Sofia Bell")))
        assert platform == "aitown"
        assert name == "Sofia Bell"

    def test_none_defaults_to_not_aitown(self):
        assert tagging.chat_source_platform(None) == (None, None)

    def test_empty_dict_defaults_to_not_aitown(self):
        assert tagging.chat_source_platform({}) == (None, None)

    def test_malformed_string_defaults_to_not_aitown(self):
        assert tagging.chat_source_platform("not json") == (None, None)

    def test_non_dict_non_string_defaults_to_not_aitown(self):
        assert tagging.chat_source_platform(123) == (None, None)

    def test_malformed_external_room_shape_defaults_to_not_aitown(self):
        assert tagging.chat_source_platform({"external_room": "not-a-dict"}) == (None, None)

    def test_hub_platform_is_not_aitown(self):
        client_meta = {"external_room": {"platform": "hub", "room_id": "hub-direct"}}
        platform, name = tagging.chat_source_platform(client_meta)
        assert platform == "hub"
        assert name is None


class TestRenderLabeledChatText:
    def test_aitown_row_gets_marker_and_correct_speaker(self):
        text = tagging.render_labeled_chat_text("npc line", "orion reply", _aitown_client_meta())
        assert text == "[ai-town] Nico Sable: npc line\nOrion (in ai-town): orion reply"

    def test_aitown_row_without_participant_name_falls_back_to_npc(self):
        client_meta = {"external_room": {"platform": "aitown"}, "external_participant": {}}
        text = tagging.render_labeled_chat_text("line", "reply", client_meta)
        assert text.startswith("[ai-town] NPC:")

    def test_hub_row_is_unchanged(self):
        text = tagging.render_labeled_chat_text("hey, Orion!", "Hey, Juniper.", None)
        assert text == "User: hey, Orion!\nOrion: Hey, Juniper."
        assert "[ai-town]" not in text


class TestRenderQuotedChatText:
    def test_aitown_row_gets_marker_and_correct_speaker(self):
        text = tagging.render_quoted_chat_text("npc line", "orion reply", _aitown_client_meta())
        assert text == '[ai-town] Nico Sable: "npc line"\nOrion (in ai-town): "orion reply"'

    def test_hub_row_is_unchanged(self):
        text = tagging.render_quoted_chat_text("hi", "hello", None)
        assert text == 'ExactUserText: "hi"\nOrionResponse: "hello"'
        assert "[ai-town]" not in text


class TestChatSourceTags:
    def test_appends_aitown_tag_for_aitown_rows(self):
        assert tagging.chat_source_tags(_aitown_client_meta(), ["dialogue"]) == ["dialogue", "aitown"]

    def test_leaves_base_tags_unchanged_for_hub_rows(self):
        assert tagging.chat_source_tags(None, ["dialogue"]) == ["dialogue"]

    def test_does_not_mutate_base_tags_list(self):
        base = ["chat_timeline"]
        tagging.chat_source_tags(_aitown_client_meta(), base)
        assert base == ["chat_timeline"]

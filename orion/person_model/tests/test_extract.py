from orion.person_model.extract import PINNED, PERSON_LABEL, ExtractorSpec, mention_id, spans_to_mentions


def test_version_pins_revision_labels_and_threshold():
    v = PINNED.version
    assert PINNED.revision[:10] in v and f"t={PINNED.threshold}" in v
    other = ExtractorSpec(PINNED.model_id, PINNED.revision, PINNED.labels + ("x",), PINNED.threshold)
    assert other.version != v
    assert PERSON_LABEL in PINNED.labels


def test_mention_rows_are_replay_stable():
    text = "My boss, Marlow, and Zed, Zed."
    spans = [{"start": 9, "end": 15, "label": PERSON_LABEL, "score": 0.98765432}, {"start": 21, "end": 24, "label": PERSON_LABEL, "score": 0.9}]
    a = [m.model_dump_json() for m in spans_to_mentions(spans, text=text, source_kind="chat_prompt", source_id="p1", version="v")]
    b = [m.model_dump_json() for m in spans_to_mentions(list(reversed(spans)), text=text, source_kind="chat_prompt", source_id="p1", version="v")]
    assert a == b
    assert spans_to_mentions(spans, text=text, source_kind="chat_prompt", source_id="p1", version="v")[0].text == "Marlow"
    assert mention_id("chat_prompt", "p1", 9, 15, "v") != mention_id("chat_prompt", "p1", 9, 15, "v2")

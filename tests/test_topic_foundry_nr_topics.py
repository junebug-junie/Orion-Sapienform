import sys

sys.path.insert(0, "services/orion-topic-foundry")

from app.topic_engine import _parse_nr_topics, build_topic_engine


def test_parse_nr_topics_supports_auto_and_positive_int_only():
    assert _parse_nr_topics("auto") == "auto"
    assert _parse_nr_topics(" 12 ") == 12
    assert _parse_nr_topics(8) == 8
    assert _parse_nr_topics("0") is None
    assert _parse_nr_topics("bad") is None


def test_build_topic_engine_emits_nr_topics_kwarg_when_set():
    parts = build_topic_engine({"nr_topics": "auto"})
    assert parts.bertopic_kwargs.get("nr_topics") == "auto"

    parts_n = build_topic_engine({"nr_topics": 14})
    assert parts_n.bertopic_kwargs.get("nr_topics") == 14

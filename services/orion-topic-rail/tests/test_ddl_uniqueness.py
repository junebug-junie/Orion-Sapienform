from app.topic_rail.db.ddl import CHAT_TOPIC_DDL


def test_chat_topic_ddl_contains_unique_constraint():
    assert "UNIQUE (chat_id, model_version)" in CHAT_TOPIC_DDL


def test_chat_topic_ddl_contains_indexes():
    assert "ix_chat_topic_model_version" in CHAT_TOPIC_DDL
    assert "ix_chat_topic_correlation_id" in CHAT_TOPIC_DDL
    assert "ix_chat_topic_session_id" in CHAT_TOPIC_DDL
    assert "ix_chat_topic_topic_id" in CHAT_TOPIC_DDL

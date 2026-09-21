from datetime import datetime, timezone

from orion.self_knowledge_freshness import NEWEST_SELF_KNOWLEDGE_ITEM_SQL, normalize_newest


def test_sql_targets_the_right_table_and_column():
    assert "self_knowledge_items" in NEWEST_SELF_KNOWLEDGE_ITEM_SQL
    assert "created_at" in NEWEST_SELF_KNOWLEDGE_ITEM_SQL


def test_normalize_passes_none_through():
    assert normalize_newest(None) is None


def test_normalize_stamps_naive_timestamp_as_utc():
    naive = datetime(2026, 9, 19, 8, 0, 0)
    result = normalize_newest(naive)
    assert result.tzinfo is timezone.utc


def test_normalize_leaves_aware_timestamp_untouched():
    aware = datetime(2026, 9, 19, 8, 0, 0, tzinfo=timezone.utc)
    assert normalize_newest(aware) is aware

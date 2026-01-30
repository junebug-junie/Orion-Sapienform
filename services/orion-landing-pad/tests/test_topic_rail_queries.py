from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "app" / "reducers" / "topic_rail.py"
spec = spec_from_file_location("topic_rail", MODULE_PATH)
topic_rail = module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(topic_rail)


def test_build_topic_summary_query_parameters():
    sql, params = topic_rail._build_topic_summary_query(60, "model-v1", 25)

    assert "chat_topic_summary" in sql
    assert "ORDER BY doc_count DESC" in sql
    assert params == ["model-v1", 60, 25]


def test_build_topic_drift_query_parameters():
    sql, params = topic_rail._build_topic_drift_query(120, "model-v2", 5, 40)

    assert "chat_topic_session_drift" in sql
    assert "ORDER BY switch_rate DESC" in sql
    assert params == ["model-v2", 120, 5, 40]

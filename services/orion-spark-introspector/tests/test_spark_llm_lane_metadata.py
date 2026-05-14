from __future__ import annotations

from app.settings import Settings


def test_spark_introspection_llm_lane_defaults() -> None:
    s = Settings()
    assert s.spark_introspection_llm_lane == "spark"
    assert s.spark_introspection_allow_chat_fallback is False
    assert s.spark_introspection_max_tokens == 384

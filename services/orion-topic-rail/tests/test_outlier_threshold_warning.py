from app.main import TopicRailService
from app.settings import settings


def test_outlier_threshold_warning(monkeypatch):
    service = TopicRailService()
    monkeypatch.setattr(settings, "topic_rail_outlier_max_pct", 0.1)
    monkeypatch.setattr(settings, "topic_rail_outlier_enabled", True)

    called = {"warn": False}

    def fake_warning(*args, **kwargs):
        called["warn"] = True

    monkeypatch.setattr(service, "publisher", None)
    monkeypatch.setattr("app.main.logger.warning", fake_warning)

    service._warn_outliers(outlier_count=5, outlier_pct=0.5)
    assert called["warn"] is True

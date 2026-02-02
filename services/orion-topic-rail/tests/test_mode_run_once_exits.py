from app.main import TopicRailService
from app.settings import settings


def test_mode_run_once_exits(monkeypatch):
    monkeypatch.setattr(settings, "topic_rail_mode", "daemon")
    monkeypatch.setattr(settings, "topic_rail_run_once", True)

    service = TopicRailService()

    calls = {"iterations": 0}

    def fake_iteration():
        calls["iterations"] += 1
        return 0

    monkeypatch.setattr(service, "_run_single_iteration", fake_iteration)
    monkeypatch.setattr(service, "_sleep", lambda: None)
    monkeypatch.setattr(service.writer, "ensure_tables_exist", lambda: None)

    service.run()

    assert calls["iterations"] == 1


def test_mode_backfill_exits(monkeypatch):
    monkeypatch.setattr(settings, "topic_rail_mode", "backfill")
    monkeypatch.setattr(settings, "topic_rail_force_refit", False)

    service = TopicRailService()

    monkeypatch.setattr(service.model_store, "exists", lambda _: True)
    monkeypatch.setattr(service, "_assign_only", lambda: 0)
    monkeypatch.setattr(service.writer, "ensure_tables_exist", lambda: None)

    service.run()

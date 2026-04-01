from __future__ import annotations

import logging

from app import main as exec_main


def _clear_graphdb_env(monkeypatch) -> None:
    for key in (
        "AUTONOMY_REPOSITORY_BACKEND",
        "GRAPHDB_QUERY_ENDPOINT",
        "GRAPHDB_URL",
        "GRAPHDB_REPO",
        "GRAPHDB_USER",
        "GRAPHDB_PASS",
        "CONCEPT_PROFILE_GRAPHDB_ENDPOINT",
        "CONCEPT_PROFILE_GRAPHDB_URL",
        "CONCEPT_PROFILE_GRAPHDB_REPO",
        "CONCEPT_PROFILE_GRAPHDB_USER",
        "CONCEPT_PROFILE_GRAPHDB_PASS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_autonomy_graph_probe_configured_success(monkeypatch, caplog) -> None:
    _clear_graphdb_env(monkeypatch)
    monkeypatch.setenv("AUTONOMY_REPOSITORY_BACKEND", "graph")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_URL", "http://orion-athena-graphdb:7200")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_REPO", "collapse")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_USER", "admin")
    monkeypatch.setenv("CONCEPT_PROFILE_GRAPHDB_PASS", "super-secret")

    class _Resp:
        status_code = 200

    def _fake_get(url, timeout, auth):
        assert url == "http://orion-athena-graphdb:7200/repositories/collapse"
        assert timeout <= 4.0
        assert auth == ("admin", "super-secret")
        return _Resp()

    monkeypatch.setattr(exec_main.requests, "get", _fake_get)
    caplog.set_level(logging.INFO)

    exec_main._run_autonomy_graph_probe()

    assert "autonomy_graph_probe backend=graph endpoint_present=yes repo=collapse auth_present=yes source=concept_profile_fallback" in caplog.text
    assert "autonomy_graph_probe result=ok endpoint=http://orion-athena-graphdb:7200/repositories/collapse" in caplog.text
    assert "super-secret" not in caplog.text


def test_autonomy_graph_probe_unconfigured_graceful(monkeypatch, caplog) -> None:
    _clear_graphdb_env(monkeypatch)
    monkeypatch.setenv("AUTONOMY_REPOSITORY_BACKEND", "graph")
    caplog.set_level(logging.INFO)

    exec_main._run_autonomy_graph_probe()

    assert "autonomy_graph_probe backend=graph endpoint_present=no repo=collapse auth_present=no source=unconfigured" in caplog.text
    assert "autonomy_graph_probe result=fail reason=graph_not_configured endpoint=graphdb:unconfigured repo=collapse" in caplog.text


def test_autonomy_graph_probe_failure_logs_without_secret(monkeypatch, caplog) -> None:
    _clear_graphdb_env(monkeypatch)
    monkeypatch.setenv("AUTONOMY_REPOSITORY_BACKEND", "graph")
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    monkeypatch.setenv("GRAPHDB_USER", "admin")
    monkeypatch.setenv("GRAPHDB_PASS", "top-secret")

    def _fake_get(url, timeout, auth):
        raise RuntimeError("boom")

    monkeypatch.setattr(exec_main.requests, "get", _fake_get)
    caplog.set_level(logging.INFO)

    exec_main._run_autonomy_graph_probe()

    assert "autonomy_graph_probe result=fail reason=RuntimeError endpoint=http://graphdb:7200/repositories/collapse repo=collapse" in caplog.text
    assert "top-secret" not in caplog.text

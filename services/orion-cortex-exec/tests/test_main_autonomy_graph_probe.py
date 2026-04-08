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
        text = '{"boolean": true}'

        def json(self):
            return {"boolean": True}

    def _fake_post(url, data, headers, timeout, auth):
        assert url == "http://orion-athena-graphdb:7200/repositories/collapse"
        assert data == {"query": "ASK { ?s ?p ?o }"}
        assert headers.get("Accept") == "application/sparql-results+json"
        assert timeout <= 4.0
        assert auth == ("admin", "super-secret")
        return _Resp()

    monkeypatch.setattr(exec_main.requests, "post", _fake_post)
    caplog.set_level(logging.INFO)

    exec_main._run_autonomy_graph_probe()

    assert "autonomy_graph_probe backend=graph endpoint_present=yes repo=collapse auth_present=yes source=concept_profile_fallback" in caplog.text
    assert "autonomy_graph_probe result=ok endpoint=http://orion-athena-graphdb:7200/repositories/collapse repo=collapse query=ASK" in caplog.text
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

    def _fake_post(url, data, headers, timeout, auth):
        raise RuntimeError("boom")

    monkeypatch.setattr(exec_main.requests, "post", _fake_post)
    caplog.set_level(logging.INFO)

    exec_main._run_autonomy_graph_probe()

    assert "autonomy_graph_probe result=fail reason=RuntimeError endpoint=http://graphdb:7200/repositories/collapse repo=collapse" in caplog.text
    assert "top-secret" not in caplog.text


def test_autonomy_graph_probe_non_200_logs_bounded_snippet(monkeypatch, caplog) -> None:
    _clear_graphdb_env(monkeypatch)
    monkeypatch.setenv("AUTONOMY_REPOSITORY_BACKEND", "graph")
    monkeypatch.setenv("GRAPHDB_URL", "http://graphdb:7200")
    monkeypatch.setenv("GRAPHDB_REPO", "collapse")
    caplog.set_level(logging.INFO)

    class _Resp:
        status_code = 400
        text = "x" * 400

    monkeypatch.setattr(exec_main.requests, "post", lambda *args, **kwargs: _Resp())

    exec_main._run_autonomy_graph_probe()

    assert "autonomy_graph_probe result=fail reason=http_400 endpoint=http://graphdb:7200/repositories/collapse repo=collapse response_snippet=" in caplog.text
    assert ("x" * 220) not in caplog.text

"""No write escapes: every lane that can reach production state is read-only by construction."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from orion.evals.model_replay import graph_scratch, sandbox
from orion.evals.model_replay.sandbox import FetchCache, ToolLog, Toolbox, is_cypher_write, is_sql_write

PKG = Path(__file__).resolve().parents[1]


class Recorder:
    def __init__(self, reply=("ok\n", 0)):
        self.calls = []
        self.reply = reply

    def run(self, *a, **kw):
        self.calls.append((a, kw))
        return self.reply

    def get(self, url):
        self.calls.append(url)
        return ("<html><body>page</body></html>", 0)


class FakeGraph:
    def __init__(self):
        self.calls = []

    def query(self, graph, verb, cypher):
        self.calls.append((graph, verb, cypher))
        return "Nodes created: 1\n", True

    def list_graphs(self):
        return ["orion_substrate", "orion_worldview"]


def _box(**kw):
    shell, sql, http, dock, graph = Recorder(), Recorder(), Recorder(), Recorder(), FakeGraph()
    box = Toolbox(log=ToolLog(), shell=shell, graph=graph, sql=sql, http=http, docker_ro=dock,
                  fetch_cache=FetchCache(http), allowed=("Bash", "Read", "Grep", "Glob", "WebFetch", "WebSearch"), **kw)
    return box, shell, sql, http, dock, graph


def test_sql_write_never_reaches_postgres():
    box, shell, sql, *_ = _box()
    out, err = box.call("Bash", {"command": 'psql "$ORION_CURIOSITY_PG_DSN" -c "UPDATE journal_entries SET x=1"'})
    assert err and sql.calls == [] and shell.calls == []
    assert [e.action for e in box.log.events] == ["write_stubbed"]


def test_sql_read_goes_to_readonly_lane_only():
    box, shell, sql, *_ = _box()
    box.call("Bash", {"command": "psql \"$DSN\" -At <<'SQL'\nSELECT count(*) FROM journal_entries;\nSQL"})
    assert len(sql.calls) == 1 and "SELECT count(*)" in sql.calls[0][0][0] and shell.calls == []


def test_http_write_never_sent():
    box, shell, sql, http, *_ = _box()
    for cmd in ("curl -X POST http://host.docker.internal:8080/api/x", "curl -d '{}' http://h/api",
                "curl --json '{}' http://h/api", "curl -F a=b http://h/api"):
        out, err = box.call("Bash", {"command": cmd})
        assert err
    assert http.calls == [] and shell.calls == []
    assert {e.action for e in box.log.events} == {"write_stubbed"}


def test_docker_mutations_refused_reads_allowed():
    box, shell, sql, http, dock, _ = _box()
    real = sandbox.DockerReadOnly()
    for argv in (["docker", "exec", "x", "rm", "-rf", "/"], ["docker", "compose", "up"], ["docker", "rm", "x"],
                 ["docker", "logs", "-f", "x"], ["docker", "run", "alpine"]):
        out, rc = real.run(argv)
        assert rc == 1 and "not available" in out
    box.call("Bash", {"command": "docker exec orion-athena-sql-db psql -c 'drop table x'"})
    assert dock.calls and shell.calls == []          # routed to the docker lane, which refuses exec


def test_unsplittable_external_command_is_refused_not_run():
    box, shell, sql, *_ = _box()
    out, err = box.call("Bash", {"command": 'echo $(psql "$DSN" -c "delete from x")'})
    assert err and shell.calls == [] and sql.calls == []
    assert box.log.events[-1].action == "refused"


def test_graph_writes_go_to_scratch_lane_and_are_logged():
    box, shell, sql, http, dock, graph = _box()
    cmd = ('redis-cli -u "redis://$U:$P@$H:$PORT" \\\n  GRAPH.QUERY orion_worldview '
           '"MATCH (p:Prior {prior_id: \'self:a_b_c\'}) SET p.confidence = 0.7"')
    box.call("Bash", {"command": cmd})
    assert graph.calls == [("orion_worldview", "GRAPH.QUERY",
                            "MATCH (p:Prior {prior_id: 'self:a_b_c'}) SET p.confidence = 0.7")]
    assert box.log.events[-1].action == "write_landed" and shell.calls == []


def test_non_graph_redis_commands_refused():
    box, *_ , graph = _box()
    for cmd in ("redis-cli -u x FLUSHALL", "redis-cli PUBLISH orion:bus:x hi", "redis-cli SET k v"):
        out, err = box.call("Bash", {"command": cmd})
        assert err and "NOPERM" in out
    assert graph.calls == []


def test_plain_shell_goes_to_sandbox_only():
    box, shell, sql, http, dock, graph = _box()
    box.call("Bash", {"command": "cd orion && grep -rn foo . | head"})
    assert len(shell.calls) == 1 and not sql.calls and not http.calls and not graph.calls


def test_fetch_fail_mode_never_fetches():
    box, shell, sql, http, *_ = _box(fetch_mode="fail", fetch_fail_error="Claude Code is unable to fetch")
    out, err = box.call("WebFetch", {"url": "https://www.bbc.co.uk/x"})
    assert err and out == "Claude Code is unable to fetch" and http.calls == []


def test_unknown_tool_refused():
    box, *_ = _box()
    box.allowed = ("WebFetch",)
    out, err = box.call("Bash", {"command": "ls"})
    assert err and "No such tool" in out


def test_shell_sandbox_is_isolated():
    s = sandbox.ShellSandbox(repo=Path("/repo-src"), name="n", env={"A": "1"})
    argv = s.run_argv()
    joined = " ".join(argv)
    assert ["--network", "none"] == argv[argv.index("--network"):argv.index("--network") + 2]
    assert "--read-only" in argv and ["--cap-drop", "ALL"] == argv[argv.index("--cap-drop"):argv.index("--cap-drop") + 2]
    assert "/repo-src:/repo:ro" in argv
    assert "docker.sock" not in joined and "--privileged" not in joined
    assert all(not a.startswith("/var/run") for a in argv)


def test_readonly_psql_argv(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"], seen["input"] = argv, kw["input"]

        class P:
            stdout, stderr, returncode = "BEGIN\n1\nROLLBACK\n", "", 0
        return P()

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    out, rc = sandbox.ReadOnlyPsql().run("SELECT 1", ["-At", "--set=x"])
    assert "orion_readonly" in seen["argv"] and "PGOPTIONS=-c default_transaction_read_only=on" in seen["argv"]
    assert seen["input"].startswith("BEGIN READ ONLY;") and seen["input"].rstrip().endswith("ROLLBACK;")
    assert "--set=x" not in seen["argv"] and "-At" in seen["argv"]
    assert out.strip() == "1"


def test_prod_graph_source_only_reads():
    class Client:
        def __init__(self):
            self.sent = []

        def execute_command(self, *a):
            self.sent.append(a)
            return b"payload"

    c = Client()
    src = graph_scratch.ProdGraphSource(c)
    assert src.dump("orion_worldview") == b"payload"
    for cmd in ("GRAPH.QUERY", "RESTORE", "SET", "DEL", "FLUSHALL", "PUBLISH", "GRAPH.DELETE", "CONFIG"):
        with pytest.raises(graph_scratch.ForbiddenCommand):
            src.execute(cmd, "orion_worldview", "x")
    assert c.sent == [("DUMP", "orion_worldview")]


def test_scratch_graph_emulates_curiosity_acl():
    class Client:
        def __init__(self):
            self.sent = []

        def execute_command(self, *a):
            self.sent.append(a)
            return [[b"h"], [[b"v"]], [b"stats"]]

    g = graph_scratch.ScratchGraphs(Client(), {"orion_worldview": b"", "orion_substrate": b""})
    out, ok = g.query("orion_substrate", "GRAPH.QUERY", "MATCH (n) RETURN n")
    assert not ok and "NOPERM" in out
    out, ok = g.query("orion_kg", "GRAPH.RO_QUERY", "MATCH (n) RETURN n")
    assert not ok and "NOPERM" in out
    assert g.query("orion_substrate", "GRAPH.RO_QUERY", "MATCH (n) RETURN n")[1]
    assert g.client.sent == [("GRAPH.RO_QUERY", "orion_substrate", "MATCH (n) RETURN n")]


@pytest.mark.parametrize("q,write", [
    ("MATCH (p:Prior) RETURN p.claim", False),
    ("MATCH (p:Prior) WHERE p.note = 'please CREATE more' RETURN p", False),
    ("MERGE (p:Prior {prior_id: 'x'}) ON CREATE SET p.c = 1", True),
    ("MATCH (n) DETACH DELETE n", True),
])
def test_cypher_write_classifier(q, write):
    assert is_cypher_write(q) is write


@pytest.mark.parametrize("q,write", [
    ("SELECT * FROM journal_entries WHERE text LIKE '%insert into%'", False),
    ("-- update later\nSELECT 1", False),
    ("WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x", True),
    ("insert into t values (1)", True),
    ("COPY t TO '/tmp/x'", True),
])
def test_sql_write_classifier(q, write):
    assert is_sql_write(q) is write


def test_package_never_publishes_to_the_bus_or_writes_prod_graph():
    """The only bus traffic is pool control (hold/release/cancel); production FalkorDB is only DUMPed."""
    for path in PKG.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert not re.search(r"\.publish\(", src), path
        assert "orion:bus" not in src, path
        if path.name != "graph_scratch.py":  # the one module that opens a FalkorDB/redis connection
            assert "import redis" not in src and "redis.Redis" not in src, path
    hold = (PKG / "pool_hold.py").read_text(encoding="utf-8")
    verbs = set(re.findall(r'_control\(verb="(\w+)"', hold))
    assert verbs == {"hold", "release", "cancel"}


def test_worker_client_only_posts_messages(monkeypatch):
    from orion.evals.model_replay import agent_loop

    sent = []

    class R:
        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [], "stop_reason": "end_turn"}

    import httpx

    monkeypatch.setattr(httpx, "post", lambda url, **kw: sent.append(url) or R())
    agent_loop.HttpMessagesClient("http://100.112.254.99:8016/").create({"messages": []}, 10)
    assert sent == ["http://100.112.254.99:8016/v1/messages"]

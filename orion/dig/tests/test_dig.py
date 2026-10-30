from __future__ import annotations

import json
import os
import subprocess
from datetime import date
from pathlib import Path

import pytest
import yaml

from orion.dig import __main__ as dig
from orion.dig.brief import DIG_BRIEF_LINE, MAX_BRIEF_CHARS, append_dig_harness_brief


def _commit(repo: Path, when: str, files: dict[str, str], msg: str) -> None:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", msg], check=True, env=env)


CATALOG = yaml.safe_dump({"channels": [
    {"name": "orion:a:x", "producer_services": ["orion-a"], "consumer_services": ["orion-b"]},
    {"name": "orion:a:y", "producer_services": ["orion-b"], "consumer_services": ["orion-a"]},
]})


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _commit(repo, "2025-07-01T18:00:00Z", {"services/orion-a/app/main.py": "", "README.md": "day one"}, "first")
    # 23:00 Denver on 07-01 (MDT, -6) is 05:00Z on 07-02: still 07-01's last commit
    _commit(repo, "2025-07-02T05:00:00Z", {"orion/bus/channels.yaml": CATALOG}, "late on 07-01")
    _commit(repo, "2025-07-03T18:00:00Z", {"services/orion-b/app/main.py": ""}, "second service")
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORION_DIG_REPO", str(repo))
    monkeypatch.setenv("ORION_DIG_NOW_ROOT", str(repo))  # main checked out = today
    return tmp_path


def run(capsys, *argv):
    code = dig.main(["--dig", "t", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_open_takes_the_last_commit_of_the_denver_day_as_plain_files(world, capsys):
    code, out, _ = run(capsys, "open", "2025-07-01")
    assert code == 0 and "of 2025-07-02T05:00:00" in out  # the 23:00-Denver commit, by its UTC date
    snap = world / "home" / "past" / "2025-07-01"
    assert (snap / "orion" / "bus" / "channels.yaml").exists()  # the 23:00 Denver commit
    assert not (snap / "services" / "orion-b").exists()          # 07-03 not included
    assert not (snap / ".git").exists()                          # no git link back
    meta = json.loads((snap / dig.MARKER).read_text())
    assert meta["date"] == "2025-07-01" and meta["commit_date"].startswith("2025-07-02T05:00")
    assert "already open" in run(capsys, "open", "2025-07-01")[1]


def test_open_refuses_past_the_cap_and_close_frees_a_slot(world, capsys):
    assert run(capsys, "open", "2025-07-01")[0] == 0
    assert run(capsys, "open", "2025-07-03")[0] == 0
    code, _, err = run(capsys, "open", "2025-07-02")
    assert code == 1 and "already open" in err
    assert "closed 2025-07-01" in run(capsys, "close", "2025-07-01")[1]
    assert run(capsys, "open", "2025-07-02")[0] == 0
    assert "2025-07-02" in run(capsys, "list")[1]


def test_sweep_removes_only_stale_exports(world, capsys):
    run(capsys, "open", "2025-07-01")
    snap = world / "home" / "past" / "2025-07-01"
    meta = json.loads((snap / dig.MARKER).read_text())
    assert dig.sweep(now_epoch=meta["opened_epoch"] + 60) == []
    assert dig.sweep(now_epoch=meta["opened_epoch"] + dig.SNAPSHOT_TTL_SEC + 1) == ["2025-07-01"]
    assert not snap.exists()


def test_remove_refuses_anything_that_is_not_an_export(world):
    stranger = world / "home" / "past" / "keep-me"
    stranger.mkdir(parents=True)
    with pytest.raises(dig.DigError):
        dig._remove(stranger)
    with pytest.raises(dig.DigError):
        dig._remove(world / "repo")
    assert stranger.exists()


def test_compare_names_what_was_added_since(world, capsys):
    run(capsys, "open", "2025-07-01")
    code, out, _ = run(capsys, "compare", "2025-07-01")
    assert code == 0
    assert "services: +1 -0 =1" in out and "added: orion-b" in out
    full = json.loads(Path(out.strip().splitlines()[-1].split(": ", 1)[1]).read_text())
    assert full["diff"]["service"]["added"] == ["orion-b"]


def test_inventory_needs_an_open_export(world, capsys):
    code, _, err = run(capsys, "inventory", "2025-07-01")
    assert code == 1 and "python -P -m orion.dig open 2025-07-01" in err
    code, out, _ = run(capsys, "inventory", "now")
    assert code == 0 and "services: 2" in out


def test_history_lists_commits_and_prs(world, capsys, monkeypatch):
    monkeypatch.setattr(dig, "_gh_merged", lambda s, e: [{"number": 7, "title": "add b", "mergedAt": "2025-07-03T18:00:00Z", "body": "x" * 9000}])
    code, out, _ = run(capsys, "history", "2025-07-01", "2025-07-03")
    assert code == 0 and "1 merged PRs (GitHub, live), 3 commits on main (clone never fetched)" in out and "#7 2025-07-03 add b" in out
    assert "x" * 100 not in out  # bodies only in the file


def test_dates_are_checked(world, capsys):
    assert "before the first commit" in run(capsys, "open", "2025-01-01")[2]
    assert "in the future" in run(capsys, "open", "2999-01-01")[2]
    assert "not a date" in run(capsys, "open", "yesterday")[2]
    assert run(capsys, "frobnicate")[0] == 2


def test_body_of_a_past_day_uses_that_days_catalog(world, capsys):
    run(capsys, "open", "2025-07-01")
    code, out, _ = run(capsys, "body", "2025-07-01")
    assert code == 0 and "catalog only" in out and "organs 1" in out


def test_body_now_without_a_graph_address_refuses(world, capsys, monkeypatch):
    monkeypatch.delenv("FALKORDB_URI", raising=False)
    code, _, err = run(capsys, "body")
    assert code == 1 and "FALKORDB_URI" in err


def test_notes_are_shared_within_a_dig(world, capsys):
    assert "no notes yet" in run(capsys, "notes")[1]
    run(capsys, "notes", "append", "what it was: no scheduler yet")
    assert "no scheduler yet" in run(capsys, "notes")[1]


def test_printout_caps_keep_the_file_path():
    text = dig.capped([f"line {i} " + "z" * 200 for i in range(100)], tail="full result: /f.json")
    assert len(text.splitlines()) <= dig.MAX_LINES and len(text) <= dig.MAX_CHARS
    assert text.endswith("full result: /f.json") and "the rest is in the file" in text


def test_help_fits_and_teaches_the_source_labels(capsys):
    code, out, _ = run(capsys, "help")
    assert code == 0 and len(out) <= 2500
    for label in ("code_then", "pr_text", "lived_record", "now", "reflection"):
        assert label in out


def test_brief_is_one_short_line_off_on_reading_only_and_by_flag(monkeypatch):
    assert len(DIG_BRIEF_LINE) <= MAX_BRIEF_CHARS
    # Juniper 2026-10-30: Orion must know they can ask what is using their power.
    assert "orion.dig body-economy hosts" in DIG_BRIEF_LINE and "using your power" in DIG_BRIEF_LINE
    assert "what your activity costs" in DIG_BRIEF_LINE
    parts: list[str] = []
    append_dig_harness_brief(parts)
    assert parts == [DIG_BRIEF_LINE]
    parts = []
    append_dig_harness_brief(parts, reading_only=True)
    assert parts == []
    monkeypatch.setenv("HARNESS_FCC_DIG_BRIEF_ENABLED", "false")
    append_dig_harness_brief(parts)
    assert parts == []


def test_harness_prefix_carries_the_line(monkeypatch):
    from orion.harness.prefix import compile_harness_prefix
    from orion.harness.tests.fixtures import make_thought
    from orion.schemas.harness_finalize import HarnessRepairOverlayV1

    monkeypatch.delenv("HARNESS_FCC_DIG_BRIEF_ENABLED", raising=False)
    prompt = compile_harness_prefix(make_thought(), repair_overlay=HarnessRepairOverlayV1(), user_message="hi")
    assert prompt.count(DIG_BRIEF_LINE) == 1
    quiet = compile_harness_prefix(make_thought(), repair_overlay=HarnessRepairOverlayV1(),
                                   user_message="hi", reading_only=True)
    assert DIG_BRIEF_LINE not in quiet


# --- body-economy (skills.physiology.state.v1 from Bash) ------------------------------------

_HOSTS_FIXTURE = Path(__file__).resolve().parents[2] / "physiology/evals/fixtures/physiology_report_live_2026-10-30_24h.json"
_REPORT_FIXTURE = Path(__file__).resolve().parents[2] / "physiology" / "evals" / "fixtures" / \
    "physiology_report_live_2026-10-29_24h.json"


@pytest.fixture
def report_server(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list[str] = []
    body = _REPORT_FIXTURE.read_bytes()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ORION_PHYSIOLOGY_REPORT_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    yield seen
    srv.shutdown()


def test_body_economy_prints_text_and_writes_the_full_report(world, capsys, report_server):
    code, out, _ = run(capsys, "body-economy")
    assert code == 0 and "Binding now:" in out and "Money:" in out
    assert len(out.splitlines()) <= dig.MAX_LINES + 1 and "full report: " in out
    path = Path(out.strip().splitlines()[-1].split("full report: ", 1)[1])
    assert json.loads(path.read_text())["schema_version"] == "physiology.report.v1"
    assert "sections=limits%2Chosts%2Cmoney%2Cwants%2Cwaste%2Cservices" in report_server[0]


def test_body_economy_sections_window_family(world, capsys, report_server):
    code, out, _ = run(capsys, "body-economy", "queue_wait", "decisions", "--window", "7d", "--family", "hub.outreach")
    assert code == 0 and "Queue wait" in out and "Bid decisions" in out
    assert "window=7d" in report_server[0] and "family=hub.outreach" in report_server[0]


def test_body_economy_hosts_host_answers_what_is_spinning_circe(world, capsys, monkeypatch, tmp_path):
    """Juniper 2026-10-30: Orion must be able to check what is spinning a machine from Bash."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    body = _HOSTS_FIXTURE.read_bytes()
    seen: list[str] = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("ORION_PHYSIOLOGY_REPORT_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    try:
        code, out, _ = run(capsys, "body-economy", "hosts", "--host", "circe")
    finally:
        srv.shutdown()
    assert code == 0 and "sections=hosts" in seen[0]
    lines = out.splitlines()
    circe = next(h for h in json.loads(body)["hosts"] if h["host"] == "circe")
    head = f"circe draws {circe['chassis_w_now']:.0f} W now vs 541 W at rest (+{circe['above_rest_w']:.0f} W"
    assert "machine circe" in lines[0] and lines[2].startswith(head)
    assert "athena draws" not in out and "vision-council" in out and "- rest: pinned" in out
    assert "measured:" in out and "Estimated:" in out
    assert "body-economy-24h-circe" in lines[-1]


def test_body_economy_help_lists_sections_and_flags(world, capsys):
    code, out, _ = run(capsys, "body-economy", "--help")
    assert code == 0 and out.strip() == dig.BODY_ECONOMY_HELP
    for word in ("hosts", "processes", "unit_costs", "--host", "--window", "--family", "measured", "estimate"):
        assert word in out, word
    code, _, err = run(capsys, "body-economy", "--host")
    assert code == 1 and "--host needs a machine name" in err
    code, _, err = run(capsys, "body-economy", "--hots", "circe")
    assert code == 1 and "--help lists them" in err
    code, out, _ = run(capsys, "help")
    assert "body-economy --help" in out


def test_body_economy_refuses_bad_input_and_says_when_unreachable(world, capsys, monkeypatch):
    code, _, err = run(capsys, "body-economy", "--window", "1y")
    assert code == 1 and "--window" in err
    code, _, err = run(capsys, "body-economy", "feelings")
    assert code == 1 and "unknown section" in err
    monkeypatch.setenv("ORION_PHYSIOLOGY_REPORT_URL", "http://127.0.0.1:9")
    code, _, err = run(capsys, "body-economy")
    assert code == 1 and "unreachable" in err

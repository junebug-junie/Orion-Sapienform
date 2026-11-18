import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "session_start_open_tickets.py"
_spec = importlib.util.spec_from_file_location("session_start_open_tickets", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _issue(n, title, *labels):
    return {"number": n, "title": title, "labels": [{"name": l} for l in labels]}


def test_empty_board_says_nothing():
    assert mod.summarize([]) == ""


def test_counts_kinds_and_lists_juniper_blockers():
    out = mod.summarize([
        _issue(1, "a", "follow-up"),
        _issue(2, "b", "follow-up", "needs-juniper"),
        _issue(3, "c", "verify-live"),
    ])
    assert out.startswith("Open tickets: 3 (2 follow-up, 1 verify-live)")
    assert "Waiting on Juniper (1):" in out
    assert "#2 b" in out
    assert "#1 a" not in out


def test_long_blocker_list_is_capped():
    issues = [_issue(n, f"t{n}", "needs-juniper") for n in range(20)]
    out = mod.summarize(issues)
    assert f"...and {20 - mod.MAX_LISTED} more" in out
    assert out.count("\n  #") == mod.MAX_LISTED


def test_main_targets_repo_explicitly(monkeypatch, capsys):
    seen = {}

    class R:
        stdout = '[{"number": 7, "title": "x", "labels": [{"name": "needs-juniper"}]}]'

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.main()
    assert seen["argv"][3:5] == ["-R", mod.REPO]
    assert "#7 x" in capsys.readouterr().out


def test_main_is_silent_when_gh_fails(monkeypatch, capsys):
    def boom(*a, **k):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(mod.subprocess, "run", boom)
    mod.main()
    assert capsys.readouterr().out == ""

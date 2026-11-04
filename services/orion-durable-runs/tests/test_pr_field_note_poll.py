"""PR field notes Patch 2: the 15-min GitHub poll (app/pr_field_note_poll.py) and the one durable-runs
module that touches the ledger (app/pr_field_note_ledger.py).

What must hold: GitHub's public API is the only merge source; a dependency-only / tiny docs-only PR gets
a code record and never a run (decision 2); any other PR with no record gets exactly one run; nothing
merged before PR_FIELD_NOTES_NOT_BEFORE is touched (decision 3); runs go one at a time; a failed PR gets
one more run later; a second poll submits nothing new; GitHub or ledger trouble ends the tick quietly.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.pr_field_note_ledger import FieldNoteLedger, is_code_record_pr
from app.pr_field_note_poll import (
    MAX_RUNS_PER_PR, FieldNotePoller, GitHubPublic, GitHubUnavailable, gh_shape, next_run_id, run_request,
)
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.pr_field_note_run import base_run_id, pr_number_of_run

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
NOT_BEFORE = datetime(2026, 11, 4, 6, 0, tzinfo=timezone.utc)


def settings(**kw):
    base = dict(field_note_lookback_hours=72.0, field_note_not_before=NOT_BEFORE, field_note_poll_sec=900.0,
                field_note_github_repo="o/r", field_note_turn_timeout_sec=420.0)
    return SimpleNamespace(**{**base, **kw})


def item(n, *, hours_ago=1.0, title=None):
    return {"number": n, "title": title or f"feat: thing {n}", "body": f"body of {n}",
            "merged_at": (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")}


CODE_FILES = [{"filename": "services/orion-hub/app/x.py", "additions": 30, "deletions": 2}]
TINY_DOCS = [{"filename": "docs/superpowers/pr-reports/x.md", "additions": 3, "deletions": 1}]
DEPS = [{"filename": "services/orion-hub/requirements.txt", "additions": 1, "deletions": 1}]


class FakeGitHub:
    def __init__(self, merged, files=None, *, fail=None):
        self.merged, self.file_map, self.fail = merged, files or {}, fail
        self.calls = 0
        self.log: list[str] = []

    def merged_since(self, since):
        self.calls += 1
        self.log.append(f"search:{since.isoformat()}")
        if self.fail:
            raise GitHubUnavailable(self.fail)
        return list(self.merged)

    def files(self, n):
        self.calls += 1
        self.log.append(f"files:{n}")
        return self.file_map.get(n, CODE_FILES)

    def pull(self, n):
        self.calls += 1
        self.log.append(f"pull:{n}")
        lines = sum(f["additions"] + f["deletions"] for f in self.file_map.get(n, CODE_FILES))
        return {"merge_commit_sha": f"{n:040d}", "merged_at": None, "changed_files": len(self.file_map.get(n, [])),
                "additions": lines, "deletions": 0, "head": {"ref": f"docs/b{n}"}, "title": f"docs: {n}", "body": "b"}

    def commits(self, n):
        self.calls += 1
        self.log.append(f"commits:{n}")
        return [{"commit": {"author": {"email": "athena@orion"}}}]


class FakePool:
    """``durable_admission_runs`` as a list of rows; answers the poller's two queries."""

    def __init__(self, rows=None):
        self.rows = list(rows or [])

    @asynccontextmanager
    async def connection(self):
        pool = self

        class Cur:
            def __init__(self, rows):
                self._rows = rows

            async def fetchall(self):
                return self._rows

            async def fetchone(self):
                return self._rows[0] if self._rows else None

        class Conn:
            async def execute(self, sql, params):
                if "max(created_at)" in sql:
                    ts = [r["created_at"] for r in pool.rows]
                    return Cur([{"last": max(ts) if ts else None}])
                since = params[1]
                return Cur([r for r in pool.rows if r["created_at"] >= since])

        yield Conn()


class Submitted:
    def __init__(self, pool):
        self.pool, self.requests = pool, []

    async def __call__(self, request: DurableRunRequestV1):
        if any(r["run_id"] == request.run_id for r in self.pool.rows):
            raise ValueError("run_id already exists with a different request")
        self.requests.append(request)
        self.pool.rows.append({"run_id": request.run_id, "terminal": None, "created_at": NOW, "updated_at": NOW})
        return {"run_id": request.run_id}


def poller(tmp_path, merged, *, rows=None, files=None, fail=None, **kw):
    pool = FakePool(rows)
    submit = Submitted(pool)
    gh = FakeGitHub(merged, files, fail=fail)
    p = FieldNotePoller(settings=settings(**kw), pool=pool, submit=submit, ledger=FieldNoteLedger(tmp_path),
                        github=gh, now=lambda: NOW)
    return p, submit, gh, pool


def ledger_lines(tmp_path):
    path = FieldNoteLedger(tmp_path).path
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


def run(coro):
    return asyncio.run(coro)


def test_a_code_pr_gets_one_run_and_a_second_poll_submits_nothing(tmp_path):
    p, submit, gh, _ = poller(tmp_path, [item(3105)])
    r1 = run(p.poll_once())
    assert r1.submitted == ["pr-field-note-3105"] and r1.code_records == [] and r1.error is None
    [req] = submit.requests
    assert req.workflow == "pr.field_note" and req.brief.pr_number == 3105
    assert req.brief.title == "feat: thing 3105" and req.brief.timeout_sec == 420.0
    assert req.admission.preferred_lane == "agent" and req.admission.priority == "background"
    assert req.admission.deadline_at == NOW + timedelta(hours=20)
    assert gh.log == [f"search:{NOT_BEFORE.isoformat()}", "files:3105"]   # the later of lookback, not_before
    r2 = run(p.poll_once())
    assert r2.submitted == [] and len(submit.requests) == 1
    assert gh.log[-1].startswith("search:")   # no second classify of a PR that has a run
    assert ledger_lines(tmp_path) == []       # a note PR gets no code record


def test_tiny_docs_and_deps_only_prs_get_code_records_and_never_a_run(tmp_path):
    p, submit, gh, _ = poller(tmp_path, [item(3106), item(3107)], files={3106: TINY_DOCS, 3107: DEPS})
    r = run(p.poll_once())
    assert r.submitted == [] and sorted(r.code_records) == [3106, 3107]
    lines = ledger_lines(tmp_path)
    assert sorted(ln["pr_number"] for ln in lines) == [3106, 3107]
    assert all(ln["note"] is None and ln["body_map"] == [] for ln in lines)
    by = {ln["pr_number"]: ln for ln in lines}
    assert by[3106]["flags"]["tiny_docs"] and by[3107]["flags"]["deps_only"]
    assert by[3106]["merge_sha"] == f"{3106:040d}"
    # recorded now: the next poll neither classifies nor submits them
    calls = gh.calls
    r2 = run(p.poll_once())
    assert r2.recorded == 2 and r2.code_records == [] and gh.calls == calls + 1   # the search only


def test_nothing_merged_before_not_before_is_touched(tmp_path):
    p, submit, gh, _ = poller(tmp_path, [item(3001, hours_ago=40), item(3105, hours_ago=1)])
    r = run(p.poll_once())
    # NOW - 72 h is before NOT_BEFORE, so the search floor is NOT_BEFORE and #3001 is filtered out
    assert gh.log[0] == f"search:{NOT_BEFORE.isoformat()}"
    assert r.merged_seen == 1 and r.submitted == ["pr-field-note-3105"]


def test_runs_go_one_at_a_time(tmp_path):
    p, submit, gh, pool = poller(tmp_path, [item(3105, hours_ago=2), item(3106, hours_ago=1)])
    r = run(p.poll_once())
    assert r.submitted == ["pr-field-note-3105"] and r.waiting_turn == 1   # oldest merge first
    pool.rows[0]["terminal"] = "completed"
    r2 = run(p.poll_once())
    assert r2.submitted == ["pr-field-note-3106"]
    assert gh.log.count("files:3106") == 1   # classified once, remembered while waiting its turn


def test_a_pr_already_in_the_ledger_is_skipped(tmp_path):
    p0, _, _, _ = poller(tmp_path, [item(3106)], files={3106: TINY_DOCS})
    run(p0.poll_once())   # writes the code record
    p, submit, gh, _ = poller(tmp_path, [item(3106)], files={3106: TINY_DOCS})
    r = run(p.poll_once())
    assert r.recorded == 1 and submit.requests == [] and gh.log == [gh.log[0]]


def test_a_failed_pr_gets_one_later_run_then_stops(tmp_path):
    old = NOW - timedelta(hours=2)
    rows = [{"run_id": "pr-field-note-3105", "terminal": "failed", "created_at": old, "updated_at": old}]
    p, submit, gh, pool = poller(tmp_path, [item(3105, hours_ago=3)], rows=rows)
    r = run(p.poll_once())
    assert r.submitted == ["pr-field-note-3105-a2"] and "files:3105" not in gh.log
    pool.rows[-1].update(terminal="failed", updated_at=old)
    assert MAX_RUNS_PER_PR == 2
    assert run(p.poll_once()).submitted == []


@pytest.mark.parametrize("attempts, expected", [
    ([], "pr-field-note-7"),
    ([("pr-field-note-7", None, 5)], None),
    ([("pr-field-note-7", "completed", 500)], None),
    ([("pr-field-note-7", "failed", 30)], None),
    ([("pr-field-note-7", "failed", 90)], "pr-field-note-7-a2"),
    ([("pr-field-note-7", "failed", 300), ("pr-field-note-7-a2", "failed", 90)], None),
])
def test_next_run_id(attempts, expected):
    rows = [{"run_id": r, "terminal": t, "updated_at": NOW - timedelta(minutes=m)} for r, t, m in attempts]
    assert next_run_id(7, rows, now=NOW) == expected


def test_run_ids_round_trip():
    assert base_run_id(3105) == "pr-field-note-3105"
    assert pr_number_of_run("pr-field-note-3105") == 3105 == pr_number_of_run("pr-field-note-3105-a2")
    assert pr_number_of_run("harness-echo-smoke-x") is None and pr_number_of_run("pr-field-note-") is None


def test_github_trouble_ends_the_tick_without_submitting(tmp_path):
    p, submit, gh, _ = poller(tmp_path, [item(3105)], fail="rate_limited: GET /search")
    r = run(p.poll_once())
    assert r.error.startswith("github: rate_limited") and submit.requests == []


def test_an_unmounted_ledger_never_reads_as_no_records(tmp_path):
    p, submit, gh, _ = poller(tmp_path / "absent", [item(3105)])
    r = run(p.poll_once())
    assert r.error and r.error.startswith("ledger:") and submit.requests == []


def test_the_detail_call_budget_bounds_a_burst(tmp_path):
    merged = [item(4000 + i, hours_ago=1 + i / 100) for i in range(14)]
    p, submit, gh, _ = poller(tmp_path, merged, files={4000 + i: TINY_DOCS for i in range(14)})
    r = run(p.poll_once())
    # 10 detail calls: 3 code records (3 calls each) and one classify; the rest wait for the next tick
    assert len(r.code_records) == 3 and r.github_calls == 11 and r.skipped_budget >= 10


def test_gap_warning_when_the_last_run_is_older_than_the_lookback(tmp_path):
    long_ago = NOW - timedelta(days=5)
    rows = [{"run_id": "pr-field-note-1", "terminal": "completed", "created_at": long_ago, "updated_at": long_ago}]
    p, *_ = poller(tmp_path, [], rows=rows, field_note_not_before=datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert "POST /pr-field-notes/poll" in run(p.poll_once()).gap_warning
    p2, *_ = poller(tmp_path, [], rows=rows)   # not_before is inside the lookback: no gap to report
    assert run(p2.poll_once()).gap_warning is None


def test_manual_since_replaces_the_floor(tmp_path):
    p, submit, gh, _ = poller(tmp_path, [item(3105, hours_ago=100)])
    since = NOW - timedelta(hours=120)
    r = run(p.poll_once(since=since))
    assert gh.log[0] == f"search:{since.isoformat()}" and r.submitted == ["pr-field-note-3105"]


# --- the public API client ------------------------------------------------------------------------

class Resp:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_github_client_pages_the_search_and_keeps_only_merged():
    pages = {1: {"total_count": 101, "items": [{"number": i, "title": "t", "pull_request": {"merged_at": "2026-11-05T00:00:00Z"}}
                                               for i in range(100)]},
             2: {"total_count": 101, "items": [{"number": 100, "pull_request": {"merged_at": None}}]}}
    seen = []

    def opener(req, timeout):
        seen.append(req.full_url)
        assert "Authorization" not in req.headers   # public API, no token
        return Resp(pages[int(req.full_url.rsplit("page=", 1)[1])])

    gh = GitHubPublic("o/r", opener=opener)
    out = gh.merged_since(NOW)
    assert len(out) == 100 and gh.calls == 2 and "is%3Amerged" in seen[0]


def test_github_client_maps_a_rate_limit_to_unavailable():
    def opener(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 403, "rate", {"X-RateLimit-Remaining": "0"}, None)

    with pytest.raises(GitHubUnavailable, match="rate_limited"):
        GitHubPublic("o/r", opener=opener).files(1)


def test_gh_shape_classifies_like_the_dig_cli():
    assert is_code_record_pr(gh_shape(item(1), TINY_DOCS))
    assert is_code_record_pr(gh_shape(item(1), DEPS))
    assert not is_code_record_pr(gh_shape(item(1), CODE_FILES))
    big_docs = [{"filename": "docs/a.md", "additions": 60, "deletions": 0}]
    assert not is_code_record_pr(gh_shape(item(1), big_docs))      # docs-only but not tiny: a note
    cut = [{"filename": f"docs/{i}.md", "additions": 0, "deletions": 0} for i in range(100)]
    assert gh_shape(item(1), cut)["changedFiles"] == 101           # a full page may be cut: never "tiny"


def test_run_request_is_a_valid_durable_request():
    req = run_request(item(3105), "pr-field-note-3105", now=NOW, timeout_sec=420)
    again = run_request(item(3105), "pr-field-note-3105", now=NOW + timedelta(minutes=1), timeout_sec=420)
    assert DurableRunRequestV1.model_validate(req.model_dump(mode="json")).workflow == "pr.field_note"
    assert req.correlation_id == again.correlation_id   # deterministic per run id


# --- the ledger module ----------------------------------------------------------------------------

def test_ledger_refuses_a_code_record_for_a_note_pr_and_reports_receipts(tmp_path):
    ledger = FieldNoteLedger(tmp_path)
    with pytest.raises(ValueError, match="needs a note"):   # field_notes.FieldNoteError
        ledger.write_code_record(pr=gh_shape(item(9), CODE_FILES, {"merge_commit_sha": "a" * 40}))
    nid = ledger.write_code_record(pr=gh_shape(item(9), TINY_DOCS, {"merge_commit_sha": "a" * 40}))
    assert nid and nid.startswith("fn:9:")
    assert ledger.write_code_record(pr=gh_shape(item(9), TINY_DOCS)) is None   # one live record per PR
    assert ledger.live_records()[9].has_note is False
    assert ledger.receipt_ok(9, nid) is False   # a code record is not a note: no receipt can name it
    with pytest.raises(FileNotFoundError):
        FieldNoteLedger(tmp_path / "absent").receipt_ok(9, nid)


def test_flag_ships_on_and_every_key_is_in_settings_env_example_and_compose():
    """Env parity (AGENTS.md section 7) and ship-flags-ON, as a gate rather than a reminder."""
    import re
    from pathlib import Path

    from app.settings import Settings

    root = Path(__file__).resolve().parents[1]
    example = (root / ".env_example").read_text()
    compose = (root / "docker-compose.yml").read_text()
    keys = ["PR_FIELD_NOTES_ENABLED", "PR_FIELD_NOTES_POLL_SEC", "PR_FIELD_NOTES_LOOKBACK_H",
            "PR_FIELD_NOTES_GITHUB_REPO", "PR_FIELD_NOTES_NOT_BEFORE", "PR_FIELD_NOTES_LEDGER_DIR",
            "PR_FIELD_NOTES_TURN_TIMEOUT_SEC"]
    aliases = {f.alias for f in Settings.model_fields.values()}
    for key in keys:
        assert re.search(rf"^{key}=", example, re.M), key
        assert f"{key}=${{{key}:-" in compose, key
        assert key in aliases, key
    assert "PR_FIELD_NOTES_SELF_HOST=" in example and "${PR_FIELD_NOTES_SELF_HOST:-" in compose
    assert re.search(r"^PR_FIELD_NOTES_ENABLED=true$", example, re.M)
    assert "PR_FIELD_NOTES_ENABLED=${PR_FIELD_NOTES_ENABLED:-true}" in compose
    assert Settings.model_fields["field_note_poll_enabled"].default is True

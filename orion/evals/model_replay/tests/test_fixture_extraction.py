from __future__ import annotations

import re

import pytest

from orion.evals.model_replay import extract
from orion.evals.model_replay.fixture import FIXTURE_PATH, HARNESS_TOOLS, READING_TOOLS, load_tasks


def _brief_row(run_id: str, line: str = "investigate", **brief):
    return {"run_id": run_id, "terminal": "completed",
            "request": {"brief": {"line": line, "prompt": f"brief prompt for {run_id}", **brief}}}


def _reading_row(run_id: str):
    prompt = (f"Fetch the url below with WebFetch\nseed_id=digest_item:{run_id}:x kind=digest_item run_id=r-{run_id}\n"
              f"url=https://example.org/{run_id}\ntitle=T {run_id}\nsection=world\nRequired JSON shape:")
    return {"run_id": run_id, "terminal": "completed", "request_json": {"brief": {"prompt": prompt}}}


@pytest.fixture()
def rows():
    questions = [["what_are_you", "What are you?"], ["last_day_unasked", "What did you do?"],
                 ["cannot_do_now", "What can't you do?"], ["who_matters", "Who matters?"]]
    cur = {r: _brief_row(r) for r, _, _ in extract.CURIOSITY_RUNS}
    ss = {r: _brief_row(r, line="self_sense_eval", questions=questions,
                        lived_answers=[{"question_id": "lived.what_are_you", "content": f"lived {r}"}])
          for r, _ in extract.SELF_SENSE_SNAPSHOTS}
    rd = {r: _reading_row(r) for r, _, _, _ in extract.READING_RUNS}
    return cur, ss, rd


def test_build_tasks_shape(rows):
    tasks = extract.build_tasks(*rows)
    assert extract.task_counts(tasks) == {"curiosity": 10, "self_sense": 8, "reading": 6, "stance_react": 6}
    assert len({t.task_id for t in tasks}) == 30
    reading = [t for t in tasks if t.kind == "reading"]
    assert sum(t.fetch.mode == "fail" for t in reading) == 2
    assert all(t.fetch.fail_error for t in reading if t.fetch.mode == "fail")
    assert all(t.reading_only and t.tools == list(READING_TOOLS) and t.reading_seed for t in reading)
    assert reading[0].reading_seed.url.startswith("https://example.org/")
    cur = [t for t in tasks if t.kind == "curiosity"]
    assert all(t.write_claim_check and t.tools == list(HARNESS_TOOLS) for t in cur)
    assert "curiosity-d4db8c2bacb4" in {t.task_id for t in cur}
    stance = [t for t in tasks if t.kind == "stance_react"]
    assert all(t.tools == [] and t.expect == "thought_json" for t in stance)


def test_stance_prompt_uses_production_template_and_lived_answers(rows):
    tasks = extract.build_tasks(*rows)
    ss = next(t for t in tasks if t.task_id == "self_sense-s1-what_are_you")
    assert "internal stance reactor" in ss.stance_prompt
    assert '"What are you?"' in ss.stance_prompt          # user_message | tojson
    assert "In my own words (lived / what_are_you): lived" in ss.stance_prompt
    orion_lines, _ = extract.fallback_identity()
    assert orion_lines[0] in ss.stance_prompt


def test_missing_source_row_raises(rows):
    cur, ss, rd = rows
    cur.pop("d4db8c2bacb4")
    with pytest.raises(KeyError):
        extract.build_tasks(cur, ss, rd)


def test_lived_answer_lines_cap():
    lines = extract.lived_answer_lines([{"question_id": "lived.q", "content": "x" * 2000}])
    assert len(lines) == 1 and len(lines[0]) <= 800 and lines[0].endswith("…")


def test_committed_fixture_loads_and_matches_design():
    tasks = load_tasks(FIXTURE_PATH)
    assert extract.task_counts(tasks) == {"curiosity": 10, "self_sense": 8, "reading": 6, "stance_react": 6}
    d4db = next(t for t in tasks if t.task_id == "curiosity-d4db8c2bacb4")
    assert 'run_id "d4db8c2bacb4"' in d4db.user_message
    assert {t.snapshot for t in tasks if t.kind == "self_sense"} == {"s1", "s2"}


def test_committed_fixture_has_no_credentials():
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    for pattern in (r"ghp_[A-Za-z0-9]{20}", r"sk-[A-Za-z0-9]{20}", r"PGPASSWORD=\S", r"://[^\s:/@$]+:[^\s@$]+@",
                    r"github_pat_[A-Za-z0-9_]{20}"):
        assert not re.search(pattern, text), pattern

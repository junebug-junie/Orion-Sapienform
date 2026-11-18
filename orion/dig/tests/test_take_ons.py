"""`dig take-ons`: sub-agent dispatches per take-on, read from the carry turn's own transcript (#3639)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from orion.dig import __main__ as dig
from orion.dig import take_ons
from orion.harness.session_resume import find_transcript, session_id_for

CID = "d8db9d7b1bba4cf7"
RUN = f"intention-{CID}"


def _assistant(*blocks: dict) -> str:
    return json.dumps({"type": "assistant", "message": {"role": "assistant", "content": list(blocks)}})


def _use(tid: str, name: str, **inp) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _write(home: Path, session_id: str, lines: list[str]) -> Path:
    path = home / "projects" / "-mnt-orion-fcc-repo" / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _runner(rows: list[dict], seen: list[str]):
    class Runner:
        def run(self, sql: str, flags: list[str]) -> tuple[str, int]:
            seen.append(sql)
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=["run_id", "created_at", "terminal"])
            w.writeheader()
            w.writerows(rows)
            return buf.getvalue(), 0

    return Runner()


def test_resume_key_matches_hub_and_durable_runs() -> None:
    # Hub: resume_key=f"{request.run_id}:{request.source_tag}"; durable-runs _intention_step: source_tag
    # f"{INTENTION_CARRY_TAG}:{brief.concern_id}". The live 11-13 take-on's gen-0 transcript has this id.
    assert take_ons.resume_key(RUN, CID) == f"{RUN}:intention_carry:{CID}"
    assert session_id_for(take_ons.resume_key(RUN, CID)) == "9d0eeec6-51c2-5225-a537-9efe37a234a3"


def test_count_tools_counts_agent_and_task_by_type_and_dedupes_ids() -> None:
    lines = [
        _assistant(_use("a", "Bash", command="ls")),
        _assistant(_use("b", "Agent", subagent_type="Explore", prompt="find x")),
        _assistant(_use("c", "Task", prompt="old name, no type")),
        _assistant(_use("b", "Agent", subagent_type="Explore", prompt="find x")),  # the same block logged twice
        json.dumps({"type": "assistant", "isSidechain": True,  # a sub-agent's own call (older CLIs): not counted
                    "message": {"content": [_use("s", "Agent", subagent_type="Explore")]}}),
        json.dumps({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "a"}]}}),
        "not json with \"tool_use\"",
    ]
    tools, subs = take_ons.count_tools(lines, set())
    assert tools == 3
    assert subs == {"Explore": 1, "general-purpose": 1}


def test_read_take_ons_sums_generations_and_marks_missing_transcript_unread(tmp_path: Path) -> None:
    key = take_ons.resume_key(RUN, CID)
    _write(tmp_path, session_id_for(key, 0), [_assistant(_use("a", "Bash", command="ls"))])
    # generation 1 missing, generation 2 present: gaps are probed past
    _write(tmp_path, session_id_for(key, 2), [_assistant(_use("b", "Agent", subagent_type="general-purpose"))])
    rows = [{"run_id": RUN, "created_at": "2026-11-13 19:47:08+00", "terminal": "completed"},
            {"run_id": "intention-0123456789abcdef", "created_at": "2026-11-14 10:00:00+00", "terminal": "running"}]
    found = take_ons.read_take_ons(rows, lambda sid: find_transcript(tmp_path, sid))
    assert [len(t.transcripts) for t in found] == [2, 0]
    assert found[0].tool_calls == 2 and found[0].dispatches == 1
    assert not found[1].read and found[1].dispatches == 0
    text = "\n".join(take_ons.render(found, 7))
    assert "transcripts read for 1; 1 started a sub-agent (1 sub-agent call(s))" in text
    assert "general-purpose 1" in text
    assert "intention-0123456789abcdef running: UNREAD" in text  # never shown as 0


def test_render_says_none_when_no_take_ons_and_none_when_no_subagents(tmp_path: Path) -> None:
    assert take_ons.render([], 7) == ["no take-ons in the last 7 day(s)"]
    _write(tmp_path, session_id_for(take_ons.resume_key(RUN, CID)), [_assistant(_use("a", "Bash", command="ls"))])
    found = take_ons.read_take_ons([{"run_id": RUN, "created_at": "2026-11-13", "terminal": "completed"}],
                                   lambda sid: find_transcript(tmp_path, sid))
    assert "1 attempt(s), 1 tool call(s), sub-agents: none" in take_ons.render(found, 7)[1]


def test_build_sql_clamps_days_and_reads_only_carry_runs() -> None:
    assert "interval '90 days'" in take_ons.build_sql(10_000)
    assert "interval '1 days'" in take_ons.build_sql(0)
    assert "workflow = 'intention.carry'" in take_ons.build_sql(7)


def test_cmd_take_ons_prints_counts_and_writes_full_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path / "dig"))
    home = tmp_path / "claude"
    _write(home, session_id_for(take_ons.resume_key(RUN, CID)),
           [_assistant(_use("a", "Agent", subagent_type="Explore", prompt="p"))])
    seen: list[str] = []
    rows = [{"run_id": RUN, "created_at": "2026-11-13 19:47:08+00", "terminal": "completed"}]
    text = dig.cmd_take_ons("30", dig.Out("t"), runner=_runner(rows, seen),
                            find=lambda sid: find_transcript(home, sid))
    assert "interval '30 days'" in seen[0]
    assert "sub-agents: Explore 1" in text and "full list:" in text
    with pytest.raises(dig.DigError):
        dig.cmd_take_ons("thirty", dig.Out("t"), runner=_runner(rows, []))
    with pytest.raises(dig.DigError):
        dig.cmd_take_ons("0", dig.Out("t"), runner=_runner(rows, []))


def test_dig_help_lists_take_ons() -> None:
    assert "take-ons [<days>]" in dig.HELP


def test_probing_follows_the_registry_generation_and_says_when_capped(tmp_path: Path) -> None:
    key = take_ons.resume_key(RUN, CID)
    _write(tmp_path, session_id_for(key, 9), [_assistant(_use("a", "Bash", command="ls"))])
    find = lambda sid: find_transcript(tmp_path, sid)  # noqa: E731
    # no registry: four misses in a row stop the probe before gen 9
    assert take_ons.transcripts_for(RUN, CID, find) == ([], False)
    # the registry says gen 9 is the last: it is read
    paths, capped = take_ons.transcripts_for(RUN, CID, find, last_generation=9)
    assert len(paths) == 1 and not capped
    # every generation present up to the cap: the cap is said
    paths, capped = take_ons.transcripts_for(RUN, CID, lambda sid: tmp_path)
    assert capped and len(paths) == take_ons.MAX_GENERATIONS


def test_sql_prints_utc_minutes_and_non_ascii_days_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI'" in take_ons.build_sql(7)
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    with pytest.raises(dig.DigError):
        dig.cmd_take_ons("\u00b2", dig.Out("t"), runner=_runner([], []))

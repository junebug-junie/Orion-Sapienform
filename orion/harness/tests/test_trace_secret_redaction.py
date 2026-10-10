"""Credentials never reach a published trace step or the final reply.

Live 2026-10-10: an FCC turn ran ``echo $ORION_CURIOSITY_PG_DSN`` and the
read-only DB password rendered in Hub's turn trace.
"""
from __future__ import annotations

from orion.core import redact
from orion.core.redact import redact_secrets, redact_secrets_deep, secret_env_values
from orion.harness.fcc_motor import build_step_frame, extract_final_from_stream_event

DSN = "postgresql://orion_readonly:4b4bf2ed563fcfda@orion-athena-sql-db:5432/conjourney"


def _tool_result_event(text: str) -> dict:
    return {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "tool_use_id": "x", "content": text}]},
    }


def test_dsn_password_stripped_from_step_but_host_kept():
    step = build_step_frame(_tool_result_event(f"DSN: '{DSN}'"))
    body = step["raw"]["message"]["content"][0]["content"]
    assert "4b4bf2ed563fcfda" not in body
    assert "postgresql://[REDACTED]@orion-athena-sql-db:5432/conjourney" in body


def test_step_redaction_does_not_mutate_the_event_the_motor_reads():
    event = _tool_result_event(DSN)
    build_step_frame(event)
    assert event["message"]["content"][0]["content"] == DSN


def test_bare_secret_env_value_is_stripped(monkeypatch):
    monkeypatch.setattr(redact, "_process_secret_values", lambda: ("s3cretPassw0rd",))
    step = build_step_frame(_tool_result_event("password is s3cretPassw0rd ok"))
    assert "s3cretPassw0rd" not in step["raw"]["message"]["content"][0]["content"]


def test_tool_use_input_is_stripped_too():
    event = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": f'psql "{DSN}" -c "select 1"'}},
    ]}}
    command = build_step_frame(event)["raw"]["message"]["content"][0]["input"]["command"]
    assert "4b4bf2ed563fcfda" not in command and "orion-athena-sql-db" in command


def test_final_reply_is_stripped():
    text, _, _ = extract_final_from_stream_event({"type": "result", "result": f"the dsn is {DSN}"}, accumulated="")
    assert "4b4bf2ed563fcfda" not in text


def test_shell_variable_userinfo_is_left_readable():
    cmd = 'redis-cli -u "redis://$ORION_CURIOSITY_GRAPH_USER:$ORION_CURIOSITY_GRAPH_PASSWORD@host:6379"'
    assert redact_secrets(cmd, ()) == cmd


def test_secret_env_values_pick_credential_names_only():
    env = {
        "ORION_CURIOSITY_GRAPH_PASSWORD": "longpassword1",
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_abcdefgh",
        "HARNESS_FCC_MAX_CONTEXT_TOKENS": "65536000",
        "ORION_CURIOSITY_GRAPH_HOST": "orion-athena-falkordb",
        "SHORT_SECRET": "abc",
    }
    assert set(secret_env_values(env)) == {"longpassword1", "ghp_abcdefgh"}


def test_deep_redaction_leaves_non_strings_alone():
    assert redact_secrets_deep({"n": 3, "b": None, "l": [1, "x"]}, ()) == {"n": 3, "b": None, "l": [1, "x"]}

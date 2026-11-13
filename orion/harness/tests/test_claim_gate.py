"""A reply may only name a correction_id that a record_correction result in the same turn returned."""

from __future__ import annotations

from orion.harness.claim_gate import UNVERIFIED_MARK, gate_correction_claims, gate_focus_claims
from orion.harness.cut_short import TurnFindings

REAL = "8f0c1c3e-5b7a-4f5e-9a77-1f3d2a9b6c10"

# The reply of live turn 5fdf2faf (2026-10-15), verbatim in the part that matters.
LIVE_REPLY = (
    "I recorded the correction for ¶9:\n"
    '- record_correction(section="¶9", priors=["self:ring_touch_is_prior_claim_text_20261013"])\n'
    "- correction_id: corr_9_218_reproduce\n"
)


def test_the_live_incident_reply_is_gated_when_no_tool_returned_an_id():
    gate = gate_correction_claims(LIVE_REPLY, [])
    assert gate.unverified_ids == ("corr_9_218_reproduce",)
    assert f"correction_id: {UNVERIFIED_MARK}" in gate.text
    assert "corr_9_218_reproduce" not in gate.text.split("[Unverified")[0]
    assert "Nothing was recorded." in gate.text


def test_an_id_a_tool_really_returned_passes_untouched():
    reply = f"Recorded. correction_id: {REAL}"
    gate = gate_correction_claims(reply, [f'{{"correction_id": "{REAL}", "status": "recorded"}}'])
    assert gate.changed is False and gate.text == reply


def test_json_and_quoted_forms_of_the_claim_are_checked_too():
    for form in (f'"correction_id": "{REAL}"', f"correction_id = `{REAL}`", f"correction id: {REAL}"):
        assert gate_correction_claims(form, []).unverified_ids == (REAL,), form
        assert gate_correction_claims(form, [REAL]).changed is False, form


def test_a_real_id_for_a_different_correction_does_not_launder_a_fake_one():
    gate = gate_correction_claims(
        f"correction_id: {REAL} and also correction_id: made_up_id_77", [f"correction_id {REAL}"]
    )
    assert gate.unverified_ids == ("made_up_id_77",)
    assert REAL in gate.text


def test_an_invented_prefix_of_a_real_id_does_not_pass():
    # Whole-token match: the claimed id must be a token of a result, not a substring of one.
    assert gate_correction_claims("correction_id: 123456789", ["id 1234567890"]).unverified_ids == ("123456789",)


def test_ordinary_prose_about_correction_ids_is_left_alone():
    for prose in (
        "Each correction id should be recorded carefully.",
        "A correction id mismatch would be visible in the table.",
        "The correction_id returned by the tool is a UUID.",
        "Check the correction id before you retry.",
    ):
        gate = gate_correction_claims(prose, [])
        assert gate.changed is False and gate.text == prose, prose


def test_text_without_a_claim_is_untouched():
    assert gate_correction_claims("", []).text == ""
    assert gate_correction_claims(None, []).text == ""


def _call(findings, tool_id, name):
    findings.observe({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": tool_id, "name": name, "input": {}}]}})


def _result(findings, tool_id, body):
    findings.observe({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": body}]}})


def test_turn_findings_keeps_only_record_correction_results():
    findings = TurnFindings()
    _call(findings, "t1", "mcp__orion-introspect__record_correction")
    _result(findings, "t1", f'{{"correction_id": "{REAL}"}}')
    _call(findings, "t2", "Bash")
    _result(findings, "t2", "correction_id: from_a_shell_echo_99")
    assert findings.correction_results == [f'{{"correction_id": "{REAL}"}}']
    # an id echoed by some other tool does not verify a claim
    assert gate_correction_claims("correction_id: from_a_shell_echo_99", findings.correction_results).changed


def test_a_long_turn_cannot_push_the_real_result_out():
    findings = TurnFindings()
    for i in range(1000):
        _call(findings, f"b{i}", "Bash")
        _result(findings, f"b{i}", "output")
    _call(findings, "rc", "mcp__orion-introspect__record_correction")
    _result(findings, "rc", f'{{"correction_id": "{REAL}"}}')
    assert gate_correction_claims(f"correction_id: {REAL}", findings.correction_results).changed is False


# --- focus_id (queue_curiosity_focus) -----------------------------------------------------------


def test_a_focus_id_no_tool_returned_is_gated_and_says_nothing_was_queued():
    reply = "Done -- I queued it for my next run. focus_id: focus_bus_fanout_01"
    gate = gate_focus_claims(reply, [])
    assert gate.unverified_ids == ("focus_bus_fanout_01",)
    assert f"focus_id: {UNVERIFIED_MARK}" in gate.text
    # An id from an earlier turn may be real: the notice says unknown, never "nothing was queued".
    assert "is unknown" in gate.text and "Nothing was queued" not in gate.text
    assert "queue_curiosity_focus" in gate.text


def test_a_focus_id_the_tool_returned_passes():
    reply = f'Queued. "focus_id": "{REAL}"'
    assert gate_focus_claims(reply, [f'{{"focus_id": "{REAL}"}}']).changed is False


def test_a_correction_result_does_not_verify_a_focus_claim():
    findings = TurnFindings()
    _call(findings, "rc", "mcp__orion-introspect__record_correction")
    _result(findings, "rc", f'{{"correction_id": "{REAL}"}}')
    assert findings.focus_results == []
    assert gate_focus_claims(f"focus_id: {REAL}", findings.focus_results).changed


def test_turn_findings_keeps_queue_curiosity_focus_results():
    findings = TurnFindings()
    _call(findings, "q1", "mcp__orion-introspect__queue_curiosity_focus")
    _result(findings, "q1", f'{{"focus_id": "{REAL}"}}')
    assert gate_focus_claims(f"focus_id: {REAL}", findings.focus_results).changed is False


def test_prose_about_a_focus_id_is_left_alone():
    for prose in ("The focus id comes back from the tool.", "Check the focus_id before saying it ran."):
        assert gate_focus_claims(prose, []).changed is False


def test_the_governor_gates_both_claims_on_every_reply_path():
    """Static: the one function every reply passes through applies both gates."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[3] / "services" / "orion-harness-governor" / "app"
           / "bus_listener.py").read_text()
    body = src.split("def _gate_claims", 1)[1].split("\ndef ", 1)[0]
    assert "gate_correction_claims(" in body and "gate_focus_claims(" in body


def test_a_focus_id_read_back_with_the_curiosity_tool_is_verified():
    """The brief tells Orion to call curiosity(kind=focus) to say whether a focus ran; repeating the
    id it read must not be marked unverified (review finding, 2026-10-16)."""
    findings = TurnFindings()
    _call(findings, "c1", "mcp__orion-introspect__curiosity")
    _result(findings, "c1", f'{{"items": [{{"id": "{REAL}", "kind": "curiosity_focus"}}]}}')
    assert gate_focus_claims(f"focus_id: {REAL} ran and was explored", findings.focus_results).changed is False


def test_a_tool_whose_arguments_merely_mention_the_name_does_not_count():
    findings = TurnFindings()
    findings.observe({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"cmd": "echo queue_curiosity_focus record_correction"}}]}})
    _result(findings, "b1", f'focus_id: {REAL} correction_id: {REAL}')
    assert findings.focus_results == [] and findings.correction_results == []


def test_a_long_curiosity_result_keeps_an_id_near_its_end():
    findings = TurnFindings()
    _call(findings, "c2", "mcp__orion-introspect__curiosity")
    _result(findings, "c2", "x" * 11000 + f" {REAL}")
    assert gate_focus_claims(f"focus_id: {REAL}", findings.focus_results).changed is False

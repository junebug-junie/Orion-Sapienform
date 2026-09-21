"""`is_transient_failure` against the literal live reasons from Postgres.

These strings are copied verbatim (truncated where noted) from
`world_pulse_read_seed.last_error` / `.stage2_error` on 2026-09-19: 54 Stage 1
rows and 12 Stage 2 rows dead, almost all one GPU-capacity refusal. The
predicate exists to tell that apart from a bad seed (schema/JSON failure) --
this test is the contract for that split, not just the prefix arithmetic.
"""
from orion.world_pulse_read.retry import TRANSIENT_FAILURE_PREFIXES, is_transient_failure


def test_none_and_empty_are_not_transient():
    assert is_transient_failure(None) is False
    assert is_transient_failure("") is False


def test_live_transient_reasons_are_retried():
    live_transient = [
        "turn_deferred:stance_react_failed: stance_react exec result missing thought payload",
        "turn_error:fcc turn timed out after 2400.0s",
        "stage2_turn_timeout",
        "stage1_turn_timeout",
        "turn_exception:governor unreachable",
        "empty_generation",
        "blank_final_response",
        "looks_like_error_text",
        "no_final_frame",
        "non_final_frame:tool_call",
        "bus_unavailable",
    ]
    for reason in live_transient:
        assert is_transient_failure(reason), reason


def test_live_non_transient_reasons_stay_terminal():
    live_terminal = [
        # Live 2026-09-14/15 validation errors (Bug 1, now fixed at the schema
        # level, but old rows -- and any future genuine schema drift -- must
        # still not retry).
        "6 validation errors for WorldPulseReadStage2ResultV1\ncandidate_priors\n"
        "  Extra inputs are not permitted [type=extra_forbidden, ...]",
        "1 validation error for WorldPulseReadStage2ResultV1\npriors_tested\n"
        "  Extra inputs are not permitted [type=extra_forbidden, ...]",
        # Live JSON-parse failures.
        "Could not parse JSON object from LLM text: 'I fetched the Rubin press "
        "release to supplement your request, which only inc",
        "Could not parse JSON object from LLM text: 'Stage 2 receipt of your "
        "Stage 1 handoff for seed `reading:f3283057`. I ran a live "
        "re-verification this turn to close",
        # Seed-shaped failures, not turn-shaped.
        "handoff_invalid:1 validation error for WorldPulseReadHandoffV1",
        "stage2_result_not_object",
        "bad_url",
        "round_trip_cap",
        # Reclaim markers are informational, never a fail_reason input in
        # practice, but must not accidentally match if one ever is.
        "interrupted:process_restart",
        "interrupted:stale_timeout",
    ]
    for reason in live_terminal:
        assert not is_transient_failure(reason), reason


def test_prefix_table_has_no_accidental_substring_collisions():
    """Every prefix must match itself and nothing that merely contains it
    mid-string (a plain `str.startswith` scan, not `in`)."""
    for prefix in TRANSIENT_FAILURE_PREFIXES:
        assert is_transient_failure(prefix)
        assert not is_transient_failure(f"not_{prefix}")

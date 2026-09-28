"""The urgent investigation prompt: an assignment, not Orion's self-directed invitation.

It must carry Juniper's question verbatim and the evidence bundle, ask for one
`:IncidentReport` only when a graph is there to write it to, and carry none of
the self-directed kickoff's material (priors, crystallizations, peer notes,
dreams, continuation, hops).
"""

from __future__ import annotations

from datetime import datetime, timezone

from orion.curiosity import kickoff_prompt
from orion.curiosity.urgent_prompt import EVIDENCE_CHAR_CAP, TRUNCATION_MARKER, build_urgent_prompt
from orion.curiosity.worldview import TurnOutcome, WorldviewSnapshot
from orion.schemas.curiosity_urgent import CuriosityUrgentSeedV1

RUN_ID = "a1b2c3d4e5f6"
INCIDENT_ID = "0123456789abcdef0123456789abcdef"
QUESTION = "Why did circe's gpu2 hit 91C at 3am -- is the 'fan' actually spinning?"


def _seed(**overrides) -> CuriosityUrgentSeedV1:
    fields = dict(
        incident_id=INCIDENT_ID,
        question=QUESTION,
        trigger="manual",
        subject="circe/gpu2",
        evidence={"gpus": {"circe/gpu2": {"temp_c": 91, "fan_pct": 0}}, "collected_at": "2026-09-28T09:00:00Z"},
        requested_at=datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc),
    )
    fields.update(overrides)
    return CuriosityUrgentSeedV1(**fields)


def test_question_is_verbatim_and_evidence_is_present():
    prompt = build_urgent_prompt(_seed(), run_id=RUN_ID)
    assert QUESTION in prompt
    assert "fan_pct" in prompt
    assert "circe/gpu2" in prompt


def test_manual_and_rule_headers_differ():
    manual = build_urgent_prompt(_seed(), run_id=RUN_ID)
    rule = build_urgent_prompt(_seed(trigger="heat"), run_id=RUN_ID)
    assert "Juniper asked you to investigate this now" in manual
    assert "A hardware rule fired" not in manual
    assert "A hardware rule fired" in rule
    assert "Juniper asked you to investigate this now" not in rule


def test_incident_report_template_only_when_graph_enabled():
    enabled = build_urgent_prompt(_seed(), run_id=RUN_ID, own_graph="orion_worldview")
    assert ":IncidentReport" in enabled
    assert f'run_id: "{RUN_ID}"' in enabled
    assert f'incident_id: "{INCIDENT_ID}"' in enabled
    assert "GRAPH.QUERY orion_worldview" in enabled
    assert "<RUN_ID>" not in enabled

    disabled = build_urgent_prompt(_seed(), run_id=RUN_ID, graph_enabled=False)
    assert ":IncidentReport" not in disabled
    assert RUN_ID not in disabled
    assert "redis-cli" not in disabled
    assert "prose answer is the report" in disabled


def test_pool_url_and_hardware_tables_named():
    prompt = build_urgent_prompt(_seed(), run_id=RUN_ID, pool_url="http://pool.test:9")
    assert "http://pool.test:9/v1/pool" in prompt
    assert "orion_biometrics_summary" in prompt
    assert "home_cooling_sample" in prompt
    default = build_urgent_prompt(_seed(), run_id=RUN_ID)
    assert "http://orion-athena-gpu-pool:8127/v1/pool" in default


def test_reuses_kickoff_access_and_clock_sections():
    prompt = build_urgent_prompt(_seed(), run_id=RUN_ID)
    assert kickoff_prompt._budget_section(writable=False)[0] in prompt
    assert "HOW TO REACH YOUR OWN MATERIAL" in prompt
    # The clock is the non-writable variant: there is no :TurnOutcome here,
    # so it must not promise a continuation note.
    assert "continuation note" not in prompt


def test_sections_in_order():
    prompt = build_urgent_prompt(_seed(), run_id=RUN_ID)
    order = [
        QUESTION,
        "fan_pct",
        "WHAT TO FIND OUT",
        "HOW TO REACH YOUR OWN MATERIAL",
        ":IncidentReport",
        kickoff_prompt._budget_section(writable=False)[0],
        "ANSWER IN PLAIN PROSE",
    ]
    positions = [prompt.index(marker) for marker in order]
    assert positions == sorted(positions)


def test_no_self_directed_kickoff_material():
    prompt = build_urgent_prompt(_seed(), run_id=RUN_ID)
    assert kickoff_prompt._HEADER not in prompt
    assert kickoff_prompt._INSTRUCTION not in prompt

    # Headers produced by each self-directed section for a minimal fixture.
    view = WorldviewSnapshot(
        continuation=TurnOutcome(
            run_id="ffffff", continue_line=True, continue_note="keep going", reach_out=False, reach_out_why=""
        ),
    )
    forbidden = [
        kickoff_prompt._continuation_section(view.continuation)[0],
        kickoff_prompt._priors_section(WorldviewSnapshot(), stale_after=3)[0],
        kickoff_prompt._overlay_section(own_graph="g", atlas_graph="a")[0],
        kickoff_prompt._hops_section(5)[0],
        kickoff_prompt._write_section(own_graph="orion_worldview", run_id=RUN_ID, max_hops=5)[0],
        kickoff_prompt._outcome_section(run_id=RUN_ID)[0],
        kickoff_prompt._role_and_help_section(own_graph="orion_worldview", run_id=RUN_ID)[0],
        kickoff_prompt._review_role_section(run_id=RUN_ID)[0],
    ]
    forbidden += [
        "WHAT YOU ARE STILL UNSURE OF",
        "WHAT YOU HAVE CRYSTALLISED",
        "CONCEPT INDUCTION",
        "THINGS YOUR CAMERAS SAW",
        "RUNS YOU DID",
        "PEER LOOKED",
        "WHILE YOU SLEPT",
        ":TurnOutcome",
        ":Hop ",
        ":Prior {",
    ]
    for phrase in forbidden:
        assert phrase not in prompt, phrase


def test_evidence_over_cap_is_truncated_with_marker():
    # Pretty-printed, many small keys expand well past the cap while the
    # compact form stays under the seed's 32 000-byte limit.
    evidence = {f"k{i:05d}": i for i in range(2000)}
    prompt = build_urgent_prompt(_seed(evidence=evidence), run_id=RUN_ID)
    assert TRUNCATION_MARKER in prompt
    assert "k00000" in prompt
    assert "k01999" not in prompt
    start = prompt.index("k00000")
    end = prompt.index(TRUNCATION_MARKER)
    assert end - start <= EVIDENCE_CHAR_CAP


def test_small_evidence_is_not_truncated():
    prompt = build_urgent_prompt(_seed(), run_id=RUN_ID)
    assert TRUNCATION_MARKER not in prompt


def test_empty_evidence_is_said_plainly():
    prompt = build_urgent_prompt(_seed(evidence={}), run_id=RUN_ID)
    assert "No evidence was collected" in prompt

"""The :Hop and :TurnOutcome writes Orion is taught are safe to send twice (#3543).

Both used to be taught as CREATE, so a resent command (after a compaction, or a
retry that did not see the first reply) minted a second node: live 2026-11-17,
82 duplicated (run_id, n) Hop pairs and 22 duplicated TurnOutcome run_ids, and
`agency_episode.py`'s `OPTIONAL MATCH (h:Hop {run_id, n})` returned two rows.

The commands under test are pulled out of the RENDERED prompt, not restated, so
a later edit to the prompt text is what gets checked.
"""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone

import pytest

from orion.curiosity.kickoff_prompt import _outcome_section, build_kickoff_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.worldview import WorldviewSnapshot

RUN_ID = "abc123def456"


def _prompt() -> str:
    material = StudyMaterial(generated_at=datetime(2026, 11, 17, tzinfo=timezone.utc))
    return build_kickoff_prompt(material, view=WorldviewSnapshot(), run_id=RUN_ID)


def taught_hop(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("MERGE (h:Hop")]
    assert len(lines) == 1, lines
    return lines[0]


def taught_outcome_block(focus: bool) -> str:
    """The command exactly as taught: multi-line, `//` comments included."""
    text = "\n".join(_outcome_section(run_id=RUN_ID, focus=focus))
    start = text.index("MERGE (t:TurnOutcome")
    return text[start: text.index("\n\n", start)]


def taught_outcome(focus: bool) -> str:
    return " ".join(ln.split("//")[0].strip() for ln in taught_outcome_block(focus).splitlines())


def test_hop_merges_on_identity_only():
    hop = taught_hop(_prompt())
    pattern = re.match(r"MERGE \(h:Hop \{([^}]*)\}\)", hop).group(1)
    assert pattern == f'run_id: "{RUN_ID}", n: 1'
    assert "ON CREATE SET h.note =" in hop and "h.written_at = timestamp()" in hop
    # A reused n silently keeps the earlier note, so the note comes back to be checked.
    assert hop.endswith("RETURN h.note")
    assert "that n is already taken by an earlier stop" in _prompt()


@pytest.mark.parametrize("focus", [False, True])
def test_turn_outcome_merges_on_run_id_only(focus):
    cmd = taught_outcome(focus)
    assert cmd.startswith(f'MERGE (t:TurnOutcome {{run_id: "{RUN_ID}"}}) ON CREATE SET t.written_at = timestamp() SET')
    # The decisions are a plain SET, not ON CREATE: a focus anchor may have made the
    # node first (orion/harness/task_anchor.py), and the decisions must still land.
    assert "t.continue_line = true" in cmd and "t.reach_out_why" in cmd


def test_no_taught_create_and_resend_sentence_is_true():
    text = _prompt()
    assert "CREATE (:Hop" not in text and "CREATE (:TurnOutcome" not in text
    assert "a :Hop or :TurnOutcome is not" not in text
    assert "so a resend never makes a second one" in text


def test_edges_to_a_prior_match_it_first_never_inline():
    text = _prompt()
    assert "Never draw a node inline inside an arrow" in text
    assert '(p:Prior {prior_id: "..."})' in text
    assert "-[:ABOUT]->(:Prior)`: that does not find your prior" in text


def _focus_anchor_statement() -> str:
    from types import SimpleNamespace

    from orion.harness.task_anchor import build_focus_anchor

    block = build_focus_anchor(SimpleNamespace(question="why?", run_id=RUN_ID, graph="g"))
    assert block, "focus anchor disabled in this environment"
    stmt = block.split("this one statement: ", 1)[1].split(". Replace the placeholder", 1)[0]
    return stmt.replace("<explored | set_aside: why not>", "explored")


def test_focus_anchor_and_outcome_share_one_key():
    assert f'MERGE (t:TurnOutcome {{run_id: "{RUN_ID}"}})' in _focus_anchor_statement()


@pytest.mark.skipif(
    not os.environ.get("ORION_TEST_FALKORDB_PORT"),
    reason="needs a scratch FalkorDB (ORION_TEST_FALKORDB_PORT on 127.0.0.1); never point at production",
)
def test_real_falkordb_taught_writes_sent_twice_leave_one_of_each():
    import redis

    client = redis.Redis(host="127.0.0.1", port=int(os.environ["ORION_TEST_FALKORDB_PORT"]), decode_responses=True)
    g = f"taught_resend_{uuid.uuid4().hex[:8]}"

    def count(cypher: str) -> int:
        return int(client.execute_command("GRAPH.QUERY", g, cypher)[1][0][0])

    try:
        hop = taught_hop(_prompt())
        outcome = taught_outcome_block(focus=True).replace("<explored | set_aside: why not>", "explored")
        # The focus anchor's own MERGE lands first, as in a compacted focus run.
        client.execute_command("GRAPH.QUERY", g, _focus_anchor_statement())
        for _ in range(2):
            client.execute_command("GRAPH.QUERY", g, hop)
            client.execute_command("GRAPH.QUERY", g, outcome)
        assert count(f'MATCH (h:Hop {{run_id: "{RUN_ID}", n: 1}}) RETURN count(h)') == 1
        assert count(f'MATCH (t:TurnOutcome {{run_id: "{RUN_ID}"}}) RETURN count(t)') == 1
        assert count("MATCH (h:Hop) WHERE h.written_at IS NOT NULL AND h.note <> '' RETURN count(h)") == 1
        # The outcome's decision landed on the anchor's node instead of being lost.
        assert count(f'MATCH (t:TurnOutcome {{run_id: "{RUN_ID}"}}) WHERE t.continue_line = true RETURN count(t)') == 1
        # A different stop reusing n=1 does not overwrite: the returned note shows the old one.
        reused = hop.replace("what I just learned", "a different stop")
        reply = client.execute_command("GRAPH.QUERY", g, reused)
        assert "a different stop" not in str(reply[1]) and "what I just learned" in str(reply[1])
        assert count(f'MATCH (h:Hop {{run_id: "{RUN_ID}"}}) RETURN count(h)') == 1
        # Matching both ends first never mints a blank prior.
        client.execute_command("GRAPH.QUERY", g, 'MERGE (p:Prior {prior_id: "x"}) ON CREATE SET p.claim = "c"')
        client.execute_command("GRAPH.QUERY", g, 'MERGE (f:Finding {finding_id: "f"}) ON CREATE SET f.text = "t"')
        edge = (f'MATCH (f:Finding {{finding_id: "f"}}), (p:Prior {{prior_id: "x"}}) '
                f'MERGE (f)-[e:SUPPORTS]->(p) ON CREATE SET e.run_id = "{RUN_ID}"')
        for _ in range(2):
            client.execute_command("GRAPH.QUERY", g, edge)
        assert count("MATCH (p:Prior) RETURN count(p)") == 1
        assert count("MATCH (:Finding)-[e:SUPPORTS]->(:Prior) RETURN count(e)") == 1
    finally:
        client.execute_command("GRAPH.DELETE", g)

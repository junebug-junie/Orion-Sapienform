from orion.curiosity.self_inquiry import SELF_PRIOR_LINE
from orion.curiosity.worldview import LIVE_NON_SELF_PRIORS_CYPHER, LIVE_PRIORS_CYPHER


def test_non_self_cypher_excludes_self_line_and_keeps_live_rule() -> None:
    assert "LIVE_NON_SELF_PRIORS_CYPHER" in dir(__import__("orion.curiosity.worldview", fromlist=["*"]))
    q = LIVE_NON_SELF_PRIORS_CYPHER
    assert SELF_PRIOR_LINE in q
    assert "p.line" in q
    assert "refuted" in q or "CLOSED" in q or "retired_unresolvable" in q
    assert q != LIVE_PRIORS_CYPHER

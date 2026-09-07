from orion.cockpit.builders import hop_from_run_artifact
from orion.cockpit.sequencer import advance_seq, next_seq, reset_seq


def test_next_seq_monotonic_per_correlation():
    reset_seq("corr-a")
    reset_seq("corr-b")
    assert next_seq("corr-a") == 0
    assert next_seq("corr-a") == 1
    assert next_seq("corr-b") == 0
    assert next_seq("corr-a") == 2


def test_advance_seq_after_multi_hop_batch():
    cid = "corr-batch"
    reset_seq(cid)
    assert next_seq(cid) == 0
    hops = hop_from_run_artifact(
        correlation_id=cid,
        seq=0,
        run={"draft_text": "draft", "reflection": "done"},
    )
    assert len(hops) == 2
    advance_seq(cid, len(hops))
    assert next_seq(cid) == 2

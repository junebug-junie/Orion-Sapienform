from orion.cockpit.sequencer import next_seq, reset_seq


def test_next_seq_monotonic_per_correlation():
    reset_seq("corr-a")
    reset_seq("corr-b")
    assert next_seq("corr-a") == 0
    assert next_seq("corr-a") == 1
    assert next_seq("corr-b") == 0
    assert next_seq("corr-a") == 2

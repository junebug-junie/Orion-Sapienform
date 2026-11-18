"""--gate: 'pool' (default) keeps the conservative quiet-pool pause; 'none' lets the pool's
background priority and the hub's in-flight cap pace re-reads (live 2026-10-14: the pool gate
held --concurrency 3 to one read behind an unrelated background hold)."""
import pytest

from scripts import reading_reread_backfill as bf


def test_gate_defaults_to_pool():
    assert bf.parse_args(["run", "--yes"]).gate == "pool"


def test_gate_none_is_accepted():
    assert bf.parse_args(["run", "--yes", "--gate", "none", "--concurrency", "3"]).gate == "none"


def test_gate_rejects_unknown_value():
    with pytest.raises(SystemExit):
        bf.parse_args(["run", "--yes", "--gate", "loose"])


def test_gate_none_skips_the_pool_check_in_source():
    src = open(bf.__file__).read()
    assert 'busy = [] if args.gate == "none" else await _pool_busy(' in src

"""#3395: the operator commands and the replay batch in scripts/worldview_prior_unique_rule.py."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "wpur", Path(__file__).resolve().parents[1] / "scripts" / "worldview_prior_unique_rule.py")
wpur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wpur)


def test_apply_commands_index_before_rule_on_prior_id():
    cmds = wpur.apply_commands()
    assert "CREATE INDEX FOR (p:Prior) ON (p.prior_id)" in cmds[0]
    assert "GRAPH.CONSTRAINT CREATE orion_worldview UNIQUE NODE Prior PROPERTIES 1 prior_id" in cmds[1]
    assert "GRAPH.CONSTRAINT DROP orion_worldview" in wpur.rollback_commands()[0]


def test_replay_batch_uses_only_repeat_safe_writes():
    batch = wpur.taught_batch("aaaaaaaaaaaa")
    assert not [c for c in batch if c.startswith("CREATE")]
    count = [c for c in batch if "times_tested = coalesce" in c]
    assert len(count) == 1 and 'coalesce(p.last_run_id, "") <> "aaaaaaaaaaaa"' in count[0]
    assert 'p.last_run_id = "aaaaaaaaaaaa"' in count[0] and "last_counted_run_id" not in "".join(batch)
    assert sum("MERGE (v:PriorRevision" in c for c in batch) == 2


def test_scratch_never_targets_the_live_container_by_default():
    import inspect

    src = inspect.getsource(wpur.scratch)
    assert "orion-patchf-scratch-" in src and "docker\", \"run\"" in src
    assert "GRAPH.QUERY\", g" in src and "orion_worldview" not in src

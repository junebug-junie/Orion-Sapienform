"""``ORGAN_REGISTRY`` integrity (phase-2 causal DAG contract)."""
from orion.signals.registry import ORGAN_REGISTRY


def test_causal_parent_organs_exist_in_registry() -> None:
    for oid, entry in ORGAN_REGISTRY.items():
        for p in entry.causal_parent_organs:
            assert p in ORGAN_REGISTRY, f"{oid} references unknown parent {p!r}"


def test_no_self_parent_edge() -> None:
    for oid, entry in ORGAN_REGISTRY.items():
        assert oid not in entry.causal_parent_organs, f"{oid} must not list itself as parent"


def test_registry_acyclic() -> None:
    """DFS from each organ must not revisit nodes on the stack (no cycles)."""

    def has_cycle(start: str) -> bool:
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node: str) -> bool:
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            entry = ORGAN_REGISTRY.get(node)
            if entry:
                for p in entry.causal_parent_organs:
                    if p not in ORGAN_REGISTRY:
                        continue
                    if dfs(p):
                        return True
            stack.remove(node)
            return False

        return dfs(start)

    for oid in ORGAN_REGISTRY:
        assert not has_cycle(oid), f"cycle detected involving {oid!r}"


def test_world_pulse_parents_definitive_empty() -> None:
    wp = ORGAN_REGISTRY["world_pulse"]
    assert wp.causal_parent_organs == []

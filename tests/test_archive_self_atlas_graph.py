"""scripts/archive_self_atlas_graph.py copies orion_substrate_self and never deletes it."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "archive_self_atlas_graph.py"
_spec = importlib.util.spec_from_file_location("archive_self_atlas_graph", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(mod)


class FakeFalkor:
    def __init__(self, graphs: dict[str, tuple[int, int]], *, drift_on_copy: bool = False) -> None:
        self.graphs = dict(graphs)
        self.calls: list[tuple] = []
        self.drift_on_copy = drift_on_copy

    def __call__(self, *args):
        self.calls.append(args)
        cmd = args[0]
        if cmd == "EXISTS":
            return int(args[1] in self.graphs)
        if cmd == "GRAPH.RO_QUERY":
            nodes, edges = self.graphs[args[1]]
            value = nodes if args[2] == mod.NODE_COUNT_QUERY else edges
            return [["count"], [[value]], ["Cached execution: 0"]]
        if cmd == "GRAPH.COPY":
            self.graphs[args[2]] = self.graphs[args[1]]
            if self.drift_on_copy:
                n, e = self.graphs[args[1]]
                self.graphs[args[1]] = (n + 1, e)
            return "OK"
        raise AssertionError(f"unexpected command {args!r}")


def _commands(fake: FakeFalkor) -> list[str]:
    return [c[0] for c in fake.calls]


def test_dry_run_reads_only() -> None:
    fake = FakeFalkor({mod.SOURCE_GRAPH: (22169, 54457)})
    out: list[str] = []
    assert mod.archive(fake, archive_graph=mod.DEFAULT_ARCHIVE_GRAPH, apply=False, out=out.append) == 0
    assert set(_commands(fake)) <= {"EXISTS", "GRAPH.RO_QUERY"}
    assert mod.DEFAULT_ARCHIVE_GRAPH not in fake.graphs
    assert any("DRY RUN" in line for line in out)


def test_apply_copies_and_verifies_and_keeps_the_source() -> None:
    fake = FakeFalkor({mod.SOURCE_GRAPH: (22169, 54457)})
    out: list[str] = []
    assert mod.archive(fake, archive_graph="arch", apply=True, out=out.append) == 0
    assert fake.graphs["arch"] == (22169, 54457)
    assert mod.SOURCE_GRAPH in fake.graphs
    assert ("GRAPH.COPY", mod.SOURCE_GRAPH, "arch") in fake.calls
    assert any("VERIFIED" in line for line in out)


def test_never_deletes() -> None:
    fake = FakeFalkor({mod.SOURCE_GRAPH: (5, 3)})
    mod.archive(fake, archive_graph="arch", apply=True, out=lambda _l: None)
    assert not any("DELETE" in str(c[0]).upper() or c[0] in ("DEL", "UNLINK") for c in fake.calls)


def test_refuses_existing_archive() -> None:
    fake = FakeFalkor({mod.SOURCE_GRAPH: (5, 3), "arch": (1, 1)})
    assert mod.archive(fake, archive_graph="arch", apply=True, out=lambda _l: None) == 2
    assert "GRAPH.COPY" not in _commands(fake)
    assert fake.graphs["arch"] == (1, 1)


def test_refuses_missing_source_and_same_name() -> None:
    assert mod.archive(FakeFalkor({}), archive_graph="arch", apply=True, out=lambda _l: None) == 2
    fake = FakeFalkor({mod.SOURCE_GRAPH: (5, 3)})
    assert mod.archive(fake, archive_graph=mod.SOURCE_GRAPH, apply=True, out=lambda _l: None) == 2
    assert fake.calls == []


def test_mismatch_when_source_still_being_written() -> None:
    fake = FakeFalkor({mod.SOURCE_GRAPH: (5, 3)}, drift_on_copy=True)
    out: list[str] = []
    assert mod.archive(fake, archive_graph="arch", apply=True, out=out.append) == 1
    assert any("MISMATCH" in line for line in out)


def test_delete_command_is_printed_not_run() -> None:
    assert mod.delete_command("redis://127.0.0.1:6380") == (
        "redis-cli -u redis://127.0.0.1:6380 GRAPH.DELETE orion_substrate_self"
    )

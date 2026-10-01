"""In-memory stand-in for `orion_worldview` that runs exactly the two queries
`orion.curiosity.write_stamp.WriteStamper` issues. Shared by the stamper and
FCC-motor tests."""

from __future__ import annotations

import re
from typing import Any

from orion.curiosity.write_stamp import STAMP_SOURCE, STAMPED_LABELS, baseline_cypher


class FakeGraph:
    """Executes exactly the two queries WriteStamper issues, over a node table."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.calls: list[tuple[str, str]] = []
        self.clock = 1_000
        self.fail = False
        self.closed = False

    def create(self, label: str, **props: Any) -> int:
        node_id = len(self.nodes)
        self.nodes.append({"id": node_id, "labels": [label], "props": dict(props)})
        return node_id

    def _unstamped(self) -> list[dict[str, Any]]:
        return [
            n for n in self.nodes
            if n["props"].get("written_at") is None and any(l in STAMPED_LABELS for l in n["labels"])
        ]

    def execute_command(self, cmd: str, graph: str, cypher: str) -> Any:
        if self.fail:
            raise ConnectionError("falkordb down")
        self.calls.append((cmd, cypher))
        if cypher == baseline_cypher():
            assert cmd == "GRAPH.RO_QUERY"
            return [["id"], [[n["id"]] for n in self._unstamped()], []]
        match = re.match(r"CYPHER baseline=\[([0-9,]*)\] ", cypher)
        assert match and cmd == "GRAPH.QUERY", cypher
        baseline = {int(x) for x in match.group(1).split(",") if x}
        self.clock += 1
        hit = [n for n in self._unstamped() if n["id"] not in baseline]
        for n in hit:
            n["props"]["written_at"] = self.clock
            n["props"]["written_at_source"] = STAMP_SOURCE
        return [["stamped"], [[len(hit)]], []]

    def close(self) -> None:
        self.closed = True

    def props(self, node_id: int) -> dict[str, Any]:
        return self.nodes[node_id]["props"]

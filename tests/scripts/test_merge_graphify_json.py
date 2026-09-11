"""Run with unittest; no Graphify installation or network required."""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("merge_graphify_json", ROOT / "scripts/merge_graphify_json.py")
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


def graph():
    return {"directed": False, "multigraph": False, "nodes": [{"id": "a", "source_location": "old"}],
            "links": [], "graph": {"hyperedges": []}, "hyperedges": [], "built_at_commit": "base"}


class GraphMergeTests(unittest.TestCase):
    def test_three_way_union_preserves_both_sides_and_metadata(self):
        base = graph()
        ours, theirs = copy.deepcopy(base), copy.deepcopy(base)
        ours["nodes"][0]["source_location"] = "new"
        ours["nodes"].append({"id": "b"})
        theirs["nodes"].append({"id": "c"})
        ours["links"] = [{"source": "a", "target": "b", "relation": "calls"}]
        theirs["links"] = [{"source": "a", "target": "c", "relation": "calls"}]
        ours["hyperedges"] = [{"id": "h1", "nodes": ["a", "b"]}]
        theirs["hyperedges"] = [{"id": "h2", "nodes": ["a", "c"]}]
        ours["graph"]["hyperedges"] = ours["hyperedges"]
        theirs["graph"]["hyperedges"] = theirs["hyperedges"]
        ours["custom"] = {"left": 1}
        theirs["custom"] = {"right": 2}
        result = merge.merge_graphs(base, ours, theirs)
        self.assertEqual({n["id"] for n in result["nodes"]}, {"a", "b", "c"})
        self.assertEqual(result["nodes"][0]["source_location"], "new")
        self.assertEqual(len(result["links"]), 2)
        self.assertEqual({h["id"] for h in result["hyperedges"]}, {"h1", "h2"})
        self.assertEqual(result["graph"]["hyperedges"], result["hyperedges"])
        self.assertEqual(result["custom"], {"left": 1, "right": 2})
        self.assertEqual(result["built_at_commit"], "base")

    def test_reversed_edges_and_independent_attribute_edits(self):
        base = graph()
        base["nodes"].append({"id": "b"})
        base["links"] = [{"source": "a", "target": "b", "source_location": "old", "weight": 1}]
        ours, theirs = copy.deepcopy(base), copy.deepcopy(base)
        ours["links"][0]["source_location"] = "new"
        theirs["links"][0].update(source="b", target="a", weight=2)
        result = merge.merge_graphs(base, ours, theirs)
        self.assertEqual(result["links"], [{"source": "a", "target": "b", "source_location": "new", "weight": 2}])

    def test_refuses_corruption_and_competing_edits_without_touching_current(self):
        for bad in ("{broken", json.dumps(dict(graph(), links=[{"source": "a", "target": "missing"}]))):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory) / name for name in ("base", "current", "other")]
                for path in paths:
                    path.write_text(json.dumps(graph()))
                paths[2].write_text(bad)
                before = paths[1].read_bytes()
                with self.assertRaises(ValueError):
                    merge.main(list(map(str, paths)))
                self.assertEqual(paths[1].read_bytes(), before)
        base = graph()
        ours, theirs = copy.deepcopy(base), copy.deepcopy(base)
        ours["nodes"][0]["source_location"] = "one"
        theirs["nodes"][0]["source_location"] = "two"
        with self.assertRaisesRegex(ValueError, "competing edits"):
            merge.merge_graphs(base, ours, theirs)

    def test_rejects_incompatible_graphs_and_duplicate_ids(self):
        base = graph()
        for changed in (dict(graph(), directed=True), dict(graph(), nodes=[{"id": "a"}, {"id": "a"}])):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                merge.merge_graphs(base, graph(), changed)

    def test_directed_multigraph_retains_opposite_and_parallel_edges(self):
        base = dict(graph(), directed=True, multigraph=True, nodes=[{"id": "a"}, {"id": "b"}])
        ours, theirs = copy.deepcopy(base), copy.deepcopy(base)
        ours["links"] = [{"source": "a", "target": "b", "key": 0}]
        theirs["links"] = [{"source": "a", "target": "b", "key": 1}, {"source": "b", "target": "a", "key": 0}]
        self.assertEqual(len(merge.merge_graphs(base, ours, theirs)["links"]), 3)

    def test_hyperedge_members_merge_and_dangling_members_fail(self):
        base = dict(graph(), nodes=[{"id": value} for value in ("a", "b", "c")], hyperedges=[{"id": "h", "nodes": ["a"]}])
        ours, theirs = copy.deepcopy(base), copy.deepcopy(base)
        ours["hyperedges"][0]["nodes"].append("b")
        theirs["hyperedges"][0]["nodes"].append("c")
        self.assertEqual(merge.merge_graphs(base, ours, theirs)["hyperedges"][0]["nodes"], ["a", "b", "c"])
        theirs["hyperedges"][0]["nodes"].append("missing")
        with self.assertRaisesRegex(ValueError, "hyperedge member"):
            merge.merge_graphs(base, ours, theirs)

    def test_one_sided_deletion_does_not_drop_hyperedges_or_metadata(self):
        base = graph()
        base["hyperedges"] = [{"id": "h", "nodes": ["a"]}]
        base["graph"]["hyperedges"] = copy.deepcopy(base["hyperedges"])
        base["custom"] = {"keep": True}
        ours, theirs = copy.deepcopy(base), copy.deepcopy(base)
        theirs["hyperedges"] = []
        theirs["graph"]["hyperedges"] = []
        del theirs["custom"]
        result = merge.merge_graphs(base, ours, theirs)
        self.assertEqual(result["hyperedges"], base["hyperedges"])
        self.assertEqual(result["graph"]["hyperedges"], base["hyperedges"])
        self.assertEqual(result["custom"], base["custom"])

    def test_legacy_hyperedge_locations_are_combined_for_both_consumers(self):
        base, ours, theirs = graph(), graph(), graph()
        ours["hyperedges"] = [{"id": "root", "nodes": ["a"]}]
        theirs["graph"]["hyperedges"] = [{"id": "nested", "nodes": ["a"]}]
        result = merge.merge_graphs(base, ours, theirs)
        self.assertEqual({h["id"] for h in result["hyperedges"]}, {"root", "nested"})
        self.assertEqual(result["hyperedges"], result["graph"]["hyperedges"])

    def test_accepts_graph_larger_than_old_50_mib_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("base", "current", "other")]
            for path in paths:
                path.write_text(json.dumps(graph()))
            with paths[1].open("w") as handle:
                for _ in range(51):
                    handle.write(" " * 1024 * 1024)
                json.dump(graph(), handle)
            self.assertGreater(paths[1].stat().st_size, 50 * 1024 * 1024)
            self.assertEqual(merge.main(list(map(str, paths))), 0)
            self.assertEqual(json.loads(paths[1].read_text()), graph())

    @unittest.skipUnless(shutil.which("git-lfs"), "requires Git LFS")
    def test_lfs_wrapper_returns_pointer_that_resolves_to_merged_json(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "init", "-q", directory], check=True)
            paths = [Path(directory) / name for name in ("base", "current", "other")]
            for path in paths:
                path.write_text(json.dumps(graph()))
            other = graph()
            other["nodes"].append({"id": "b"})
            paths[2].write_text(json.dumps(other))
            # Exercise actual LFS pointer inputs, including when ambient smudge is disabled.
            for path in paths:
                pointer = subprocess.check_output(["git", "lfs", "clean", "--", path.name], input=path.read_bytes(), cwd=directory)
                path.write_bytes(pointer)
            subprocess.run(["sh", str(ROOT / "scripts/graphify_lfs_merge_driver.sh"), *map(str, paths)], cwd=directory,
                           env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"), check=True)
            self.assertTrue(paths[1].read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"))
            merged = subprocess.check_output(["git", "lfs", "smudge", "--", "current"], input=paths[1].read_bytes(), cwd=directory)
            self.assertEqual({n["id"] for n in json.loads(merged)["nodes"]}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()

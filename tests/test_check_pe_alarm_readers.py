"""scripts/check_pe_alarm_readers.py (#3307): every prediction_error reader says how it treats alarms."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_pe_alarm_readers", REPO / "scripts/check_pe_alarm_readers.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _repo(tmp_path: Path, files: dict[str, str], registry: dict) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    reg = tmp_path / gate.REGISTRY
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(yaml.safe_dump({"version": 1, **registry}))
    return tmp_path


GENERIC = "def top(nodes):\n    return max(n.metadata.get('prediction_error', 0) for n in nodes)\n"
FILTERED = ("from orion.substrate.prediction_error_freshness import is_alarm_node\n"
            "def top(nodes):\n    return max(n.metadata['prediction_error'] for n in nodes if not is_alarm_node(n.id))\n")


def test_the_checked_in_repo_passes():
    assert gate.check(REPO) == []


def test_a_new_generic_reader_that_forgets_alarms_fails(tmp_path):
    root = _repo(tmp_path, {"orion/new_reader.py": GENERIC}, {})
    problems = gate.check(root)
    assert len(problems) == 1 and "orion/new_reader.py" in problems[0] and "is_alarm_node" in problems[0]
    assert gate.main(["--root", str(root)]) == 1


def test_a_reader_that_skips_alarms_passes_without_registering(tmp_path):
    assert gate.check(_repo(tmp_path, {"orion/reader.py": FILTERED}, {})) == []


def test_history_table_and_attribute_reads_count_as_touching(tmp_path):
    root = _repo(tmp_path, {"services/x/app/a.py": "SQL = 'select * from substrate_node_prediction_error_history'\n",
                            "services/x/app/b.py": "v = node.prediction_error\n"}, {})
    assert {p.split(" ")[0] for p in gate.check(root)} == {"services/x/app/a.py", "services/x/app/b.py"}


def test_registered_with_a_class_and_reason_passes_and_each_needs_both(tmp_path):
    ok = _repo(tmp_path / "ok", {"orion/w.py": GENERIC},
               {"files": {"orion/w.py": {"class": "not_a_reader", "reason": "writes the value"}}})
    assert gate.check(ok) == []
    bad = _repo(tmp_path / "bad", {"orion/w.py": GENERIC},
                {"files": {"orion/w.py": {"class": "trust_me", "reason": ""}}})
    problems = gate.check(bad)
    assert any("not one of" in p for p in problems) and any("no reason" in p for p in problems)


def test_stale_and_redundant_entries_fail(tmp_path):
    root = _repo(tmp_path, {"orion/gone_quiet.py": "x = 1\n", "orion/reader.py": FILTERED},
                 {"files": {"orion/missing.py": {"class": "not_a_reader", "reason": "r"},
                            "orion/gone_quiet.py": {"class": "not_a_reader", "reason": "r"},
                            "orion/reader.py": {"class": "filters_via", "reason": "r"}}})
    problems = gate.check(root)
    assert any("orion/missing.py" in p and "no longer exists" in p for p in problems)
    assert any("orion/gone_quiet.py" in p and "no longer touches" in p for p in problems)
    assert any("orion/reader.py" in p and "already filters" in p for p in problems)


def test_every_spelling_of_the_key_counts_as_touching(tmp_path):
    files = {
        "orion/a_getattr.py": "v = getattr(node, 'prediction_error', 0)\n",
        "orion/b_member.py": "hot = [k for k, vec in vectors.items() if 'prediction_error' in vec]\n",
        "orion/c_sql.py": "SQL = \"select field_json -> 'node_vectors' ->> 'prediction_error'\"\n",
        "orion/d_const.py": "PE_KEY = \"prediction_error\"\nv = md.get(PE_KEY)\n",
        "orion/e_fstring.py": "q = f\"{node}.prediction_error\"\n",
    }
    root = _repo(tmp_path, files, {})
    assert {p.split(" ")[0] for p in gate.check(root)} == set(files)


def test_a_mention_in_a_comment_or_docstring_is_not_a_filter(tmp_path):
    text = ('"""Uses is_alarm_node somewhere else."""\n'
            "# ALARM_NODE_IDS are handled upstream\n" + GENERIC)
    root = _repo(tmp_path, {"orion/claims.py": text}, {})
    assert len(gate.check(root)) == 1


def test_excluded_paths_must_be_narrow_existing_directories(tmp_path):
    root = _repo(tmp_path, {"orion/x.py": GENERIC, "scripts/analysis_x.py": GENERIC},
                 {"excluded_paths": [{"path": "orion/", "reason": "too broad"},
                                     {"path": "scripts/analysis", "reason": "no slash"},
                                     {"path": "scripts/nope/", "reason": "missing"}]})
    problems = gate.check(root)
    assert sum("must be an existing directory" in p for p in problems) == 3
    assert any(p.startswith("orion/x.py") for p in problems)              # not excluded by the bad entries
    assert any(p.startswith("scripts/analysis_x.py") for p in problems)


def test_tests_evals_and_excluded_paths_are_not_scanned(tmp_path):
    root = _repo(tmp_path, {"orion/tests/test_x.py": GENERIC, "services/s/evals/run.py": GENERIC,
                            "scripts/analysis/replay.py": GENERIC},
                 {"excluded_paths": [{"path": "scripts/analysis/", "reason": "offline, read by a person"}]})
    assert gate.check(root) == []
    no_reason = _repo(tmp_path / "nr", {"scripts/analysis/replay.py": GENERIC},
                      {"excluded_paths": [{"path": "scripts/analysis/", "reason": ""}]})
    assert any("no reason" in p for p in gate.check(no_reason))


def test_an_unparseable_file_never_auto_passes_on_a_comment(tmp_path):
    broken = "# is_alarm_node handles this\n" + GENERIC + "def broken(:\n    x = 'unterminated\n"
    assert len(gate.check(_repo(tmp_path, {"orion/broken.py": broken}, {}))) == 1


# --- moved-node writer fence (PE retire Patch 2) ---------------------------------------------------

_REG = {"files": {"services/x/app/w.py": {"class": "not_a_reader", "reason": "writer"}}}


def test_writing_prediction_error_for_an_alarm_id_literal_fails(tmp_path):
    src = ("def tick(self, now):\n"
           "    self._write_prediction_error_node(node_id='node:substrate.cabinet', error=1.0, now=now)\n"
           "    return {'prediction_error': 1.0}\n")
    problems = gate.check(_repo(tmp_path, {"services/x/app/w.py": src}, _REG))
    assert len(problems) == 1 and "services/x/app/w.py:2" in problems[0]
    assert "node:substrate.cabinet" in problems[0] and "_write_signal_node" in problems[0]
    assert gate.main(["--root", str(tmp_path)]) == 1


def test_writing_prediction_error_for_an_alarm_through_a_constant_fails(tmp_path):
    files = {
        "orion/consts.py": "STRAIN_NODE_ID = 'node:substrate.strain'\n",
        "services/x/app/w.py": ("from orion.consts import STRAIN_NODE_ID\n"
                                "def tick(store, now):\n"
                                "    store.save_prediction_error_history_samples([(STRAIN_NODE_ID, now, 1.0)])\n"),
    }
    problems = gate.check(_repo(tmp_path, files, {}))
    assert len(problems) == 1 and "STRAIN_NODE_ID" in problems[0] and ":3 " in problems[0]


def test_a_prediction_error_dict_naming_an_alarm_fails(tmp_path):
    src = "ROW = {'node_id': 'node:substrate.street', 'prediction_error': 1.0}\n"
    problems = gate.check(_repo(tmp_path, {"services/x/app/w.py": src}, _REG))
    assert len(problems) == 1 and "node:substrate.street" in problems[0]


def test_writing_prediction_error_for_a_real_pe_node_still_passes(tmp_path):
    src = ("def tick(self, now):\n"
           "    self._write_prediction_error_node(node_id='node:prediction.body', error=1.0, now=now)\n"
           "    self._write_signal_node(node_id='node:substrate.cabinet', value=1.0, now=now)\n"
           "    return {'prediction_error': 1.0}\n")
    assert gate.check(_repo(tmp_path, {"services/x/app/w.py": src}, _REG)) == []


def test_the_live_worker_writes_no_alarm_or_camera_through_the_pe_writer():
    assert gate.moved_writer_problems(REPO) == []
    assert gate.moved_node_ids() == {
        "node:substrate.cabinet", "node:substrate.strain", "node:substrate.street",
        "node:substrate.perception", "node:substrate.perception_walkway",
    }


# --- PE retire Patch 3: the cameras -------------------------------------------------------------


@pytest.mark.parametrize("call", [
    "self._write_prediction_error_node(node_id='node:substrate.perception', error=0.4, now=now)",
    "self._store.save_receipt(_prediction_error_receipt(reducer_key='perception_walkway', "
    "node_id='node:substrate.perception_walkway', prediction_error=0.4, now=now))",
    "store.save_prediction_error_history_samples([(PERCEPTION_NODE_ID, now, 0.4)])",
])
def test_a_producer_writing_prediction_error_for_a_camera_fails(tmp_path, call):
    files = {
        "orion/consts.py": "PERCEPTION_NODE_ID = 'node:substrate.perception'\n",
        "services/x/app/w.py": f"def tick(self, store, now):\n    {call}\n",
    }
    problems = gate.moved_writer_problems(_repo(tmp_path, files, {}))
    assert len(problems) == 1 and "services/x/app/w.py:2" in problems[0]
    assert "moved signal node" in problems[0]
    assert gate.main(["--root", str(tmp_path)]) == 1


_UNSCALED = (
    "SQL = \"select value from substrate_node_signal_history "
    "where node_id = 'node:substrate.perception_walkway'\"\n"
)


def test_sql_naming_the_signal_table_with_a_camera_and_no_scale_fails(tmp_path):
    root = _repo(tmp_path, {"services/x/evals/run_calm.py": _UNSCALED,
                            "orion/y.py": "Q = f\"select * from {SIGNAL_HISTORY_TABLE} where node_id = {PERCEPTION_NODE_ID}\"\n"},
                 {})
    problems = gate.check(root)
    assert {p.split(":")[0] for p in problems} == {"services/x/evals/run_calm.py", "orion/y.py"}
    assert all("scale" in p for p in problems)


@pytest.mark.parametrize("pred", ["AND scale = 'v2'", "AND scale IN ('v2','v2_partial')"])
def test_sql_with_a_scale_predicate_or_without_a_camera_passes(tmp_path, pred):
    ok = _UNSCALED.replace("'\"\n", f"' {pred}\"\n")
    alarm_only = "SQL = \"select value from substrate_node_signal_history where node_id = 'node:substrate.strain'\"\n"
    assert gate.check(_repo(tmp_path, {"services/x/evals/run_calm.py": ok, "orion/a.py": alarm_only}, {})) == []


def test_tests_are_not_scanned_for_the_scale_fence(tmp_path):
    assert gate.check(_repo(tmp_path, {"orion/tests/test_x.py": _UNSCALED}, {})) == []


def test_the_shared_builder_carries_the_scale_predicate_for_cameras():
    from orion.prediction import signal_history_sql as m

    assert gate.unscaled_camera_sql_problems(REPO) == []
    assert gate.builder_scale_problems(m.history_with_moved_source_sql(), gate.camera_node_ids()) == []


def test_mutation_builder_without_the_scale_predicate_is_caught(monkeypatch):
    """Mutation: drop the predicate from the signal_only branch (or put a camera in a read-both
    union) and the fence must fail."""
    from orion.prediction import signal_history_sql as m

    cameras = gate.camera_node_ids()
    stripped = m.history_with_moved_source_sql().replace(f" AND {m.SCALE_PREDICATE_SQL}", "")
    assert gate.builder_scale_problems(stripped, cameras)
    read_both_cams = m.history_with_moved_source_sql({**m.HISTORY_READ_MODE_BY_NODE_ID,
                                                      **{c: m.READ_BOTH for c in cameras}})
    assert gate.builder_scale_problems(read_both_cams, cameras)
    monkeypatch.setattr(m, "history_with_moved_source_sql", lambda read_modes=None: stripped)
    assert gate.unscaled_camera_sql_problems(REPO)


def test_mutation_live_worker_writing_a_camera_through_the_pe_writer_is_caught(tmp_path):
    """Mutation of the real worker: route a camera back through _write_prediction_error_node."""
    src = (REPO / "services/orion-substrate-runtime/app/worker.py").read_text()
    mutated = src.replace("        self._write_signal_node(\n            node_id=node_id,\n            value=score,",
                          "        self._write_prediction_error_node(\n            node_id=PERCEPTION_NODE_ID,\n"
                          "            error=score,", 1)
    assert mutated != src
    files = {"services/orion-substrate-runtime/app/worker.py": mutated,
             "orion/consts.py": "PERCEPTION_NODE_ID = 'node:substrate.perception'\n"}
    problems = gate.moved_writer_problems(_repo(tmp_path, files, {}))
    assert any("PERCEPTION_NODE_ID" in p for p in problems)

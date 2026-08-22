"""Tests for scripts/check_metric_dead_wiring.py.

find_new_token_references() is exercised end-to-end against real
`ast.parse()` (no mocking of the AST visitor itself -- it's the same
orion.metrics.consumers._MetricVisitor scan_repo() already uses) via
monkeypatched git plumbing (_staged_files/_added_line_numbers/
_staged_content), so the AST-vs-regex distinction (comments/docstrings never
match) is actually tested, not assumed. main()'s DB-dependent path is
covered by mocking open_readonly_connection/liveness_for_node so no real
Postgres is touched.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.check_metric_dead_wiring as mod  # noqa: E402


def _fake_diff_run(hunks: dict[str, str], contents: dict[str, str]):
    """Patches subprocess.run so:
      - `git diff --cached --name-only ...` returns `hunks`' keys
      - `git diff --cached -U0 -- <path>` returns `hunks[path]`
      - `git show :<path>` returns `contents[path]`
    """

    def _run(cmd, **kwargs):
        result = mock.Mock()
        result.returncode = 0
        if "--name-only" in cmd:
            result.stdout = "\n".join(hunks.keys())
        elif "-U0" in cmd:
            result.stdout = hunks.get(cmd[-1], "")
        elif cmd[:2] == ["git", "show"]:
            path = cmd[-1].split(":", 1)[1]
            result.stdout = contents.get(path, "")
        else:
            result.stdout = ""
        return result

    return _run


def test_reuses_consumers_is_test_path_not_a_second_copy():
    """find_new_token_references() imports orion.metrics.consumers._is_test_
    path rather than maintaining its own regex -- code review 2026-08-20
    found an earlier hand-rolled version disagreed with consumers.py's
    (scan_repo()'s own tool) on `*_test.py`-suffixed files, so the gate and
    the lineage/blast-radius tooling it points agents at could silently
    disagree about what counts as a real reference. This just confirms the
    import exists and behaves as expected for the shapes this repo uses."""
    from orion.metrics.consumers import _is_test_path

    assert _is_test_path("tests/test_metric_liveness.py")
    assert _is_test_path("scripts/test_check_metric_lineage_liveness_wiring.py")
    assert _is_test_path("services/orion-x/evals/run_thing_eval.py")
    assert not _is_test_path("orion/metrics/liveness.py")
    assert not _is_test_path("services/orion-equilibrium-service/app/flow_metacog_gate.py")


def test_added_line_numbers_tracks_a_simple_added_line():
    diff = "@@ -10,0 +11,1 @@\n+    lane_age = row[\"broadcast_lane_age_sec\"]\n"
    with mock.patch("subprocess.run", side_effect=_fake_diff_run({"orion/x.py": diff}, {})):
        assert mod._added_line_numbers("orion/x.py") == {11}


def test_added_line_numbers_multiple_lines_in_one_hunk():
    diff = "@@ -0,0 +20,3 @@\n+    a = 1\n+    b = 2\n+    c = 3\n"
    with mock.patch("subprocess.run", side_effect=_fake_diff_run({"orion/x.py": diff}, {})):
        assert mod._added_line_numbers("orion/x.py") == {20, 21, 22}


def test_added_line_numbers_does_not_confuse_removed_lines():
    diff = "@@ -5,1 +5,1 @@\n-    old = 1\n+    new = 2\n"
    with mock.patch("subprocess.run", side_effect=_fake_diff_run({"orion/x.py": diff}, {})):
        assert mod._added_line_numbers("orion/x.py") == {5}


def test_added_line_numbers_handles_content_that_looks_like_a_diff_header():
    """Regression: an added line whose own text starts with '++' or '--'
    (e.g. `++counter;`) renders in the raw diff as `+++counter;` / a line
    that superficially resembles the '+++ b/path' file-header line. The
    parser must gate on hunk position, not on that text prefix, or it drops
    the line and desyncs every following line number in the hunk."""
    diff = (
        "diff --git a/orion/x.py b/orion/x.py\n"
        "index abc..def 100644\n"
        "--- a/orion/x.py\n"
        "+++ b/orion/x.py\n"
        "@@ -0,0 +7,2 @@\n"
        "+++counter;\n"
        "+    confidence = 1\n"
    )
    with mock.patch("subprocess.run", side_effect=_fake_diff_run({"orion/x.py": diff}, {})):
        assert mod._added_line_numbers("orion/x.py") == {7, 8}


def test_find_new_token_references_matches_a_real_attribute_access():
    diff = "@@ -0,0 +3,1 @@\n+    return frame.broadcast_lane_age_sec\n"
    content = "def read(frame):\n    x = 1\n    return frame.broadcast_lane_age_sec\n"
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"broadcast_lane_age_sec"})
    assert hits == {"broadcast_lane_age_sec": [("orion/x.py", 3)]}


def test_find_new_token_references_ignores_a_comment_mention():
    """The AST-based rewrite's whole point: a comment naming the token is
    invisible to the parser and must never match."""
    diff = "@@ -0,0 +2,1 @@\n+    # TODO: confidence is still wired here, revisit\n"
    content = "def read():\n    # TODO: confidence is still wired here, revisit\n    return 1\n"
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {}


def test_find_new_token_references_ignores_a_log_string_mention():
    diff = "@@ -0,0 +2,1 @@\n+    logger.info(f\"dropping dead metric confidence now\")\n"
    content = 'def read():\n    logger.info(f"dropping dead metric confidence now")\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {}


def test_find_new_token_references_matches_a_real_dict_access():
    diff = "@@ -0,0 +2,1 @@\n+    return row[\"confidence\"]\n"
    content = 'def read(row):\n    return row["confidence"]\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {"confidence": [("orion/x.py", 2)]}


def test_find_new_token_references_ignores_unchanged_lines_even_if_they_match():
    # The token appears on line 1 (untouched context), the real edit is line 3.
    diff = "@@ -3,1 +3,1 @@\n-    old = 1\n+    new = row[\"confidence\"]\n"
    content = 'def read(row):\n    x = row["confidence"]\n    new = row["confidence"]\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {"confidence": [("orion/x.py", 3)]}


def test_find_new_token_references_matches_a_real_get_call():
    diff = "@@ -0,0 +2,1 @@\n+    return row.get(\"confidence\")\n"
    content = 'def read(row):\n    return row.get("confidence")\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {"confidence": [("orion/x.py", 2)]}


def test_find_new_token_references_matches_a_collection_member():
    diff = "@@ -0,0 +1,1 @@\n+FIELDS = (\"confidence\",)\n"
    content = 'FIELDS = ("confidence",)\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {"confidence": [("orion/x.py", 1)]}


def test_find_new_token_references_ignores_a_bare_literal():
    """KIND_LITERAL -- a bare string constant with no code-shaped access
    around it (e.g. a standalone log/format argument) is NOT high confidence
    and must not block. Code review 2026-08-20: an earlier version accepted
    every visitor hit kind, including this one."""
    diff = "@@ -0,0 +1,1 @@\n+_LABEL = \"confidence\"\n"
    content = '_LABEL = "confidence"\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {}


def test_find_new_token_references_ignores_a_bare_compare():
    diff = "@@ -0,0 +2,1 @@\n+    return name == \"confidence\"\n"
    content = 'def check(name):\n    return name == "confidence"\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {}


def test_find_new_token_references_ignores_an_unrelated_field_kwarg():
    """`some_unrelated_fn(confidence=0.9)` names a kwarg that happens to
    share the token's spelling but has nothing to do with the covered
    metric -- KIND_FIELD_KWARG is a WRITE_KIND (this is how a real producer
    sets the value), and blocking a WRITE would be backwards: writing to a
    dead metric is how you'd revive it, not a wiring mistake."""
    diff = "@@ -0,0 +1,1 @@\n+some_unrelated_fn(confidence=0.9)\n"
    content = "some_unrelated_fn(confidence=0.9)\n"
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {}


def test_find_new_token_references_ignores_a_subscript_write():
    diff = "@@ -0,0 +2,1 @@\n+    row[\"confidence\"] = 0.9\n"
    content = 'def write(row):\n    row["confidence"] = 0.9\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {}


def test_find_new_token_references_ignores_a_channel_kwarg():
    diff = "@@ -0,0 +1,1 @@\n+Perturbation(channel=\"confidence\", intensity=1.0)\n"
    content = 'Perturbation(channel="confidence", intensity=1.0)\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": content}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {}


def test_find_new_token_references_skips_test_files():
    diff = "@@ -0,0 +2,1 @@\n+    assert row[\"broadcast_lane_age_sec\"] == 1\n"
    content = 'def t():\n    assert row["broadcast_lane_age_sec"] == 1\n'
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"tests/test_x.py": diff}, {"tests/test_x.py": content}),
    ):
        hits = mod.find_new_token_references(["tests/test_x.py"], {"broadcast_lane_age_sec"})
    assert hits == {}


def test_find_new_token_references_skips_non_python_files():
    diff = "@@ -0,0 +1,1 @@\n+confidence: 1\n"
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"config/x.yaml": diff}, {"config/x.yaml": "confidence: 1\n"}),
    ):
        hits = mod.find_new_token_references(["config/x.yaml"], {"confidence"})
    assert hits == {}


def test_find_new_token_references_skips_a_file_that_fails_to_parse():
    diff = "@@ -0,0 +1,1 @@\n+    confidence = (\n"
    with mock.patch(
        "subprocess.run",
        side_effect=_fake_diff_run({"orion/x.py": diff}, {"orion/x.py": "confidence = (\n"}),
    ):
        hits = mod.find_new_token_references(["orion/x.py"], {"confidence"})
    assert hits == {}


def test_main_returns_zero_and_emits_empty_json_when_nothing_staged(capsys):
    with mock.patch.object(mod, "_staged_files", return_value=[]):
        rc = mod.main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    import json as _json
    assert _json.loads(out) == {"blocked": []}


def test_main_emits_valid_json_on_the_common_no_match_path(capsys):
    fake_node = mock.Mock()
    fake_node.name = "confidence"
    fake_graph = mock.Mock()
    fake_graph.nodes = {"metric://x#confidence": fake_node}
    with mock.patch.object(mod, "_staged_files", return_value=["orion/x.py"]), \
         mock.patch.object(mod, "find_new_token_references", return_value={}), \
         mock.patch("orion.metrics.lineage.build_graph", return_value=fake_graph), \
         mock.patch("orion.metrics.liveness.has_registered_source", return_value=True):
        rc = mod.main(["--json"])
    assert rc == 0
    import json as _json
    assert _json.loads(capsys.readouterr().out) == {"blocked": []}


def test_main_blocks_on_unclean_verdict_for_a_new_reference(capsys):
    fake_node = mock.Mock()
    fake_node.name = "confidence"
    fake_outcome = mock.Mock(verdict="dead", detail="n=0 over 1h", sample_count=0)
    fake_graph = mock.Mock()
    fake_graph.nodes = {"metric://x#confidence": fake_node}

    with mock.patch.object(mod, "_staged_files", return_value=["orion/x.py"]), \
         mock.patch.object(
             mod, "find_new_token_references",
             return_value={"confidence": [("orion/x.py", 12)]},
         ), \
         mock.patch("orion.metrics.lineage.build_graph", return_value=fake_graph), \
         mock.patch("orion.metrics.liveness.has_registered_source", return_value=True), \
         mock.patch("orion.metrics.liveness.open_readonly_connection", return_value=mock.Mock()), \
         mock.patch("orion.metrics.liveness.liveness_for_node", return_value=fake_outcome), \
         mock.patch("orion.field.channel_glossary.CLEAN_VERDICTS", frozenset({"live", "quiet"})):
        rc = mod.main([])

    assert rc == 1
    err = capsys.readouterr().err
    assert "BLOCK" in err
    assert "confidence" in err


def test_main_passes_on_clean_verdict(capsys):
    fake_node = mock.Mock()
    fake_node.name = "confidence"
    fake_outcome = mock.Mock(verdict="live", detail="n=100 over 1h", sample_count=100)
    fake_graph = mock.Mock()
    fake_graph.nodes = {"metric://x#confidence": fake_node}

    with mock.patch.object(mod, "_staged_files", return_value=["orion/x.py"]), \
         mock.patch.object(
             mod, "find_new_token_references",
             return_value={"confidence": [("orion/x.py", 12)]},
         ), \
         mock.patch("orion.metrics.lineage.build_graph", return_value=fake_graph), \
         mock.patch("orion.metrics.liveness.has_registered_source", return_value=True), \
         mock.patch("orion.metrics.liveness.open_readonly_connection", return_value=mock.Mock()), \
         mock.patch("orion.metrics.liveness.liveness_for_node", return_value=fake_outcome), \
         mock.patch("orion.field.channel_glossary.CLEAN_VERDICTS", frozenset({"live", "quiet"})):
        rc = mod.main([])

    assert rc == 0


def test_main_fails_open_when_db_unreachable():
    fake_node = mock.Mock()
    fake_node.name = "confidence"
    fake_graph = mock.Mock()
    fake_graph.nodes = {"metric://x#confidence": fake_node}

    with mock.patch.object(mod, "_staged_files", return_value=["orion/x.py"]), \
         mock.patch.object(
             mod, "find_new_token_references",
             return_value={"confidence": [("orion/x.py", 12)]},
         ), \
         mock.patch("orion.metrics.lineage.build_graph", return_value=fake_graph), \
         mock.patch("orion.metrics.liveness.has_registered_source", return_value=True), \
         mock.patch("orion.metrics.liveness.open_readonly_connection", return_value=None):
        rc = mod.main([])

    assert rc == 0


def test_main_fails_open_when_connection_close_raises(capsys):
    """Regression: a close() failure on an already-aborted connection must
    not propagate as an uncaught exception -- the shell hook treats a crash
    exit the same as a real gate failure, which would block an unrelated
    commit."""
    fake_node = mock.Mock()
    fake_node.name = "confidence"
    fake_outcome = mock.Mock(verdict="live", detail="n=1 over 1h", sample_count=1)
    fake_graph = mock.Mock()
    fake_graph.nodes = {"metric://x#confidence": fake_node}
    fake_conn = mock.Mock()
    fake_conn.close.side_effect = Exception("connection already aborted")

    with mock.patch.object(mod, "_staged_files", return_value=["orion/x.py"]), \
         mock.patch.object(
             mod, "find_new_token_references",
             return_value={"confidence": [("orion/x.py", 12)]},
         ), \
         mock.patch("orion.metrics.lineage.build_graph", return_value=fake_graph), \
         mock.patch("orion.metrics.liveness.has_registered_source", return_value=True), \
         mock.patch("orion.metrics.liveness.open_readonly_connection", return_value=fake_conn), \
         mock.patch("orion.metrics.liveness.liveness_for_node", return_value=fake_outcome), \
         mock.patch("orion.field.channel_glossary.CLEAN_VERDICTS", frozenset({"live", "quiet"})):
        rc = mod.main([])

    assert rc == 0  # did not crash, did not block


def test_main_fails_open_on_import_error():
    with mock.patch.object(mod, "_staged_files", return_value=["orion/x.py"]), \
         mock.patch.dict(sys.modules, {"orion.metrics.lineage": None}):
        rc = mod.main([])
    assert rc == 0


def test_main_respects_escape_hatch(monkeypatch):
    fake_node = mock.Mock()
    fake_node.name = "confidence"
    fake_outcome = mock.Mock(verdict="dead", detail="n=0 over 1h", sample_count=0)
    fake_graph = mock.Mock()
    fake_graph.nodes = {"metric://x#confidence": fake_node}

    monkeypatch.setenv("ORION_ALLOW_DEAD_METRIC_WIRE", "1")
    with mock.patch.object(mod, "_staged_files", return_value=["orion/x.py"]), \
         mock.patch.object(
             mod, "find_new_token_references",
             return_value={"confidence": [("orion/x.py", 12)]},
         ), \
         mock.patch("orion.metrics.lineage.build_graph", return_value=fake_graph), \
         mock.patch("orion.metrics.liveness.has_registered_source", return_value=True), \
         mock.patch("orion.metrics.liveness.open_readonly_connection", return_value=mock.Mock()), \
         mock.patch("orion.metrics.liveness.liveness_for_node", return_value=fake_outcome), \
         mock.patch("orion.field.channel_glossary.CLEAN_VERDICTS", frozenset({"live", "quiet"})):
        rc = mod.main([])

    assert rc == 0


def _run_git(args: list[str], cwd: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_staged_files_real_git_catches_a_rename_plus_modify():
    """Regression for the ACM -> ACMR fix (code review 2026-08-20): a file
    both renamed AND content-modified in the same commit has git status R,
    and --diff-filter=ACM alone silently excludes it. Every other test in
    this suite mocks subprocess.run entirely, which cannot catch a
    regression in the --diff-filter argument itself -- this one runs
    _staged_files() against a real temporary git repo instead, per CLAUDE.md
    section 11 ("every bug fix ships with a regression test that would have
    caught the bug")."""
    with tempfile.TemporaryDirectory() as tmp:
        _run_git(["init", "-q"], tmp)
        _run_git(["config", "user.email", "test@test.invalid"], tmp)
        _run_git(["config", "user.name", "test"], tmp)
        orig = Path(tmp) / "orig.py"
        orig.write_text("def read():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z\n")
        _run_git(["add", "orig.py"], tmp)
        _run_git(["commit", "-q", "-m", "initial", "--no-verify"], tmp)

        _run_git(["mv", "orig.py", "renamed.py"], tmp)
        renamed = Path(tmp) / "renamed.py"
        renamed.write_text(
            "def read():\n    x = 1\n    y = 2\n    z = 3\n    return fetch(\"l7_l11_ladder\")\n"
        )
        _run_git(["add", "renamed.py"], tmp)

        status = subprocess.run(
            ["git", "diff", "--cached", "--name-status"],
            cwd=tmp, check=True, capture_output=True, text=True,
        ).stdout
        assert status.startswith("R")  # sanity: this really is a rename, not add+delete

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            staged = mod._staged_files()
        finally:
            os.chdir(old_cwd)

        assert "renamed.py" in staged

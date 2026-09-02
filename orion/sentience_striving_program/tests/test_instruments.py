"""Tests for the sentience-program instrument manifest and its gate.

Two lanes, deliberately kept apart:

* Contract tests over the REAL instruments.yaml -- these are what stop the
  manifest rotting into a keyword cathedral (every instrument must name a module
  that exists, a real outcome, and an entrypoint that resolves).
* Behaviour tests over synthetic manifests, for the drift/staleness logic itself.

The synthetic lane deliberately does NOT reuse the real file: a gate proven only
against the fixture it was written alongside proves nothing, and this repo has
been bitten by exactly that (a synthetic fixture hiding a gate inversion).
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.sentience_striving_program.instruments import (
    Claim,
    build_state,
    check_repo_presence,
    evaluate_claim,
    load_manifest,
    resolve_retention_hours,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Contract lane -- runs against the real manifest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_real_manifest_parses(manifest):
    assert manifest["instruments"], "manifest declares no instruments"
    assert set(manifest["outcomes"]) == {"O1", "O2", "O3", "O4"}


def test_every_instrument_module_exists(manifest):
    """The manifest may not name code that is not there.

    This caught two stale paths while the manifest was being authored:
    novelty_for_target had moved to scoring.py, and capability_policy.py lives
    under orion/autonomy/, not orion/policy/.
    """
    missing = []
    for inst in manifest["instruments"]:
        exists, entry_ok = check_repo_presence(inst, REPO_ROOT)
        if not exists:
            missing.append(f"{inst.id}: {inst.module}")
        elif entry_ok is False:
            missing.append(f"{inst.id}: {inst.module}::{inst.entrypoint}")
    assert not missing, f"manifest names code that does not exist: {missing}"


def test_every_outcome_has_at_least_one_instrument(manifest):
    """An outcome nothing ladders to is an outcome nobody is working toward."""
    claimed = {i.outcome for i in manifest["instruments"]}
    assert claimed == set(manifest["outcomes"]), (
        f"outcomes with no instrument: {set(manifest['outcomes']) - claimed}"
    )


def test_sql_claims_are_read_only(manifest):
    """No claim may mutate. The board is an observer of the program, not an actor."""
    forbidden = ("insert", "update", "delete", "drop", "truncate", "alter", "create")
    for inst in manifest["instruments"]:
        for claim in inst.claims:
            if not claim.sql:
                continue
            lowered = claim.sql.lower()
            assert lowered.strip().startswith("select"), (
                f"{inst.id}/{claim.id}: SQL must start with SELECT"
            )
            for word in forbidden:
                assert f" {word} " not in f" {lowered} ", (
                    f"{inst.id}/{claim.id}: SQL contains {word!r}"
                )


def test_every_claim_records_a_date(manifest):
    """A recorded value without a date cannot be reasoned about later."""
    for inst in manifest["instruments"]:
        for claim in inst.claims:
            assert claim.recorded_at, f"{inst.id}/{claim.id} has no recorded_at"


def test_retention_resolves_for_instruments_that_declare_one(manifest):
    """A declared retention setting must actually resolve to a number.

    Regression guard: this returned None for every instrument in a fresh
    worktree, because `.env` is gitignored and the lookup had no fallback -- so
    the board rendered no ceiling at all, silently, which is precisely the fact
    it exists to surface.
    """
    for inst in manifest["instruments"]:
        if not inst.storage.retention_setting:
            continue
        hours, source = resolve_retention_hours(inst, REPO_ROOT)
        assert hours and hours > 0, (
            f"{inst.id}: {inst.storage.retention_setting} did not resolve ({source})"
        )


# ---------------------------------------------------------------------------
# Behaviour lane -- synthetic manifests
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, *, last_reviewed: str, recorded: int) -> Path:
    path = tmp_path / "m.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            version: 1
            outcomes:
              O1: {{title: "t", claim: "c"}}
            review_max_age_days: 90
            instruments:
              - id: probe
                title: "Probe"
                theory: "t"
                program_ref: "r"
                module: orion/sentience_striving_program/instruments.py
                entrypoint: load_manifest
                outcome: O1
                unlock: "u"
                last_reviewed: {last_reviewed}
                storage: {{kind: none}}
                claims:
                  - id: absence
                    question: "q"
                    kind: absent_from_repo
                    target: "zzz_string_that_appears_nowhere_in_this_repo"
                    recorded: {recorded}
                    recorded_at: 2026-09-02
            """
        ).strip()
    )
    return path


def test_holding_claim_reports_holds(tmp_path):
    m = load_manifest(_write_manifest(tmp_path, last_reviewed="2026-09-02", recorded=0))
    states = build_state(m, conn=None, root=REPO_ROOT, with_consumers=False)
    assert [c.status for c in states[0].claims] == ["HOLDS"]


def test_drifted_claim_is_detected(tmp_path):
    """The recorded value says 3 files mention it; reality says 0."""
    m = load_manifest(_write_manifest(tmp_path, last_reviewed="2026-09-02", recorded=3))
    states = build_state(m, conn=None, root=REPO_ROOT, with_consumers=False)
    assert states[0].claims[0].status == "DRIFTED"
    assert states[0].claims[0].drifted


def test_stale_review_is_flagged(tmp_path):
    old = (datetime.now(timezone.utc).date() - timedelta(days=400)).isoformat()
    m = load_manifest(_write_manifest(tmp_path, last_reviewed=old, recorded=0))
    states = build_state(m, conn=None, root=REPO_ROOT, with_consumers=False)
    assert states[0].review_stale
    assert states[0].review_age_days >= 400


def test_fresh_review_is_not_flagged(tmp_path):
    """The complement of the test above -- a window that always fires is not a gate."""
    recent = (datetime.now(timezone.utc).date() - timedelta(days=5)).isoformat()
    m = load_manifest(_write_manifest(tmp_path, last_reviewed=recent, recorded=0))
    states = build_state(m, conn=None, root=REPO_ROOT, with_consumers=False)
    assert not states[0].review_stale


def test_manual_claim_is_never_auto_passed(tmp_path):
    """A human-run check must report MANUAL, not HOLDS.

    Reporting it as HOLDS would let a check nobody has actually run read as
    green -- the failure mode this whole program keeps rediscovering.
    """
    claim = Claim(
        id="m", question="q", kind="manual", recorded="PASS", recorded_at="2026-09-02"
    )
    assert evaluate_claim(claim, conn=None, root=REPO_ROOT).status == "MANUAL"


def test_sql_claim_without_connection_errors_not_passes(tmp_path):
    """No database must mean ERROR, never a silent HOLDS."""
    claim = Claim(
        id="s",
        question="q",
        kind="sql",
        recorded=1,
        recorded_at="2026-09-02",
        sql="SELECT 1",
    )
    assert evaluate_claim(claim, conn=None, root=REPO_ROOT).status == "ERROR"


def test_unknown_outcome_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        textwrap.dedent(
            """
            version: 1
            outcomes:
              O1: {title: "t", claim: "c"}
            instruments:
              - id: probe
                title: "P"
                theory: "t"
                program_ref: "r"
                module: orion/sentience_striving_program/instruments.py
                outcome: O9
                unlock: "u"
                last_reviewed: 2026-09-02
                storage: {kind: none}
            """
        ).strip()
    )
    with pytest.raises(ValueError, match="unknown outcome"):
        load_manifest(path)


def test_duplicate_instrument_id_is_rejected(tmp_path):
    path = tmp_path / "dup.yaml"
    body = textwrap.dedent(
        """
              - id: probe
                title: "P"
                theory: "t"
                program_ref: "r"
                module: orion/sentience_striving_program/instruments.py
                outcome: O1
                unlock: "u"
                last_reviewed: 2026-09-02
                storage: {kind: none}
        """
    ).rstrip()
    path.write_text(
        "version: 1\noutcomes:\n  O1: {title: t, claim: c}\ninstruments:" + body + body
    )
    with pytest.raises(ValueError, match="duplicate instrument id"):
        load_manifest(path)


# ---------------------------------------------------------------------------
# Regression lane -- each of these reproduces a bug found in review
# ---------------------------------------------------------------------------


def test_repo_root_honours_orion_repo_root(monkeypatch, tmp_path):
    """The reducer must read the mounted repo, not the package's own parent.

    The Hub image copies only `orion/` and `services/orion-hub/` into `/app`
    (verified live: `/app/scripts` and `/app/services` do not exist inside
    `orion-athena-hub`), so a `__file__`-derived root rendered every instrument
    under `scripts/` as MISSING and made every retention ceiling unresolvable.
    """
    from orion.sentience_striving_program import instruments as mod

    monkeypatch.setenv("ORION_REPO_ROOT", str(tmp_path))
    assert mod._repo_root() == tmp_path
    # A bogus value must not break the CLI -- fall back, do not blow up.
    monkeypatch.setenv("ORION_REPO_ROOT", str(tmp_path / "nope"))
    assert mod._repo_root() == REPO_ROOT
    monkeypatch.delenv("ORION_REPO_ROOT")
    assert mod._repo_root() == REPO_ROOT


def test_derived_storage_is_labelled_as_an_input_not_output(tmp_path):
    """A shared upstream table's freshness is not the instrument's liveness.

    `substrate_attention_frames` is written every tick; the emergent-clustering
    probe has been run by hand twice, ever. Without this note the board renders
    "123,945 rows, writing now" for a probe that has not run in weeks.
    """
    from orion.sentience_striving_program.instruments import (
        Instrument,
        InstrumentState,
        Storage,
        _storage_state,
    )

    class _Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql):
            self.sql = sql

        def fetchone(self):
            return (7, None, None)

    class _Conn:
        def cursor(self):
            return _Cur()

    inst = Instrument(
        id="i", title="t", theory="t", program_ref="r", module="m", outcome="O1",
        unlock="u", last_reviewed="2026-09-02",
        storage=Storage(kind="derived", table="shared_upstream_table"),
        claims=(),
    )
    state = InstrumentState(instrument=inst)
    _storage_state(inst, _Conn(), state)
    assert state.row_count == 7
    assert "INPUT, not its output" in state.storage_note


def test_unparseable_env_value_falls_back_to_the_template(tmp_path):
    """One bad live value must not suppress the ceiling entirely.

    `SETTING=168  # 7 days` in `.env` used to return None and stop, so the board
    rendered no retention ceiling at all -- the single fact it exists to show.
    """
    from orion.sentience_striving_program.instruments import (
        Instrument,
        Storage,
        resolve_retention_hours,
    )

    svc = tmp_path / "services" / "svc"
    svc.mkdir(parents=True)
    (svc / ".env").write_text("RET_HOURS=168  # 7 days\n")
    (svc / ".env_example").write_text("RET_HOURS=72.0\n")
    inst = Instrument(
        id="i", title="t", theory="t", program_ref="r", module="m", outcome="O1",
        unlock="u", last_reviewed="2026-09-02",
        storage=Storage(kind="append_only", table="t",
                        retention_setting="RET_HOURS", retention_service="svc"),
        claims=(),
    )
    hours, source = resolve_retention_hours(inst, tmp_path)
    assert (hours, source) == (72.0, ".env_example")


def test_package_entrypoint_is_actually_checked(tmp_path):
    """A package module that declares a missing entrypoint must not pass.

    `check_repo_presence` returned (True, None) for any directory, so a manifest
    could name a function that does not exist and the contract test would still
    go green.
    """
    from orion.sentience_striving_program.instruments import Instrument, Storage

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "a.py").write_text("def really_here():\n    return 1\n")

    def _inst(entry):
        return Instrument(
            id="i", title="t", theory="t", program_ref="r", module="pkg",
            outcome="O1", unlock="u", last_reviewed="2026-09-02",
            storage=Storage(kind="none"), claims=(), entrypoint=entry,
        )

    assert check_repo_presence(_inst("really_here"), tmp_path) == (True, True)
    assert check_repo_presence(_inst("not_here_at_all"), tmp_path) == (True, False)


# ---------------------------------------------------------------------------
# Static lane -- the CI mode, which has no database
# ---------------------------------------------------------------------------


def test_static_only_skips_sql_claims_rather_than_passing_them(manifest):
    """SQL claims must report SKIPPED in the static lane -- not HOLDS, not ERROR.

    HOLDS would let a gate nobody can run read as green. ERROR would make the
    static lane permanently red and get it switched off. Both failure modes end
    with the gate not doing its job, so the third state is load-bearing.
    """
    states = build_state(manifest, conn=None, root=REPO_ROOT, with_consumers=False,
                         static_only=True)
    sql_claims = [
        c for s in states for c in s.claims if c.claim.kind == "sql"
    ]
    assert sql_claims, "expected at least one SQL claim in the real manifest"
    assert all(c.status == "SKIPPED" for c in sql_claims)
    assert not any(c.drifted for c in sql_claims)


def test_static_lane_still_checks_non_sql_claims(manifest):
    """The static lane must not degrade into checking nothing at all."""
    states = build_state(manifest, conn=None, root=REPO_ROOT, with_consumers=False,
                         static_only=True)
    checked = [
        c for s in states for c in s.claims if c.status in ("HOLDS", "DRIFTED")
    ]
    assert checked, "static lane checked no claims -- it would be a no-op gate"


def test_python_absence_fallback_matches_ripgrep():
    """The dependency-free fallback must agree with ripgrep, on a real hit.

    Asserted against a pattern that genuinely exists: comparing two empty
    results would pass even if the fallback were broken and always returned
    nothing.
    """
    import shutil
    import subprocess

    from orion.sentience_striving_program.instruments import _python_hit_paths

    if not shutil.which("rg"):
        pytest.skip("ripgrep not available to compare against")

    pattern = "def reduce_attention_self_model"
    fallback = set(_python_hit_paths(pattern, REPO_ROOT))
    assert fallback, "fallback found nothing for a pattern known to exist"

    proc = subprocess.run(
        ["rg", "--files-with-matches", "--fixed-strings", pattern, "."],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    expected = {ln.removeprefix("./") for ln in proc.stdout.splitlines() if ln.strip()}
    assert fallback == expected


# ---------------------------------------------------------------------------
# Absence scan -- the no-ripgrep path the Hub container actually runs
# ---------------------------------------------------------------------------


def test_absence_walk_prunes_heavy_directories(tmp_path):
    """The walk must not descend into .git / graphify-out and friends.

    Regression guard for a live 78-second page load: ripgrep is NOT installed in
    orion-athena-hub, so the Python fallback is the path that runs there, and an
    rglob-based version traversed 434,401 files including a 293MB .git and a
    640MB graphify-out. Pruning must happen DURING traversal, not as a filter
    afterwards.
    """
    from orion.sentience_striving_program.instruments import _python_hit_paths_multi

    needle = "zzz_absence_probe_symbol"
    (tmp_path / "real.py").write_text(f"def {needle}(): pass")
    for heavy in (".git", "graphify-out", "node_modules", "__pycache__"):
        d = tmp_path / heavy
        d.mkdir()
        (d / "buried.py").write_text(f"def {needle}(): pass")

    found = _python_hit_paths_multi([needle], tmp_path)[needle]
    assert found == ["real.py"], (
        f"walk descended into a pruned directory: {sorted(found)}"
    )


def test_absence_walk_serves_every_pattern_in_one_pass(tmp_path):
    """Batching is the property that keeps this off the page-load critical path."""
    from orion.sentience_striving_program.instruments import _python_hit_paths_multi

    (tmp_path / "a.py").write_text("def alpha_sym(): pass")
    (tmp_path / "b.py").write_text("def beta_sym(): pass")
    res = _python_hit_paths_multi(["alpha_sym", "beta_sym", "gamma_sym"], tmp_path)
    assert res["alpha_sym"] == ["a.py"]
    assert res["beta_sym"] == ["b.py"]
    assert res["gamma_sym"] == []  # absent target returns empty, not missing


def test_absence_counts_agree_with_and_without_ripgrep(tmp_path, monkeypatch):
    """Both backends must give the same answer, or the gate depends on the host.

    Asserted on a target that IS present as well as one that is absent: two
    zeroes would agree even if a backend were broken and always found nothing.
    """
    import subprocess as sp

    from orion.sentience_striving_program import instruments as mod

    (tmp_path / "present.py").write_text("def present_sym(): pass")

    with_rg = mod.absence_counts(["present_sym", "absent_sym"], tmp_path)

    real_run = sp.run

    def _no_rg(cmd, *a, **k):
        if cmd and cmd[0] == "rg":
            raise FileNotFoundError("rg")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(mod.subprocess, "run", _no_rg)
    without_rg = mod.absence_counts(["present_sym", "absent_sym"], tmp_path)

    assert with_rg == without_rg == {"present_sym": 1, "absent_sym": 0}

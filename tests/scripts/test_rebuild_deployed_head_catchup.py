"""#3704: a pull whose post-merge rebuild never ran must not lose its services.

2026-11-18 02:28Z: a `git pull --ff-only` on the primary checkout was interrupted
3 s after its fast-forward, before the post-merge rebuild wrote a RUN line. The
next pull's rebuild diffed only ORIG_HEAD..HEAD (its own commits), so the first
pull's services (orion-prediction, orion-mind) kept old containers.

These tests drive the real rebuild_services_from_git_diff.sh against a throwaway
git repo with a bare "origin". The planner is a fake that answers --plan from a
real `git diff --name-only <base> HEAD` (same default base order as the real
planner), and safe_docker_build.sh is a fake that only records its arguments:
no docker is touched.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rebuild_services_from_git_diff.sh"

_DIFF_PLANNER = """\
import os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_PLANNER_ARGV"], "a") as fh:
    fh.write(" ".join(args) + "\\n")
if any(a in args for a in ("--running-profile-services", "--running-extra-compose",
                            "--pool-held-services", "--migrations")):
    raise SystemExit(0)
def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True)
base = args[args.index("--base") + 1] if "--base" in args else None
if base is None:
    for cand in ("ORIG_HEAD", "MERGE_HEAD", "HEAD@{1}", "HEAD~1"):
        if git("rev-parse", "--verify", "-q", cand).returncode == 0:
            base = cand
            break
names = git("diff", "--name-only", base, "HEAD").stdout.split()
seen = []
for n in names:
    parts = n.split("/")
    if len(parts) > 2 and parts[0] == "services" and parts[1] not in seen:
        seen.append(parts[1])
for s in sorted(seen):
    print(s + "\\tbuild")
"""

SERVICES = ("svc-a", "svc-b", "svc-c")


def _git(*cmd: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", *cmd],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


class Host:
    """A primary checkout ('repo') whose origin gets commits from 'other'."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.repo = tmp_path / "repo"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(SCRIPT, scripts / SCRIPT.name)
        (scripts / "rebuild_affected_services.py").write_text(_DIFF_PLANNER)
        self.capture = tmp_path / "safe-build-args.txt"
        safe = scripts / "safe_docker_build.sh"
        safe.write_text(
            "#!/bin/sh\n"
            'printf \'%s\\n\' "$*" >> "$REBUILD_TEST_CAPTURE"\n'
            'for f in $FAKE_FAIL_SVC; do [ "$1" = "$f" ] && exit 1; done\n'
            "exit 0\n"
        )
        safe.chmod(0o755)
        for svc in SERVICES:
            (self.repo / "services" / svc).mkdir(parents=True)
            (self.repo / "services" / svc / "app.py").write_text("v = 0\n")
        _git("init", "-q", "-b", "main", cwd=self.repo)
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-qm", "init", cwd=self.repo)
        remote = tmp_path / "remote.git"
        _git("init", "--bare", "-q", str(remote), cwd=tmp_path)
        _git("remote", "add", "origin", str(remote), cwd=self.repo)
        _git("push", "-qu", "origin", "main", cwd=self.repo)
        self.other = tmp_path / "other"
        _git("clone", "-q", "-b", "main", str(remote), str(self.other), cwd=tmp_path)
        self.log = tmp_path / "orion" / "post_pull_rebuild.log"
        self.state = tmp_path / "orion" / "post_pull_deployed_head"
        self.planner_argv = tmp_path / "planner-argv.txt"
        self._n = 0

    def merge_upstream(self, svc: str) -> str:
        """A PR touching services/<svc> lands on origin/main."""
        self._n += 1
        path = self.other / "services" / svc / "app.py"
        path.write_text(f"v = {self._n}\n")
        _git("commit", "-qam", f"change {svc}", cwd=self.other)
        _git("push", "-q", "origin", "main", cwd=self.other)
        return _git("rev-parse", "HEAD", cwd=self.other)

    def pull(self) -> str:
        """`git pull --ff-only` in the primary checkout (sets ORIG_HEAD)."""
        _git("pull", "-q", "--ff-only", cwd=self.repo)
        return _git("rev-parse", "HEAD", cwd=self.repo)

    def head(self) -> str:
        return _git("rev-parse", "HEAD", cwd=self.repo)

    def state_lines(self) -> list[str]:
        return self.state.read_text().splitlines()

    def rebuild(self, *args: str, fail: str = "", env_extra: dict[str, str] | None = None,
                repo: Path | None = None):
        if self.capture.exists():
            self.capture.unlink()
        env = os.environ.copy()
        for key in ("ORION_DEPLOY_PINNED_HEAD", "ORION_POST_MERGE_REBUILD", "ORION_REBUILD_STATE",
                    "ORION_REBUILD_LOCK", "ORION_REBUILD_LOCK_WAIT_S"):
            env.pop(key, None)
        env.update({
            "REBUILD_TEST_CAPTURE": str(self.capture),
            "FAKE_PLANNER_ARGV": str(self.planner_argv),
            "FAKE_FAIL_SVC": fail or "__none__",
            "ORION_REBUILD_LOG": str(self.log),
            "ORION_POST_PULL_SANDBOX_SYNC": "0",
            "ORION_POST_PULL_CONSULT_EXPORT": "0",
            "ORION_POST_PULL_APPLY_MIGRATIONS": "0",
        })
        env.update(env_extra or {})
        done = subprocess.run(
            [str((repo or self.repo) / "scripts" / SCRIPT.name), *args],
            cwd=repo or self.repo, env=env, text=True, capture_output=True, check=False,
        )
        calls = self.capture.read_text().splitlines() if self.capture.exists() else []
        rebuilt = sorted({c.split()[0] for c in calls})
        return done, rebuilt


def test_skipped_pull_services_are_rebuilt_by_the_next_pull(tmp_path: Path) -> None:
    """The 11-18 incident: pull 1 ran no rebuild; pull 2 must rebuild the union."""
    h = Host(tmp_path)
    h.merge_upstream("svc-c")
    deployed = h.pull()
    done, rebuilt = h.rebuild()
    assert done.returncode == 0, done.stdout + done.stderr
    assert rebuilt == ["svc-c"]
    assert h.state_lines()[0] == deployed

    h.merge_upstream("svc-a")
    h.pull()  # interrupted: the post-merge rebuild never ran

    h.merge_upstream("svc-b")
    head = h.pull()
    done, rebuilt = h.rebuild()
    assert done.returncode == 0, done.stdout + done.stderr
    assert rebuilt == ["svc-a", "svc-b"]
    log = h.log.read_text()
    assert f"CATCHUP from={deployed[:9]}" in log
    assert f"RUN base=deployed:{deployed[:9]} services=svc-a svc-b " in log
    # The migration step widens with the plan: same base.
    assert f"--migrations --base {deployed}" in h.planner_argv.read_text()
    assert h.state_lines()[0] == head


def test_first_run_without_state_file_behaves_as_before(tmp_path: Path) -> None:
    """No state file: only ORIG_HEAD..HEAD, exactly today's behavior."""
    h = Host(tmp_path)
    h.merge_upstream("svc-a")
    h.pull()
    h.merge_upstream("svc-b")
    head = h.pull()
    done, rebuilt = h.rebuild()
    assert done.returncode == 0, done.stdout + done.stderr
    assert rebuilt == ["svc-b"]
    assert "CATCHUP" not in h.log.read_text()
    assert "RUN base=default services=svc-b " in h.log.read_text()
    assert h.state_lines()[0] == head


def test_state_at_default_base_changes_nothing(tmp_path: Path) -> None:
    """Every pull rebuilt: the state equals ORIG_HEAD and the plan is unchanged."""
    h = Host(tmp_path)
    for svc in ("svc-a", "svc-b"):
        h.merge_upstream(svc)
        h.pull()
        done, rebuilt = h.rebuild()
        assert done.returncode == 0, done.stdout + done.stderr
        assert rebuilt == [svc]
    assert "CATCHUP" not in h.log.read_text()


def test_failed_service_is_retried_without_replanning_the_rest(tmp_path: Path) -> None:
    """A failed stack is retried by the next pull (05:10Z refusals); the ones
    that deployed fine are not rebuilt again."""
    h = Host(tmp_path)
    h.merge_upstream("svc-c")
    h.pull()
    assert h.rebuild()[0].returncode == 0

    h.merge_upstream("svc-a")
    failed_head = h.pull()
    done, rebuilt = h.rebuild(fail="svc-a")
    assert done.returncode == 1
    assert h.state_lines() == [failed_head, "pending=svc-a"]

    h.merge_upstream("svc-b")
    head = h.pull()
    done, rebuilt = h.rebuild()
    assert done.returncode == 0, done.stdout + done.stderr
    assert rebuilt == ["svc-a", "svc-b"]
    assert "CATCHUP" not in h.log.read_text()
    assert "PENDING_RETRY services=svc-a" in h.log.read_text()
    assert h.state_lines() == [head, "pending="]

    # Nothing pending any more: the next pull plans only its own change.
    h.merge_upstream("svc-c")
    h.pull()
    assert h.rebuild()[1] == ["svc-c"]


def test_services_run_clears_what_it_deployed_from_pending(tmp_path: Path) -> None:
    h = Host(tmp_path)
    h.merge_upstream("svc-a")
    h.merge_upstream("svc-b")
    head = h.pull()
    done, _ = h.rebuild(fail="svc-a svc-b")
    assert done.returncode == 1
    assert h.state_lines() == [head, "pending=svc-a svc-b"]
    done, rebuilt = h.rebuild("--services", "svc-a")
    assert done.returncode == 0, done.stdout + done.stderr
    assert rebuilt == ["svc-a"]
    assert h.state_lines() == [head, "pending=svc-b"]


def test_nothing_to_rebuild_still_records_the_head(tmp_path: Path) -> None:
    h = Host(tmp_path)
    (h.other / "README").write_text("docs only\n")
    _git("add", "README", cwd=h.other)
    _git("commit", "-qm", "docs", cwd=h.other)
    _git("push", "-q", "origin", "main", cwd=h.other)
    head = h.pull()
    done, rebuilt = h.rebuild()
    assert done.returncode == 0, done.stdout + done.stderr
    assert rebuilt == []
    assert "NOTHING base=default" in h.log.read_text()
    assert h.state_lines() == [head, "pending="]


def test_linked_worktree_run_never_records(tmp_path: Path) -> None:
    """Only the primary checkout's runs describe what is deployed."""
    h = Host(tmp_path)
    h.merge_upstream("svc-a")
    h.pull()
    wt = tmp_path / "wt"
    _git("worktree", "add", "-q", "--detach", str(wt), "HEAD", cwd=h.repo)
    done, _ = h.rebuild(repo=wt)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not h.state.exists()


def test_dry_run_list_only_services_and_base_never_record(tmp_path: Path) -> None:
    h = Host(tmp_path)
    h.merge_upstream("svc-a")
    h.pull()
    for args in (("--dry-run",), ("--list-only",), ("--services", "svc-a"), ("--base", "HEAD~1")):
        done, _ = h.rebuild(*args)
        assert done.returncode == 0, (args, done.stdout + done.stderr)
        assert not h.state.exists(), args


def test_unpinned_run_on_stale_main_never_records(tmp_path: Path) -> None:
    h = Host(tmp_path)
    h.merge_upstream("svc-a")
    h.pull()
    h.merge_upstream("svc-b")  # origin moved; this checkout did not pull
    done, _ = h.rebuild()
    assert done.returncode == 0, done.stdout + done.stderr
    assert not h.state.exists()


def test_state_never_moves_backwards(tmp_path: Path) -> None:
    """Two overlapping runs: the one that finishes last must not rewind the state."""
    h = Host(tmp_path)
    h.merge_upstream("svc-a")
    h.pull()
    # A commit descending from this run's head, as a later run would have recorded.
    newer = _git("commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "later", cwd=h.repo)
    h.state.parent.mkdir(parents=True, exist_ok=True)
    h.state.write_text(newer + "\n")
    done, rebuilt = h.rebuild()
    assert done.returncode == 0, done.stdout + done.stderr
    assert rebuilt == ["svc-a"]  # not an ancestor of ORIG_HEAD: default base
    assert h.state_lines()[0] == newer


def test_garbage_state_file_falls_back_to_default_base(tmp_path: Path) -> None:
    h = Host(tmp_path)
    h.merge_upstream("svc-a")
    h.pull()
    h.merge_upstream("svc-b")
    head = h.pull()
    h.state.parent.mkdir(parents=True, exist_ok=True)
    h.state.write_text("not-a-commit\n")
    done, rebuilt = h.rebuild()
    assert done.returncode == 0, done.stdout + done.stderr
    assert rebuilt == ["svc-b"]
    assert h.state_lines()[0] == head


def test_second_run_waits_for_the_lock_then_proceeds(tmp_path: Path) -> None:
    """A run holding the lock makes the next one wait; a timeout still runs."""
    h = Host(tmp_path)
    h.merge_upstream("svc-a")
    h.pull()
    lock = tmp_path / "orion" / "post_pull_rebuild.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen(["flock", str(lock), "sleep", "30"])
    try:
        # Wait until the holder has the lock.
        for _ in range(100):
            if subprocess.run(["flock", "-n", str(lock), "true"]).returncode != 0:
                break
            subprocess.run(["sleep", "0.05"])
        done, rebuilt = h.rebuild(env_extra={"ORION_REBUILD_LOCK_WAIT_S": "1"})
    finally:
        holder.kill()
        holder.wait()
    assert done.returncode == 0, done.stdout + done.stderr
    assert rebuilt == ["svc-a"]
    log = h.log.read_text()
    assert "LOCK_WAIT" in log and "LOCK_TIMEOUT wait_s=1" in log
    assert log.index("LOCK_TIMEOUT") < log.index("RUN base=")


def test_waiting_run_pins_the_head_it_finds_after_the_wait(tmp_path: Path) -> None:
    """The holder releases; the waiting run pins and records the current head."""
    h = Host(tmp_path)
    h.merge_upstream("svc-a")
    h.pull()
    lock = tmp_path / "orion" / "post_pull_rebuild.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen(["flock", str(lock), "sleep", "2"])
    try:
        for _ in range(100):
            if subprocess.run(["flock", "-n", str(lock), "true"]).returncode != 0:
                break
            subprocess.run(["sleep", "0.05"])
        done, rebuilt = h.rebuild(env_extra={"ORION_REBUILD_LOCK_WAIT_S": "30"})
    finally:
        holder.kill()
        holder.wait()
    assert done.returncode == 0, done.stdout + done.stderr
    log = h.log.read_text()
    assert "LOCK_WAIT" in log and "LOCK_ACQUIRED" in log and "LOCK_TIMEOUT" not in log
    assert h.state_lines()[0] == h.head()

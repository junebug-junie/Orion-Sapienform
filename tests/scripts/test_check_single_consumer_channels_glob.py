"""Regression coverage for check_single_consumer_channels.py's glob-pattern
resolution -- added alongside PR #1860's new "orion:exec:request:
VisionHostService:*" catalog entry, which a code review caught this gate
was silently skipping entirely (glob-shaped channel names were excluded from
every check with no resolution step). See that script's module docstring
and load_single_consumer_glob_patterns's docstring for the live incident
this closes coverage for.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_single_consumer_channels as gate  # noqa: E402


_FIXTURE_YAML = """
channels:
  - name: "orion:exec:request:LLMGatewayService"
    single_consumer: true
  - name: "orion:exec:request:VisionHostService:*"
    single_consumer: true
  - name: "orion:vision:reply:*"
    single_consumer: false
  - name: "orion:some:non_single_consumer:channel"
    single_consumer: false
"""


def test_load_single_consumer_channels_still_excludes_globs(tmp_path) -> None:
    f = tmp_path / "channels.yaml"
    f.write_text(_FIXTURE_YAML)
    names = gate.load_single_consumer_channels(f)
    assert names == ["orion:exec:request:LLMGatewayService"]


def test_load_single_consumer_glob_patterns_finds_only_single_consumer_globs(tmp_path) -> None:
    """orion:vision:reply:* is glob-shaped but single_consumer: false -- must
    NOT show up here (it's not a duplicate-execution risk, it's a fan-out
    reply channel by design)."""
    f = tmp_path / "channels.yaml"
    f.write_text(_FIXTURE_YAML)
    patterns = gate.load_single_consumer_glob_patterns(f)
    assert patterns == ["orion:exec:request:VisionHostService:*"]


class _FakeCompletedProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_resolve_glob_channels_parses_pubsub_channels_output() -> None:
    fake = _FakeCompletedProcess(stdout='"orion:exec:request:VisionHostService:circe-vl"\n')
    with patch("subprocess.run", return_value=fake):
        resolved = gate.resolve_glob_channels("redis://fake:6379/0", "orion:exec:request:VisionHostService:*")
    assert resolved == ["orion:exec:request:VisionHostService:circe-vl"]


def test_resolve_glob_channels_returns_empty_list_when_nothing_matches() -> None:
    """No realized channel yet (e.g. a foveal host not deployed) is not an
    error -- just nothing to check."""
    fake = _FakeCompletedProcess(stdout="")
    with patch("subprocess.run", return_value=fake):
        resolved = gate.resolve_glob_channels("redis://fake:6379/0", "orion:exec:request:VisionHostService:*")
    assert resolved == []


def test_resolve_glob_channels_raises_on_connection_failure() -> None:
    fake = _FakeCompletedProcess(stdout="", stderr="Could not connect to Redis")
    with patch("subprocess.run", return_value=fake):
        try:
            gate.resolve_glob_channels("redis://fake:6379/0", "orion:exec:request:VisionHostService:*")
        except RuntimeError as exc:
            assert "could not connect" in str(exc).lower()
        else:
            raise AssertionError("expected RuntimeError")


def test_main_folds_resolved_glob_channels_into_the_checked_set(tmp_path) -> None:
    """End-to-end: a glob pattern that resolves to one live, correctly
    single-subscribed channel must produce an OK gate result, not be
    silently dropped -- this is the exact behavior this PR's own
    orion:exec:request:VisionHostService:circe-vl channel depends on."""
    f = tmp_path / "channels.yaml"
    f.write_text(_FIXTURE_YAML)

    def _fake_run(cmd, **kwargs):
        if cmd[3:5] == ["PUBSUB", "CHANNELS"]:
            return _FakeCompletedProcess(stdout='"orion:exec:request:VisionHostService:circe-vl"\n')
        if cmd[3:5] == ["PUBSUB", "NUMSUB"]:
            lines = []
            for ch in cmd[5:]:
                lines.append(f'"{ch}"')
                lines.append("1")
            return _FakeCompletedProcess(stdout="\n".join(lines) + "\n")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run):
        exit_code = gate.main(["--bus-url", "redis://fake:6379/0", "--channels-file", str(f)])
    assert exit_code == 0


def test_main_still_fails_on_a_glob_resolved_channel_with_two_subscribers(tmp_path) -> None:
    """The whole point: a duplicate consumer on a glob-registered channel
    must fail the gate, exactly like a literal one does."""
    f = tmp_path / "channels.yaml"
    f.write_text(_FIXTURE_YAML)

    def _fake_run(cmd, **kwargs):
        if cmd[3:5] == ["PUBSUB", "CHANNELS"]:
            return _FakeCompletedProcess(stdout='"orion:exec:request:VisionHostService:circe-vl"\n')
        if cmd[3:5] == ["PUBSUB", "NUMSUB"]:
            lines = []
            for ch in cmd[5:]:
                lines.append(f'"{ch}"')
                # The glob-resolved channel has 2 subscribers (violation);
                # any other requested channel is fine at 1.
                lines.append("2" if ch == "orion:exec:request:VisionHostService:circe-vl" else "1")
            return _FakeCompletedProcess(stdout="\n".join(lines) + "\n")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run):
        exit_code = gate.main(["--bus-url", "redis://fake:6379/0", "--channels-file", str(f)])
    assert exit_code == 1

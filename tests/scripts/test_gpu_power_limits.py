"""scripts/ops/gpu_power_limits.py: holds listed cards at their target, refuses what it must not
act on, steps aside for the power bench, and rolls back. A fake nvidia-smi on PATH plays the cards."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PATH = _REPO / "scripts" / "ops" / "gpu_power_limits.py"
_spec = importlib.util.spec_from_file_location("gpu_power_limits", _PATH)
gpl = importlib.util.module_from_spec(_spec)
sys.modules["gpu_power_limits"] = gpl
_spec.loader.exec_module(gpl)

FAKE_SMI = r'''#!/usr/bin/env python3
import json, sys
state_path, calls_path = sys.argv[0] + ".state", sys.argv[0] + ".calls"
cards = json.load(open(state_path))
args = sys.argv[1:]
open(calls_path, "a").write(" ".join(args) + "\n")
gpu = args[args.index("-i") + 1]
c = cards[gpu]
if "-pl" in args:
    c["cur"] = float(args[args.index("-pl") + 1])
    json.dump(cards, open(state_path, "w"))
    print(f"Power limit for GPU {gpu} was set to {c['cur']:.2f} W")
else:
    print(f"{c['cur']:.2f}, {c['default']:.2f}, {c['min']:.2f}, {c['max']:.2f}")
'''


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """Cards 0..3 as on circe; config lists only card 2 at 150 W unless a test rewrites it."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    cards = {str(i): {"cur": 250.0, "default": 250.0, "min": 100.0, "max": 250.0} for i in range(3)}
    cards["3"] = {"cur": 300.0, "default": 300.0, "min": 150.0, "max": 300.0}
    Path(f"{smi}.state").write_text(json.dumps(cards))
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    monkeypatch.setattr(socket, "gethostname", lambda: "circe")
    monkeypatch.setattr(gpl, "LOCK_DIR", tmp_path)
    monkeypatch.delenv("ORION_GPU_POWER_LIMITS_CHECK", raising=False)
    config = tmp_path / "gpu_power_limits.json"
    monkeypatch.setattr(gpl, "CONFIG", config)

    class Rig:
        def write(self, cards_cfg: dict, host: str = "circe") -> None:
            config.write_text(json.dumps({"host": host, "cards": cards_cfg}))

        def limit(self, gpu: int) -> float:
            return json.loads(Path(f"{smi}.state").read_text())[str(gpu)]["cur"]

        def set(self, gpu: int, watts: float) -> None:
            state = json.loads(Path(f"{smi}.state").read_text())
            state[str(gpu)]["cur"] = watts
            Path(f"{smi}.state").write_text(json.dumps(state))

        def sets(self) -> list[str]:
            calls = Path(f"{smi}.calls")
            return [c for c in calls.read_text().splitlines() if "-pl" in c] if calls.exists() else []

    r = Rig()
    r.write({"2": {"watts": 150}})
    return r


def _main(*argv: str) -> int:
    return gpl.main(list(argv))


def test_the_checked_in_config_is_valid_and_only_caps_gpu2(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "circe")
    assert gpl.load_config(_REPO / "config" / "gpu_power_limits.json") == {2: 150}


def test_apply_sets_the_target_and_logs_it(rig, capsys):
    assert _main("apply") == 0
    assert rig.limit(2) == 150.0
    assert "gpu2: power limit 250 -> 150 W (apply)" in capsys.readouterr().out


def test_cards_not_listed_are_never_touched(rig):
    assert _main("apply") == 0
    assert [rig.limit(i) for i in (0, 1, 3)] == [250.0, 250.0, 300.0]
    assert rig.sets() == ["-i 2 -pl 150"]


def test_a_card_already_at_target_is_left_alone(rig, capsys):
    rig.set(2, 150.0)
    assert _main("apply") == 0
    assert rig.sets() == []
    assert capsys.readouterr().out == ""


def test_timer_check_puts_back_a_drifted_card_and_says_drift(rig, capsys, monkeypatch):
    rig.set(2, 150.0)
    rig.set(2, 250.0)  # someone typed nvidia-smi -pl 250, or the driver reloaded
    monkeypatch.setenv("ORION_GPU_POWER_LIMITS_CHECK", "1")
    assert _main("apply") == 0
    assert rig.limit(2) == 150.0
    assert "250 -> 150 W (drift)" in capsys.readouterr().out


def test_target_outside_the_card_range_is_refused(rig, capsys):
    rig.write({"2": {"watts": 90}})
    assert _main("apply") == 1
    assert rig.limit(2) == 250.0
    assert "outside this card's range 100-250 W; refused" in capsys.readouterr().out


def test_config_for_another_host_touches_nothing(rig, capsys):
    rig.write({"2": {"watts": 150}}, host="athena")
    assert _main("apply") == 1
    assert rig.sets() == []
    assert "is for host 'athena', this is 'circe'" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ['{"host": "circe", "cards": {"2": {"watts": "lots"}}}', "not json"])
def test_broken_config_touches_nothing(rig, bad):
    gpl.CONFIG.write_text(bad)
    assert _main("apply") == 1
    assert rig.sets() == []


def test_a_running_bench_is_left_alone(rig, capsys):
    held = open(gpl.LOCK_DIR / "gpu-power-bench-2.lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert _main("apply") == 0
    finally:
        held.close()
    assert rig.limit(2) == 250.0
    assert "holds the card, skipped" in capsys.readouterr().out


def test_remove_puts_listed_cards_back_at_their_default(rig):
    assert _main("apply") == 0
    assert _main("remove") == 0
    assert rig.limit(2) == 250.0
    assert [rig.limit(i) for i in (0, 1, 3)] == [250.0, 250.0, 300.0]


def test_status_reads_and_writes_nothing(rig, capsys):
    assert _main("status") == 0
    row = json.loads(capsys.readouterr().out)
    assert row == {"gpu": 2, "target_w": 150, "current_w": 250.0, "default_w": 250.0, "at_target": False}
    assert rig.sets() == []


def test_install_copies_script_and_units_then_enables_both(monkeypatch, capsys):
    monkeypatch.setenv("DRY_RUN", "1")
    assert _main("install") == 0
    out = capsys.readouterr().out
    assert f"/scripts/ops/gpu_power_limits.py {gpl.SBIN}" in out
    for unit in gpl.UNITS:
        assert f"/etc/systemd/system/{unit}" in out
        assert (_REPO / "scripts" / "ops" / unit).exists()
    assert "systemctl enable orion-gpu-power-limits.service orion-gpu-power-limits.timer" in out


def test_units_wire_apply_remove_and_the_check(monkeypatch):
    unit = (_REPO / "scripts" / "ops" / "orion-gpu-power-limits.service").read_text()
    assert f"ExecStart={gpl.SBIN} apply" in unit and f"ExecStop={gpl.SBIN} remove" in unit
    check = (_REPO / "scripts" / "ops" / "orion-gpu-power-limits-check.service").read_text()
    assert "ExecStop=" not in check and "ORION_GPU_POWER_LIMITS_CHECK=1" in check
    timer = (_REPO / "scripts" / "ops" / "orion-gpu-power-limits.timer").read_text()
    assert "OnUnitActiveSec=5min" in timer and "Unit=orion-gpu-power-limits-check.service" in timer


def test_unknown_verb_is_usage():
    assert _main("frobnicate") == 2


def test_remove_waits_for_a_held_card_and_fails_loudly_instead_of_skipping(rig, capsys, monkeypatch):
    # Review finding: a check (or bench) holding the lock made `remove` skip and report success,
    # leaving the card capped after a rollback.
    rig.set(2, 150.0)
    monkeypatch.setattr(gpl, "REMOVE_WAIT_S", 0.2)
    held = open(gpl.LOCK_DIR / "gpu-power-bench-2.lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert _main("remove") == 1
    finally:
        held.close()
    assert rig.limit(2) == 150.0
    assert "NOT reset to default" in capsys.readouterr().out
    assert _main("remove") == 0  # once the holder lets go, the reset goes through
    assert rig.limit(2) == 250.0


def test_uninstall_resets_explicitly_before_deleting(monkeypatch, capsys):
    # Review finding: ExecStop never runs for a oneshot whose boot apply failed, so uninstall
    # deleted everything and left the card capped.
    monkeypatch.setenv("DRY_RUN", "1")
    assert _main("uninstall") == 0
    lines = capsys.readouterr().out.splitlines()
    timer = lines.index("systemctl disable --now orion-gpu-power-limits.timer")
    reset = lines.index(f"{gpl.SBIN} remove")
    gone = next(i for i, line in enumerate(lines) if line.startswith("rm -f"))
    assert timer < reset < gone


def test_uninstall_keeps_units_when_the_reset_fails(monkeypatch, capsys):
    calls = []

    def run(cmd):
        calls.append(cmd)
        if cmd[0] == gpl.SBIN:
            raise gpl.subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(gpl, "_run", run)
    assert _main("uninstall") == 1
    assert not any(c[0] == "rm" for c in calls)
    assert "NOT all reset" in capsys.readouterr().err


def test_timer_stops_with_the_service():
    # Review finding: stopping only the service was undone by the next 5-minute check.
    timer = (_REPO / "scripts" / "ops" / "orion-gpu-power-limits.timer").read_text()
    assert "PartOf=orion-gpu-power-limits.service" in timer


def test_default_sentinel_uncaps_a_card_through_the_config(rig):
    rig.set(2, 150.0)
    rig.write({"2": {"watts": "default"}})
    assert _main("apply") == 0
    assert rig.limit(2) == 250.0


@pytest.mark.parametrize("bad", ['["not", "an", "object"]',
                                 '{"host": "circe", "cards": {"2": {"watts": 150}, "02": {"watts": 100}}}',
                                 '{"host": "circe", "cards": {"2": {"watts": true}}}',
                                 '{"host": "circe", "cards": {"2": {"watts": 150.5}}}'])
def test_ambiguous_configs_are_refused(rig, bad):
    gpl.CONFIG.write_text(bad)
    assert _main("apply") == 1
    assert rig.sets() == []

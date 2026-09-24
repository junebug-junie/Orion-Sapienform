"""Pin the transport rule: waiting in line is not transport.

The pool's queue-wait hop must be excluded by equilibrium's transport baseline (a baseline that
"cannot learn busy" would read a queue as saturation), while the lease RPC itself, which answers
at once, stays a real transport hop."""
from __future__ import annotations

from pathlib import Path

from orion.gpu_pool.client import RPC_HEALTH_LABEL, WAIT_HOP_LABEL
from orion.metacog.transport_baseline import is_excluded

ROOT = Path(__file__).resolve().parents[3]
EQ = ROOT / "services" / "orion-equilibrium-service"


def _labels_in(path: Path, needle: str) -> list[str]:
    line = next(l for l in path.read_text().splitlines() if needle in l)
    value = line.split("=", 1)[1].split("}")[0].replace("${EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS:-", "")
    return [v.strip().strip('"') for v in value.split(",") if v.strip()]


def test_queue_wait_hop_is_excluded_everywhere_the_default_is_declared():
    hop = f"gpu_pool:metacog#{WAIT_HOP_LABEL}"
    for path in (EQ / ".env_example", EQ / "docker-compose.yml"):
        assert is_excluded(hop, _labels_in(path, "EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS=")), path
    settings_src = (EQ / "app" / "settings.py").read_text()
    assert f"log_orion_metacognition,{WAIT_HOP_LABEL}" in settings_src


def test_lease_rpc_hop_stays_transport():
    labels = _labels_in(EQ / ".env_example", "EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS=")
    assert not is_excluded(f"orion:gpu_pool:lease:request#{RPC_HEALTH_LABEL}", labels)

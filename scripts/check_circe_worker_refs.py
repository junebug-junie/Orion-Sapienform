#!/usr/bin/env python3
"""Gate: only the GPU pool's granted URL reaches a circe llama.cpp worker.

Why this exists (GPU pool stage 6.6, spec
docs/superpowers/specs/2026-09-30-gpu-pool-stage6-telemetry-reducers-lockdown.md, "Lockdown"):

circe's llama.cpp workers (chat 8011, metacog 8012, fast 8013, agent 8015, agent-burst 8016,
bonsai 8017, experiment 8099), the diffusion host (8014, pool-leased) and the lane-controller
actuator (8090) all listen on 0.0.0.0. Nothing stopped a service from hard-coding
`http://100.112.254.99:8011` and skipping the pool's lease queue entirely -- which is exactly how
background work used to land on Juniper's single-slot chat worker. Every LLM call is meant to go
through orion-llm-gateway, which asks orion-gpu-pool for a grant and dispatches to the URL the
pool hands back.

This is the deterministic half of the port gate (CLAUDE.md section 4). The runtime half is the
circe firewall in docs/runbooks/2026-10-01-circe-llm-port-firewall.md.

What it flags, one line at a time (comment-only lines are skipped):

    <circe address>:<worker port>        100.112.254.99:8011, circe:8015, circe.<tailnet>.ts.net:8090
    <worker container name>:<any port>   orion-atlas-llamacpp-chat:8080, ${PROJECT}-bonsai-worker:8080

The circe address, its name and the worker ports are read from config/gpu_pool.yaml plus the
*_HOST_PORT keys of the worker services' .env_example; worker names from the seat compose files'
service keys and container_name. A new seat is covered with no edit here.

What it scans: the whole repo (code, config, compose, templates, units, .env_example, Dockerfile,
Makefile) -- never a live `.env` unless --live-env is given (operator report). Excluded: hidden
directories, tests/, evals/, docs/, bench/, graphify-out/, venv/, node_modules/, *.test.js.
Addresses: circe's tailnet address and name (config/gpu_pool.yaml) and its LAN addresses.

ALLOW keys are per file + host + port, not per line: a second call to an already-allowed address in
the same file passes. Accepted -- every allowed file is small and reviewed with its entry.

Zones (whole paths that ARE the pool, its dispatch, its actuator or the workers themselves) are
not scanned. Each zone glob must still match a real file, so a renamed service shows up as stale.

Everything else must be named in ALLOW with a reason. An ALLOW entry that matches nothing fails
too (same rule as check_chat_route_poachers.py), so the list cannot rot.

Usage:
    python3 scripts/check_circe_worker_refs.py              # gate (exit 1 on fail)
    python3 scripts/check_circe_worker_refs.py --json
    python3 scripts/check_circe_worker_refs.py --live-env   # also report live services/*/.env (no gate)
    python3 scripts/check_circe_worker_refs.py --root DIR
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import pathlib
import re
import sys
from dataclasses import asdict, dataclass

REPO = pathlib.Path(__file__).resolve().parents[1]

RULE = (
    "a direct circe worker address skips the GPU pool's lease queue; call orion-llm-gateway "
    "(route name) instead, or add an ALLOW entry with a real reason"
)

# Paths that are the pool, the gateway's dispatch, the actuator, or the workers themselves.
# fnmatch globs against the repo-relative path; `*` crosses `/`.
ZONES: dict[str, str] = {
    "config/gpu_pool.yaml": "the pool's own seat map",
    "config/llm_profiles.yaml": "worker profiles the workers announce to the pool",
    "orion/gpu_pool/*": "pool library",
    "services/orion-gpu-pool/*": "the pool",
    "services/orion-llm-gateway/*": "the only dispatcher; it calls the URL the pool grants",
    "services/orion-gpu-lane-controller/*": "the actuator that starts/stops seats on circe",
    "services/orion-llamacpp-host/*": "the llama.cpp workers themselves",
    "services/orion-llamacpp-bonsai-host/*": "the bonsai llama.cpp worker itself",
    "services/orion-diffusion-host/*": "the diffusion worker itself",
    # Operator tooling for the port gate itself: it must name the addresses it guards.
    "scripts/check_circe_worker_refs.py": "this gate (its docstring and ALLOW keys name the addresses)",
    "scripts/ops/circe_llm_port_gate.sh": "the circe firewall for the same ports (runbook 2026-10-01)",
    "scripts/ops/orion-llm-port-gate.service": "its systemd unit",
}

# Hits outside the zones. Keys: "<repo-relative path>:<host>:<port>"; fnmatch globs allowed.
ALLOW: dict[str, str] = {
    # The diffusion host is a pool role (kind: service, port 8014). Since stage 5.4 the hold IS
    # the grant: orion-thought validates the durable-run hold ref before it calls this URL and
    # takes no second gate, so a direct URL here is pool-leased, not a bypass.
    "services/orion-thought/app/settings.py:100.112.254.99:8014": (
        "diffusion host, called only under a validated pool hold (stage 5.4)"
    ),
    "services/orion-thought/.env_example:100.112.254.99:8014": (
        "diffusion host, called only under a validated pool hold (stage 5.4)"
    ),
    # Docstring prose, not a call. Kept rather than reworded: both explain WHY the code goes
    # through the gateway instead of the worker.
    "services/orion-cortex-exec/app/executor.py:circe:801[15]": (
        "docstring describing which worker the chat route used to land on; no request is made"
    ),
    "services/orion-juniper-affective-state/app/vision_backend.py:circe:8011": (
        "docstring stating this service goes through the gateway, NOT straight at circe:8011"
    ),
}

_EXCLUDED_PARTS = {
    "tests", "evals", "docs", "graphify-out", "venv", ".venv", "node_modules", "__pycache__",
    ".worktrees", ".claude", "bench", ".git", ".github", ".pytest_cache", "site-packages",
}
_SUFFIXES = {
    ".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".sh", ".yml", ".yaml", ".toml", ".json",
    ".conf", ".ini", ".cfg", ".txt", ".html", ".j2", ".jinja", ".jinja2", ".service", ".env_example",
}
_NAMES = {".env_example", "Dockerfile", "Makefile"}
_MAX_FILE_BYTES = 4 * 1024 * 1024
# circe's LAN addresses (eno1 / enp179s0, `ip -br addr` on circe 2026-10-01). config/gpu_pool.yaml
# only names the tailnet address, but the worker ports listen on these too.
_LAN_ADDRESSES = ("192.168.1.22", "192.168.1.24")
# circe seat compose files: every compose service key and container_name in them is a worker name.
_WORKER_COMPOSE_FILES = (
    "services/orion-llamacpp-host/docker-compose.atlas-workers.yml",
    "services/orion-llamacpp-host/docker-compose.dsv41.yml",
    "services/orion-llamacpp-bonsai-host/docker-compose.yml",
    "services/orion-diffusion-host/docker-compose.yml",
    "services/orion-gpu-lane-controller/docker-compose.yml",
)

# Default worker identity if config/gpu_pool.yaml is unreadable (keeps the gate fail-closed).
_DEFAULT_ADDRESS = "100.112.254.99"
_DEFAULT_NAME = "circe"
_HOST_PORT_SOURCES = (
    "services/orion-llamacpp-host/.env_example",
    "services/orion-llamacpp-bonsai-host/.env_example",
    "services/orion-gpu-lane-controller/.env_example",
)
_HOST_PORT_RE = re.compile(r"^([A-Z0-9_]*HOST_PORT)=\s*['\"]?(\d{2,5})['\"]?\s*(?:#.*)?$")
# Fallback worker names if the compose files are unreadable.
_DEFAULT_CONTAINERS = (
    "atlas-chat", "atlas-metacog", "atlas-fast", "atlas-agent", "atlas-agent-burst", "atlas-llamacpp-chat",
    "atlas-llamacpp-metacog", "atlas-llamacpp-fast", "atlas-llamacpp-agent", "atlas-llamacpp-agent-burst",
    "dsv41-flash", "bonsai-worker", "diffusion-host", "gpu-lane-controller",
)


@dataclass(frozen=True)
class Identity:
    address: str
    name: str
    ports: frozenset[int]
    lan_addresses: tuple[str, ...] = _LAN_ADDRESSES
    # Worker container / compose service names. Matched with ANY port: on circe's docker network a
    # container reaches the worker on its internal 8080, which the firewall never sees (same bridge).
    containers: tuple[str, ...] = _DEFAULT_CONTAINERS


@dataclass(frozen=True)
class Hit:
    path: str
    lineno: int
    host: str
    port: int
    line: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.host}:{self.port}"


def load_identity(root: pathlib.Path) -> Identity:
    """circe's address/name and every worker host port, from the files that define them."""
    address, name = _DEFAULT_ADDRESS, _DEFAULT_NAME
    ports: set[int] = set()
    cfg_path = root / "config" / "gpu_pool.yaml"
    try:
        import yaml  # PyYAML; installed in the static-gates job

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        host = cfg.get("host") or {}
        address = str(host.get("address") or address)
        name = str(host.get("name") or name)
        cards = cfg.get("cards") or {}

        def off_circe(role: dict) -> bool:
            # Multi-host pool: a role whose card names another node (hecate's agent-deep) is not
            # a circe worker; its port is that node's, outside circe's firewall.
            return any(isinstance(cards.get(c), dict) and cards[c].get("host") not in (None, name)
                       for c in role.get("cards") or [])

        for role in (cfg.get("roles") or {}).values():
            if not isinstance(role, dict) or "port" not in role or off_circe(role):
                continue
            # llm seats are llama.cpp workers; a service role with a `launch` block is a
            # pool-leased GPU seat (diffusion). `world` is neither and is not a worker port.
            if role.get("kind") == "llm" or role.get("launch"):
                ports.add(int(role["port"]))
    except (OSError, ImportError, ValueError, TypeError):
        pass
    for rel in _HOST_PORT_SOURCES:
        path = root / rel
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            m = _HOST_PORT_RE.match(raw.strip())
            # orion-llamacpp-host's generic LLAMACPP_HOST_PORT (7005) is the athena-side
            # single-model server, not a circe seat; the atlas/dsv41 keys are.
            # HECATE_* keys are hecate's worker (docker-compose.hecate.yml), not a circe seat.
            if m and m.group(1) != "LLAMACPP_HOST_PORT" and not m.group(1).startswith("HECATE_"):
                ports.add(int(m.group(2)))
    if not ports:  # fail closed: never scan with an empty port set
        ports = {8011, 8012, 8013, 8014, 8015, 8016, 8017, 8090, 8099}
    return Identity(address=address, name=name, ports=frozenset(ports), containers=_load_containers(root))


def _load_containers(root: pathlib.Path) -> tuple[str, ...]:
    """Compose service keys + container_name (the literal part after any `${...}-` prefix)."""
    names: set[str] = set()
    try:
        import yaml

        for rel in _WORKER_COMPOSE_FILES:
            path = root / rel
            if not path.is_file():
                continue
            for key, svc in ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("services") or {}).items():
                names.add(str(key))
                cname = str((svc or {}).get("container_name") or "")
                literal = cname.rsplit("}", 1)[-1].lstrip("-")
                if literal:
                    names.add(literal)
    except (OSError, ImportError, ValueError, TypeError, AttributeError):
        pass
    # Too-generic names would flag unrelated text; every real seat name has a dash.
    names = {n for n in names if "-" in n and len(n) >= 8}
    return tuple(sorted(names | set(_DEFAULT_CONTAINERS), key=len, reverse=True))


def _address_re(ident: Identity) -> re.Pattern[str]:
    ports = "|".join(str(p) for p in sorted(ident.ports))
    addrs = "|".join(re.escape(a) for a in (ident.address, *ident.lan_addresses))
    host = rf"{addrs}|{re.escape(ident.name)}(?:\.[A-Za-z0-9-]+)*"
    return re.compile(rf"(?<![\w.-])({host}):({ports})(?!\d)")


def _container_re(ident: Identity) -> re.Pattern[str]:
    names = "|".join(re.escape(n) for n in ident.containers)
    # Any prefix of name characters / compose interpolation: `orion-atlas-llamacpp-chat`,
    # `${PROJECT:-orion}-bonsai-worker`, `orion-circe-diffusion-host`.
    return re.compile(rf"(?<![\w.])(?:[\w${{}}:.-]*-)?({names}):(\d{{2,5}})(?!\d)")


def _excluded(rel: pathlib.PurePath) -> bool:
    return any(part in _EXCLUDED_PARTS for part in rel.parts) or rel.name.endswith(".test.js")


def _zone_for(rel: str) -> str | None:
    for pattern in ZONES:
        if rel == pattern or fnmatch.fnmatchcase(rel, pattern):
            return pattern
    return None


def _walk(base: pathlib.Path):
    """os.walk with excluded directories pruned (node_modules etc. are never descended into)."""
    for dirpath, dirnames, filenames in os.walk(base):
        # Hidden directories (.git, .cursor, local smoke logs) are tooling state, not code.
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_PARTS and not d.startswith(".")]
        for name in filenames:
            yield pathlib.Path(dirpath) / name


def _scannable(path: pathlib.Path) -> bool:
    name = path.name
    if name == ".env" or name.startswith(".env.") and name != ".env_example":
        return False  # live env files: only with --live-env
    return path.suffix in _SUFFIXES or name in _NAMES or name.startswith(("docker-compose", "Dockerfile"))


def iter_scan_files(root: pathlib.Path) -> list[pathlib.Path]:
    """The whole repo minus excluded directories, zones and live .env files."""
    out: list[pathlib.Path] = []
    for path in _walk(root):
        if not _scannable(path):
            continue
        rel = path.relative_to(root)
        if _excluded(rel) or _zone_for(rel.as_posix()):
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        out.append(path)
    return sorted(out)


def _is_comment(stripped: str) -> bool:
    return stripped.startswith("#") or stripped.startswith("//")


def scan_text(text: str, rel: str, ident: Identity) -> list[Hit]:
    # Cheap prefilter: most files never mention circe or a worker at all.
    needles = (ident.address, ident.name, *ident.lan_addresses, *ident.containers)
    if not any(n in text for n in needles):
        return []
    addr_re = _address_re(ident)
    container_re = _container_re(ident)
    hits: list[Hit] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or _is_comment(stripped):
            continue
        for m in addr_re.finditer(raw):
            hits.append(Hit(rel, lineno, m.group(1), int(m.group(2)), stripped[:200]))
        for m in container_re.finditer(raw):
            hits.append(Hit(rel, lineno, m.group(1), int(m.group(2)), stripped[:200]))
    return hits


def scan_tree(root: pathlib.Path, ident: Identity | None = None) -> list[Hit]:
    ident = ident or load_identity(root)
    hits: list[Hit] = []
    for path in iter_scan_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits.extend(scan_text(text, path.relative_to(root).as_posix(), ident))
    return hits


def scan_live_env(root: pathlib.Path, ident: Identity | None = None) -> list[Hit]:
    """Operator report only: live services/*/.env files (gitignored, never in CI)."""
    ident = ident or load_identity(root)
    hits: list[Hit] = []
    for path in sorted(list(root.glob("services/*/.env")) + [root / ".env"]):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _zone_for(rel):
            continue
        for hit in scan_text(path.read_text(encoding="utf-8", errors="replace"), rel, ident):
            key = hit.line.split("=", 1)[0]
            hits.append(Hit(hit.path, hit.lineno, hit.host, hit.port, key))  # never print values
    return hits


def classify(
    hits: list[Hit], allow: dict[str, str]
) -> tuple[list[Hit], list[tuple[Hit, str]], list[str]]:
    """Split hits into (unallowed, allowed-with-reason, stale-allow-keys)."""
    unallowed: list[Hit] = []
    allowed: list[tuple[Hit, str]] = []
    used: set[str] = set()
    for hit in hits:
        reason = None
        for pattern, why in allow.items():
            if hit.key == pattern or fnmatch.fnmatchcase(hit.key, pattern):
                reason = why
                used.add(pattern)
                break
        if reason is None:
            unallowed.append(hit)
        else:
            allowed.append((hit, reason))
    return unallowed, allowed, [k for k in allow if k not in used]


def stale_zones(root: pathlib.Path, zones: dict[str, str]) -> list[str]:
    files = [p.relative_to(root).as_posix() for p in _walk(root)] if root.is_dir() else []
    return [z for z in zones if not any(f == z or fnmatch.fnmatchcase(f, z) for f in files)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=REPO)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--live-env", action="store_true",
        help="also list live .env keys that hold a direct worker address (report only, never fails)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    ident = load_identity(root)
    hits = scan_tree(root, ident)
    unallowed, allowed, stale = classify(hits, ALLOW)
    dead_zones = stale_zones(root, ZONES)
    ok = not unallowed and not stale and not dead_zones
    live = scan_live_env(root, ident) if args.live_env else []

    if args.json:
        print(json.dumps({
            "ok": ok,
            "identity": {"address": ident.address, "name": ident.name, "ports": sorted(ident.ports)},
            "unallowed": [asdict(h) for h in unallowed],
            "allowed": [{**asdict(h), "reason": r} for h, r in allowed],
            "stale_allow": stale,
            "stale_zones": dead_zones,
            "live_env": [asdict(h) for h in live],
        }, indent=2))
        return 0 if ok else 1

    ports = ",".join(str(p) for p in sorted(ident.ports))
    if unallowed:
        print("circe worker port gate: FAIL")
        print(f"\n  {RULE}\n")
        for hit in unallowed:
            print(f"    {hit.path}:{hit.lineno} -> {hit.host}:{hit.port}")
            print(f"        {hit.line}")
    if stale:
        print("circe worker port gate: STALE allow entries (match nothing; remove them):")
        for key in stale:
            print(f"    {key}")
    if dead_zones:
        print("circe worker port gate: STALE zones (match no file; fix or remove them):")
        for z in dead_zones:
            print(f"    {z}")
    if live:
        print(f"live .env keys holding a direct circe worker address ({len(live)}; report only):")
        for hit in live:
            print(f"    {hit.path}:{hit.lineno} {hit.line} -> {hit.host}:{hit.port}")
    if not ok:
        return 1
    print(
        f"circe worker port gate: PASS (circe={ident.name}/{ident.address}, ports {ports}; "
        f"{len(allowed)} allowed hit(s), {len(ALLOW)} allow entries, {len(ZONES)} zones)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

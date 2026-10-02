"""Bonsai fork image stays pinned, Volta-built, and off the shared llamacpp-host tag."""
import importlib
import json
from pathlib import Path

import yaml

HOST = Path(__file__).resolve().parents[1]
REPO = HOST.parents[1]
PROFILE = "ternary-bonsai2-27b-pq2-v100-32gb-circe-np4"


def _profile_cfg() -> dict:
    raw = yaml.safe_load((REPO / "config" / "llm_profiles.yaml").read_text(encoding="utf-8"))
    return raw["profiles"][PROFILE]


def test_dockerfile_pins_prism_fork_for_volta():
    text = (HOST / "Dockerfile").read_text(encoding="utf-8")
    assert "PrismML-Eng/llama.cpp" in text
    assert "88c4bc60b9c9578f134385be9535e853f2db9b9f" in text
    assert "CMAKE_CUDA_ARCHITECTURES=70" in text
    assert "GGML_CUDA_FA_ALL_QUANTS=ON" in text
    # Prism: CUDA 13.3 builds segfault; host nvcc 13.x dropped sm_70.
    assert "nvidia/cuda:12.8.1-devel-ubuntu24.04" in text
    assert "orion-llamacpp-host:0.1.0" in text
    # Base image supplies deps only; its baked wrapper/profiles can predate this profile.
    final = text.split("FROM ${HOST_IMAGE}", 1)[1]
    for copy in ("COPY services/orion-llamacpp-host/app /app/app", "COPY config /app/config", "COPY orion /app/orion"):
        assert copy in final
    # Build number = rev-list count; a shallow clone reports 1 and the wrapper
    # then drops --flash-attn off (main.py is_b5332_compatible).
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "--depth" not in code
    assert "-gt 5332" in text


def test_compose_stays_off_shared_image_and_pool_ports():
    compose = yaml.safe_load((HOST / "docker-compose.yml").read_text(encoding="utf-8"))
    svc = compose["services"]["bonsai-worker"]
    # Auto-rebuild runs `up -d --build`: without a build section it only restarts a stale image.
    assert svc["build"]["dockerfile"] == "services/orion-llamacpp-bonsai-host/Dockerfile"
    assert "llamacpp-bonsai-prism" in svc["image"]
    assert svc["restart"] == "no"
    env = "\n".join(svc["environment"])
    assert PROFILE in env
    # `experiment` is the DeepSeek soak's pool role; announcements are keyed by role.
    pool = yaml.safe_load((REPO / "config" / "gpu_pool.yaml").read_text(encoding="utf-8"))
    role = next(e.split("=", 1)[1] for e in svc["environment"] if e.startswith("LLM_ROLE="))
    assert role not in pool["roles"]
    # 8011/8015/8016 are pool lanes, 8099 is the DeepSeek soak.
    assert svc["ports"] == ["${BONSAI_HOST_PORT:-8017}:8080"]
    example = (HOST / ".env_example").read_text(encoding="utf-8")
    assert "BONSAI_LLAMACPP_IMAGE=llamacpp-bonsai-prism:server-local-volta" in example
    assert f"BONSAI_PROFILE_NAME={PROFILE}" in example


def test_shared_llamacpp_host_keeps_the_fork_off_every_lane_but_agent_gpu2():
    """Stage 7.2 moved the fork into orion-llamacpp-host as Dockerfile.prism for atlas-agent-burst
    (pool role agent-gpu2) only. The stock image, its compose and every other atlas worker stay off it."""
    shared = HOST.parent / "orion-llamacpp-host"
    for name in ("Dockerfile", "docker-compose.yml", ".env_example"):
        text = (shared / name).read_text(encoding="utf-8").lower()
        assert "bonsai" not in text and "prism" not in text, name
    workers = yaml.safe_load((shared / "docker-compose.atlas-workers.yml").read_text(encoding="utf-8"))["services"]
    for name, svc in workers.items():
        uses_fork = "prism" in (str(svc["build"]["dockerfile"]) + str(svc["image"])).lower()
        assert uses_fork == (name == "atlas-agent-burst"), name


def test_profile_launch_argv(monkeypatch):
    monkeypatch.setenv("LLM_PROFILE_NAME", PROFILE)
    monkeypatch.setenv("LLM_PROFILES_CONFIG_PATH", str(REPO / "config" / "llm_profiles.yaml"))
    main = importlib.import_module("app.main")
    profiles_mod = importlib.import_module("app.profiles")
    profile = profiles_mod.LLMProfile(name=PROFILE, **_profile_cfg())

    monkeypatch.setattr(main, "_ensure_model_file", lambda *_a, **_k: None)
    monkeypatch.setattr(
        main,
        "_get_supported_llama_server_flags",
        lambda _bin: {
            "--jinja",
            "--reasoning",
            "--reasoning-format",
            "--chat-template-kwargs",
            "--flash-attn",
            "--no-context-shift",
            "--n-predict",
        },
    )
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _bin: 9000)

    cmd, _env = main.build_llama_server_cmd_and_env(profile)
    flag = lambda f: cmd[cmd.index(f) + 1]  # noqa: E731

    assert cmd[cmd.index("-m") + 1].endswith("Ternary-Bonsai-2-27B-PQ2_0.gguf")
    # --ctx-size is split across --parallel: 65536 per concurrent run.
    assert int(flag("--ctx-size")) // int(flag("--parallel")) == 65536
    assert flag("--parallel") == "4"
    # Prism: --reasoning on overrides a client's reasoning_effort "none".
    assert flag("--reasoning") == "auto"
    kwargs = json.loads(flag("--chat-template-kwargs"))
    # Template accepts low/medium/xhigh; "high" returns HTTP 500.
    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["preserve_thinking"] is False
    assert flag("--n-predict") == "16384"
    # Measured win on circe gpu0 2026-09-30; see the field note.
    assert flag("--flash-attn") == "on"


def test_never_auto_deployed():
    """Manual only: it borrows a pool card (gpu1/gpu2) the lane controller does not know it holds,
    so a post-merge `up` would collide with pool launches."""
    common = REPO / "mesh-utilities" / "common"
    service = "orion-llamacpp-bonsai-host"
    for include in common.glob("include_services*"):
        files = include.rglob("*.txt") if include.is_dir() else [include]
        for f in files:
            lines = {l.strip() for l in f.read_text(encoding="utf-8").splitlines()}
            assert service not in lines, f
    excludes = (common / "exclude_services.txt").read_text(encoding="utf-8").splitlines()
    assert service in {l.strip() for l in excludes}


def _card_indices(pool: dict, role: str) -> set[int]:
    return {int(pool["cards"][c]["index"]) for c in pool["roles"][role]["cards"]}


def test_no_bonsai_config_targets_chats_card():
    """Juniper, 2026-09-30: Bonsai runs on gpu1/gpu2 (the agent cards), never chat's gpu0.

    Card indices come from config/gpu_pool.yaml, so a card move there moves this gate too.
    """
    import re

    pool = yaml.safe_load((REPO / "config" / "gpu_pool.yaml").read_text(encoding="utf-8"))
    chat = _card_indices(pool, "chat")
    allowed = _card_indices(pool, "agent") | _card_indices(pool, "agent-gpu2")
    assert chat and allowed and not (chat & allowed)

    # Compose: required, no fallback that could land on any card.
    compose = yaml.safe_load((HOST / "docker-compose.yml").read_text(encoding="utf-8"))
    env = compose["services"]["bonsai-worker"]["environment"]
    cuda = next(e.split("=", 1)[1] for e in env if e.startswith("CUDA_VISIBLE_DEVICES_OVERRIDE="))
    assert re.fullmatch(r"\$\{BONSAI_CUDA_VISIBLE_DEVICES:\?[^}]+\}", cuda), cuda

    # Operator template.
    example = (HOST / ".env_example").read_text(encoding="utf-8")
    vals = re.findall(r"^BONSAI_CUDA_VISIBLE_DEVICES=(.*)$", example, re.M)
    assert len(vals) == 1, vals
    devices = {int(d) for d in vals[0].split(",")}
    assert devices and devices <= allowed and not (devices & chat), devices

    # Every Bonsai profile's (doc-only) pin.
    profiles = yaml.safe_load((REPO / "config" / "llm_profiles.yaml").read_text(encoding="utf-8"))["profiles"]
    bonsai = {k: v for k, v in profiles.items() if "bonsai" in k.lower()}
    assert PROFILE in bonsai
    for name, cfg in bonsai.items():
        ids = set((cfg.get("gpu") or {}).get("device_ids") or [])
        assert ids <= allowed and not (ids & chat), (name, ids)


def test_compose_refuses_to_start_without_a_card(tmp_path):
    """`docker compose config` with no BONSAI_CUDA_VISIBLE_DEVICES must fail, not pick a card."""
    import shutil
    import subprocess

    import pytest

    if not shutil.which("docker"):
        pytest.skip("docker CLI not installed")
    probe = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("docker compose plugin not installed")
    envf = tmp_path / "empty.env"
    envf.write_text("LLM_CACHE_DIR=/tmp\n", encoding="utf-8")
    run = lambda extra: subprocess.run(  # noqa: E731
        ["docker", "compose", "--env-file", str(envf), "-f", str(HOST / "docker-compose.yml"), "config"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": __import__("os").environ["PATH"], **extra},
    )
    missing = run({})
    assert missing.returncode != 0
    assert "never 0" in missing.stderr
    ok = run({"BONSAI_CUDA_VISIBLE_DEVICES": "2"})
    assert ok.returncode == 0, ok.stderr
    assert "CUDA_VISIBLE_DEVICES_OVERRIDE: \"2\"" in ok.stdout or "CUDA_VISIBLE_DEVICES_OVERRIDE: '2'" in ok.stdout or "CUDA_VISIBLE_DEVICES_OVERRIDE: 2" in ok.stdout

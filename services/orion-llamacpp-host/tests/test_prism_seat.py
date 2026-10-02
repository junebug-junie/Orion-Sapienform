"""GPU pool stage 7.2: one image (Dockerfile.prism) serves agent-gpu2's Bonsai profile on the PrismML
fork and its Q4 rollback profile on the stock binary; the profile picks the binary."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml

HOST = Path(__file__).resolve().parents[1]
REPO = HOST.parents[1]
BONSAI = "ternary-bonsai2-27b-pq2-v100-32gb-circe-agent"
Q4 = "qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex"
FLAGS = {"--jinja", "--reasoning", "--reasoning-format", "--chat-template-kwargs", "--flash-attn",
         "--no-context-shift", "--n-predict", "--temp", "--top-k", "--top-p", "--min-p",
         "--presence-penalty", "--cache-ram", "--cache-idle-slots", "--no-cache-idle-slots"}


def _profile(name: str, **llamacpp_over):
    profiles_mod = importlib.import_module("app.profiles")
    raw = yaml.safe_load((REPO / "config" / "llm_profiles.yaml").read_text(encoding="utf-8"))
    cfg = json.loads(json.dumps(raw["profiles"][name]))
    cfg["llamacpp"].update(llamacpp_over)
    return profiles_mod.LLMProfile(name=name, **cfg)


@pytest.fixture
def wrapper(monkeypatch, tmp_path):
    """app.main with the image layout faked under tmp: /app/llama-server and /app/prism/llama-server."""
    main = importlib.import_module("app.main")
    stock = tmp_path / "app" / "llama-server"
    prism_dir = tmp_path / "app" / "prism"
    prism_dir.mkdir(parents=True)
    stock.write_text("")
    (prism_dir / "llama-server").write_text("")
    monkeypatch.setattr(main, "STOCK_SERVER_BIN", str(stock))
    monkeypatch.setattr(main, "PRISM_SERVER_DIR", str(prism_dir))
    monkeypatch.setattr(main, "_ensure_model_file", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "_get_supported_llama_server_flags", lambda _bin: set(FLAGS))
    monkeypatch.setattr(main, "_get_llama_server_build", lambda _bin: 10750)
    settings = importlib.import_module("app.settings").settings
    for knob in ("llamacpp_ctx_size_override", "llamacpp_n_parallel_override", "llamacpp_model_path_override"):
        monkeypatch.setattr(settings, knob, None)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
    return main, stock, prism_dir


def _flag(cmd, f):
    return cmd[cmd.index(f) + 1]


def test_bonsai_agent_profile_runs_the_fork_with_two_131k_slots(wrapper):
    main, _stock, prism_dir = wrapper
    cmd, env = main.build_llama_server_cmd_and_env(_profile(BONSAI))
    assert cmd[0] == str(prism_dir / "llama-server")
    assert _flag(cmd, "-m").endswith("Ternary-Bonsai-2-27B-PQ2_0.gguf")
    assert _flag(cmd, "--parallel") == "2"
    assert int(_flag(cmd, "--ctx-size")) // int(_flag(cmd, "--parallel")) == 131072
    assert _flag(cmd, "--flash-attn") == "on"
    # Prism KNOWN_ISSUES: --reasoning on overrides a client's thinking-off request.
    assert _flag(cmd, "--reasoning") == "auto"
    kwargs = json.loads(_flag(cmd, "--chat-template-kwargs"))
    assert kwargs == {"reasoning_effort": "xhigh", "preserve_thinking": True}
    assert _flag(cmd, "--n-predict") == "16384"
    # The fork loads its own libllama/libggml first.
    assert env["LD_LIBRARY_PATH"] == f"{prism_dir}:/usr/local/cuda/lib64"
    # #27148 knobs unset: the binary's defaults (spec D2, not reproduced on 2026-10-01).
    assert not {"--cache-ram", "--cache-idle-slots", "--no-cache-idle-slots"} & set(cmd)


def test_q4_rollback_profile_runs_the_stock_binary_with_an_untouched_env(wrapper):
    main, stock, _prism = wrapper
    cmd, env = main.build_llama_server_cmd_and_env(_profile(Q4))
    assert cmd[0] == str(stock)
    assert _flag(cmd, "--parallel") == "1" and _flag(cmd, "--ctx-size") == "131072"
    assert env["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"
    assert main._server_bin_env(str(stock)) is None


def test_prism_profile_refuses_to_boot_on_an_image_without_the_fork(wrapper):
    main, _stock, prism_dir = wrapper
    (prism_dir / "llama-server").unlink()
    with pytest.raises(RuntimeError, match="Dockerfile.prism"):
        main.build_llama_server_cmd_and_env(_profile(BONSAI))


def test_probes_run_the_fork_with_its_own_libraries(monkeypatch, tmp_path):
    """--help/--version probes decide which flags are emitted, so they must load the fork's own
    libraries too; the stock binary's probes keep the inherited env (env=None)."""
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "PRISM_SERVER_DIR", str(tmp_path))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
    seen = []

    class _Done:
        returncode = 0
        stdout = "version: 0.2.0-dev (build 10750, commit 88c4bc6)\n  --flash-attn"
        stderr = ""

    def fake_run(argv, **kw):
        seen.append((argv[0], (kw.get("env") or {}).get("LD_LIBRARY_PATH")))
        return _Done()

    monkeypatch.setattr(main.subprocess, "run", fake_run)
    fork = str(tmp_path / "llama-server")
    assert main._get_llama_server_build.__wrapped__(fork) == 10750
    main._get_supported_llama_server_flags.__wrapped__(fork)
    main._get_llama_server_build.__wrapped__("/app/llama-server")
    assert seen == [(fork, f"{tmp_path}:/usr/local/cuda/lib64"), (fork, f"{tmp_path}:/usr/local/cuda/lib64"),
                    ("/app/llama-server", None)]


@pytest.mark.parametrize("over,expected", [
    ({"cache_ram_mib": 0}, ["--cache-ram", "0"]),
    ({"cache_idle_slots": False}, ["--no-cache-idle-slots"]),
    ({"cache_idle_slots": True}, ["--cache-idle-slots"]),
])
def test_cache_knobs_emit_their_flags_when_set(wrapper, over, expected):
    main, _stock, _prism = wrapper
    cmd, _env = main.build_llama_server_cmd_and_env(_profile(BONSAI, **over))
    i = cmd.index(expected[0])
    assert cmd[i:i + len(expected)] == expected


def test_cache_knob_fails_closed_on_a_binary_without_it(wrapper, monkeypatch):
    main, _stock, _prism = wrapper
    monkeypatch.setattr(main, "_get_supported_llama_server_flags", lambda _bin: FLAGS - {"--no-cache-idle-slots"})
    with pytest.raises(RuntimeError, match="no-cache-idle-slots"):
        main.build_llama_server_cmd_and_env(_profile(BONSAI, cache_idle_slots=False))


def test_prism_dockerfile_keeps_both_binaries_and_pins_the_volta_fork():
    text = (HOST / "Dockerfile.prism").read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "PrismML-Eng/llama.cpp" in code and "88c4bc60b9c9578f134385be9535e853f2db9b9f" in code
    assert "CMAKE_CUDA_ARCHITECTURES=70" in code and "GGML_CUDA_FA_ALL_QUANTS=ON" in code
    assert "nvidia/cuda:12.8.1-devel-ubuntu24.04" in code
    # Build number = rev-list count; a shallow clone reports 1 and the wrapper drops --flash-attn.
    assert "--depth" not in code and "-gt 5332" in code
    # Same stock base as Dockerfile (the Q4 rollback runs on it), fork beside it, never over /app.
    assert "FROM ghcr.io/ggml-org/llama.cpp:${LLAMACPP_IMAGE_TAG}" in code
    assert "COPY --from=prism-build /src/build/bin/ /app/prism/" in code
    assert "/src/build/bin/ /app/\n" not in code
    for copy in ("COPY services/orion-llamacpp-host/app /app/app", "COPY config /app/config", "COPY orion /app/orion"):
        assert copy in code
    assert 'ENTRYPOINT ["python3", "-m", "app.main"]' in code


def test_burst_seat_builds_the_prism_image_under_its_own_tag():
    compose = yaml.safe_load((HOST / "docker-compose.atlas-workers.yml").read_text(encoding="utf-8"))["services"]
    burst = compose["atlas-agent-burst"]
    assert burst["build"]["dockerfile"] == "services/orion-llamacpp-host/Dockerfile.prism"
    assert burst["image"] == "orion-llamacpp-host-prism:0.1.0"
    for name in ("atlas-chat", "atlas-metacog", "atlas-fast", "atlas-agent"):
        assert compose[name]["image"] == "orion-llamacpp-host:0.1.0", name
        assert compose[name]["build"]["dockerfile"] == "services/orion-llamacpp-host/Dockerfile", name
    script = (HOST / "scripts" / "build-prism-volta.sh").read_text(encoding="utf-8")
    assert "Dockerfile.prism" in script and "orion-llamacpp-host-prism:0.1.0" in script

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import sync_local_env_from_example as sync_mod  # noqa: E402
from sync_local_env_from_example import (  # noqa: E402
    NEVER_SYNC_KEYS,
    example_value_is_host_placeholder,
    should_sync_key,
    sync_file,
)


def test_orion_bus_url_never_synced() -> None:
    assert "ORION_BUS_URL" in NEVER_SYNC_KEYS
    assert should_sync_key("ORION_BUS_URL", all_keys=False) is False
    assert should_sync_key("ORION_BUS_URL", all_keys=True) is False


def test_default_sync_reaches_admission_templates_and_preserves_overrides(tmp_path, monkeypatch, capsys):
    """The documented default command must populate the real admission contract."""
    prefixes = {
        "orion-durable-runs": ("DURABLE_RUNS_",),
        "orion-cortex-orch": ("CORTEX_DURABLE_",),
        "orion-hub": ("HUB_CURIOSITY_DURABLE_",),
        "orion-llm-gateway": ("GPU_POOL_", "LLM_GATEWAY_POOL_"),
    }
    branch = tmp_path / "branch"
    primary = tmp_path / "primary"
    expected = {}
    for name in sync_mod.DEFAULT_SERVICES:
        example_dir = branch / "services" / name
        env_dir = primary / "services" / name
        example_dir.mkdir(parents=True)
        env_dir.mkdir(parents=True)
        values = {}
        if name in prefixes:
            values = {key: value for key, value in sync_mod.parse_kv(
                ROOT / "services" / name / ".env_example").items()
                if key.startswith(prefixes[name])}
        (example_dir / ".env_example").write_text("".join(f"{k}={v}\n" for k, v in values.items()))
        (env_dir / ".env").write_text("ORION_BUS_URL=redis://100.92.216.81:6379/0\n")
        if name in prefixes:
            assert values
            expected[name] = values
    assert set(expected) == set(prefixes), "an admission service is missing from the default scan"
    live = primary / "services" / "orion-durable-runs" / ".env"
    live.write_text(live.read_text() + "DURABLE_RUNS_LEASE_SECONDS=180\n")
    monkeypatch.setattr(sync_mod, "ROOT", branch)
    monkeypatch.setattr(sync_mod, "main_worktree_root", lambda: primary)
    monkeypatch.setattr(sys, "argv", ["sync_local_env_from_example.py"])
    assert sync_mod.main() == 0
    first = {}
    for name, values in expected.items():
        path = primary / "services" / name / ".env"
        actual = sync_mod.parse_kv(path)
        assert actual["ORION_BUS_URL"] == "redis://100.92.216.81:6379/0"
        for key, value in values.items():
            assert actual[key] == ("180" if key == "DURABLE_RUNS_LEASE_SECONDS" else value)
        first[name] = path.read_bytes()
    assert sync_mod.main() == 0
    assert all((primary / "services" / name / ".env").read_bytes() == data for name, data in first.items())
    capsys.readouterr()


def test_recall_graphiti_chat_keys_never_synced() -> None:
    """RECALL_GRAPHITI_IN_CHAT and RECALL_GRAPHITI_ADAPTER_URL gate real chat-time graph
    search (orion-recall). Both must change together by hand or the feature silently
    no-ops (enabled flag + empty/stale URL) -- the exact bug already hit twice this
    session with CONCEPT_RELATION_RESOLUTION_ENABLED and GRAPHITI_BACKEND."""
    for key in ("RECALL_GRAPHITI_IN_CHAT", "RECALL_GRAPHITI_ADAPTER_URL"):
        assert key in NEVER_SYNC_KEYS
        assert should_sync_key(key, all_keys=False) is False
        assert should_sync_key(key, all_keys=True) is False


def test_sync_file_force_does_not_touch_recall_graphiti_chat_keys(tmp_path: Path) -> None:
    """Even --force (which overwrites ordinary diverged keys) must never touch these --
    NEVER_SYNC_KEYS is a stricter, unconditional exclusion, unlike the diverged/--force
    behavior that applies to everything else."""
    svc = tmp_path / "orion-recall"
    svc.mkdir()
    (svc / ".env_example").write_text(
        "RECALL_GRAPHITI_IN_CHAT=true\n"
        "RECALL_GRAPHITI_ADAPTER_URL=http://orion-athena-graphiti-adapter:8000\n",
        encoding="utf-8",
    )
    (svc / ".env").write_text(
        "RECALL_GRAPHITI_IN_CHAT=false\nRECALL_GRAPHITI_ADAPTER_URL=\n",
        encoding="utf-8",
    )

    result = sync_file(
        svc / ".env", svc / ".env_example", dry_run=False, all_keys=True, force=True
    )

    text = (svc / ".env").read_text(encoding="utf-8")
    assert "RECALL_GRAPHITI_IN_CHAT=false" in text
    assert "RECALL_GRAPHITI_ADAPTER_URL=\n" in text
    assert not any("RECALL_GRAPHITI" in c for c in result.updated + result.diverged)


def test_placeholder_bus_url_skipped_even_if_all_keys() -> None:
    assert example_value_is_host_placeholder("ORION_BUS_URL", "redis://100.x.x.x:6379/0")
    assert example_value_is_host_placeholder("ORION_BUS_URL", "redis://bus-core:6379/0")
    assert not example_value_is_host_placeholder("ORION_BUS_URL", "redis://100.92.216.81:6379/0")


def test_sync_file_does_not_clobber_local_bus_url(tmp_path: Path) -> None:
    svc = tmp_path / "orion-thought"
    svc.mkdir()
    (svc / ".env_example").write_text(
        "ORION_BUS_URL=redis://100.x.x.x:6379/0\nSTANCE_REACT_TIMEOUT_SEC=120\n",
        encoding="utf-8",
    )
    (svc / ".env").write_text(
        "ORION_BUS_URL=redis://100.92.216.81:6379/0\nSTANCE_REACT_TIMEOUT_SEC=12\n",
        encoding="utf-8",
    )
    result = sync_file(svc / ".env", svc / ".env_example", dry_run=False, all_keys=True)
    text = (svc / ".env").read_text(encoding="utf-8")
    # ORION_BUS_URL is in NEVER_SYNC_KEYS: excluded entirely, regardless of divergence.
    assert "ORION_BUS_URL=redis://100.92.216.81:6379/0" in text
    assert not any("ORION_BUS_URL" in c for c in result.updated + result.diverged)
    # STANCE_REACT_TIMEOUT_SEC diverges (12 vs 120) but force defaults to False, so it
    # must NOT be silently overwritten — this is the bug this module now guards against.
    assert "STANCE_REACT_TIMEOUT_SEC=12" in text
    assert "STANCE_REACT_TIMEOUT_SEC=120" not in text
    assert any("STANCE_REACT_TIMEOUT_SEC" in c for c in result.diverged)
    assert not any("STANCE_REACT_TIMEOUT_SEC" in c for c in result.updated)


def test_sync_file_default_does_not_overwrite_diverged_key(tmp_path: Path) -> None:
    """Regression test for the graphiti-adapter incident: an existing local value that
    differs from .env_example (an intentional deployment-specific override) must be
    left alone by default, and reported as diverged rather than silently reset."""
    svc = tmp_path / "orion-graphiti-adapter"
    svc.mkdir()
    (svc / ".env_example").write_text("GRAPHITI_BACKEND=orion_postgres\n", encoding="utf-8")
    (svc / ".env").write_text("GRAPHITI_BACKEND=graphiti_core\n", encoding="utf-8")

    result = sync_file(svc / ".env", svc / ".env_example", dry_run=False, all_keys=False)

    text = (svc / ".env").read_text(encoding="utf-8")
    assert "GRAPHITI_BACKEND=graphiti_core" in text
    assert not result.updated
    assert any("GRAPHITI_BACKEND" in c for c in result.diverged)
    assert "local='graphiti_core'" in result.diverged[0]
    assert "example='orion_postgres'" in result.diverged[0]


def test_sync_file_force_overwrites_diverged_key(tmp_path: Path) -> None:
    svc = tmp_path / "orion-graphiti-adapter"
    svc.mkdir()
    (svc / ".env_example").write_text("GRAPHITI_BACKEND=orion_postgres\n", encoding="utf-8")
    (svc / ".env").write_text("GRAPHITI_BACKEND=graphiti_core\n", encoding="utf-8")

    result = sync_file(
        svc / ".env", svc / ".env_example", dry_run=False, all_keys=False, force=True
    )

    text = (svc / ".env").read_text(encoding="utf-8")
    assert "GRAPHITI_BACKEND=orion_postgres" in text
    assert not result.diverged
    assert any("GRAPHITI_BACKEND" in c for c in result.updated)


def test_sync_file_missing_key_auto_added_default_and_force(tmp_path: Path) -> None:
    for force in (False, True):
        svc = tmp_path / f"orion-graphiti-adapter-{force}"
        svc.mkdir()
        (svc / ".env_example").write_text(
            "GRAPHITI_BACKEND=orion_postgres\nCRYSTALLIZER_NEW_KEY=example_value\n",
            encoding="utf-8",
        )
        (svc / ".env").write_text("GRAPHITI_BACKEND=orion_postgres\n", encoding="utf-8")

        result = sync_file(
            svc / ".env", svc / ".env_example", dry_run=False, all_keys=False, force=force
        )

        text = (svc / ".env").read_text(encoding="utf-8")
        assert "CRYSTALLIZER_NEW_KEY=example_value" in text
        assert any("CRYSTALLIZER_NEW_KEY" in c for c in result.updated)
        assert not result.diverged


# --- worktree resolution (2026-07-31) -------------------------------------
#
# `.env` is gitignored, so it exists only in the primary checkout. CLAUDE.md
# mandates all work happen in a linked worktree, so running this script the way
# the contract instructs made it a silent no-op: 24 services reported
# `skip <name>: no .env` and it exited 0. A real divergence
# (EQUILIBRIUM_METACOG_TRANSPORT_BUS_SYNAPTIC_ERROR_THRESHOLD left at the retired
# metric's 1.0 after PR #1542 moved the template to 0.15) went unreported, and a
# detector that could not fire shipped with the parity gate reporting clean.


def test_main_worktree_root_resolves_primary_checkout_from_a_linked_worktree(tmp_path):
    """A linked worktree must resolve `.env` back to the primary checkout."""
    import subprocess as sp

    primary = tmp_path / "primary"
    (primary / "services").mkdir(parents=True)
    sp.run(["git", "init", "-q", str(primary)], check=True)
    sp.run(["git", "-C", str(primary), "config", "user.email", "t@t"], check=True)
    sp.run(["git", "-C", str(primary), "config", "user.name", "t"], check=True)
    (primary / "README.md").write_text("x\n")
    sp.run(["git", "-C", str(primary), "add", "-A"], check=True)
    sp.run(["git", "-C", str(primary), "commit", "-qm", "init"], check=True)

    linked = tmp_path / "linked"
    sp.run(
        ["git", "-C", str(primary), "worktree", "add", "-q", "-b", "wt", str(linked)],
        check=True,
    )

    assert sync_mod.main_worktree_root(linked).resolve() == primary.resolve()
    # Idempotent from the primary checkout itself.
    assert sync_mod.main_worktree_root(primary).resolve() == primary.resolve()


def test_main_worktree_root_falls_back_outside_a_git_repo(tmp_path):
    """No git, no repo, no surprises -- behave exactly as before."""
    plain = tmp_path / "plain"
    plain.mkdir()
    assert sync_mod.main_worktree_root(plain) == plain


def test_main_worktree_root_falls_back_when_services_dir_is_absent(tmp_path):
    """Guard against resolving to something that is not this repo.

    A bare/odd git layout must not silently redirect `.env` writes somewhere
    unrelated -- the `services/` probe is what makes the redirect safe.
    """
    import subprocess as sp

    repo = tmp_path / "norepo"
    repo.mkdir()
    sp.run(["git", "init", "-q", str(repo)], check=True)
    assert sync_mod.main_worktree_root(repo) == repo


def test_camera_rtsp_urls_never_synced() -> None:
    # They carry camera credentials; a --force sync must not replace a live URL
    # with the template placeholder.
    assert {"REOLINK_URL", "WALKWAY_RTSP_URL"} <= NEVER_SYNC_KEYS


def test_gpu_pool_keys_are_reached_by_the_default_sync() -> None:
    """Every key the GPU pool cutover added must be synced by the default command: a key no
    prefix matches is skipped silently, with a "No changes needed" that looks like a pass."""
    assert "orion-gpu-pool" in sync_mod.DEFAULT_SERVICES
    for service, prefix_filter in (("orion-llm-gateway", ("GPU_POOL_", "LLM_GATEWAY_POOL_", "LLM_GATEWAY_EXECUTOR_")),
                                   ("orion-gpu-pool", ("GPU_POOL_",))):
        keys = [k for k in sync_mod.parse_kv(ROOT / "services" / service / ".env_example")
                if k.startswith(prefix_filter)]
        assert keys, service
        for key in keys:
            assert should_sync_key(key, all_keys=False), key


def test_substrate_reconcile_keys_are_reached_by_the_default_sync() -> None:
    """The bounded-reconciler keys (2026-09-25) must land in local .env on a default sync."""
    for service, prefix in (("orion-policy-runtime", "POLICY_RECONCILE_"),
                            ("orion-execution-dispatch-runtime", "DISPATCH_RECONCILE_"),
                            ("orion-feedback-runtime", "FEEDBACK_RECONCILE_")):
        assert service in sync_mod.DEFAULT_SERVICES
        keys = [k for k in sync_mod.parse_kv(ROOT / "services" / service / ".env_example")
                if k.startswith(prefix)]
        assert len(keys) == 4, (service, keys)
        for key in keys:
            assert should_sync_key(key, all_keys=False), key


def test_secret_values_never_printed(tmp_path) -> None:
    """A diverged/forced/added secret must be reported by key only, never by value."""
    svc = tmp_path / "svc"
    svc.mkdir()
    live, example = "ghp_LIVEsecretVALUE123", "ghp_EXAMPLEsecretVALUE456"
    (svc / ".env_example").write_text(
        f"COCREATION_SIGNALS_GH_TOKEN={example}\nNEW_API_KEY=example-key-789\nPLAIN_LEVEL=info\n"
    )
    (svc / ".env").write_text(f"COCREATION_SIGNALS_GH_TOKEN={live}\nPLAIN_LEVEL=debug\n")

    diverged = sync_file(svc / ".env", svc / ".env_example", dry_run=True, all_keys=True)
    forced = sync_file(svc / ".env", svc / ".env_example", dry_run=True, all_keys=True, force=True)
    report = "\n".join(diverged.diverged + diverged.updated + forced.updated)

    for secret in (live, example, "example-key-789"):
        assert secret not in report
    assert "COCREATION_SIGNALS_GH_TOKEN" in report and "NEW_API_KEY" in report
    assert "'debug'" in report and "'info'" in report

    sync_file(svc / ".env", svc / ".env_example", dry_run=False, all_keys=True, force=True)
    assert f"COCREATION_SIGNALS_GH_TOKEN={example}" in (svc / ".env").read_text()


def test_display_value_masks_url_passwords_and_pass_keys_but_not_token_budgets() -> None:
    from sync_local_env_from_example import display_value

    shown = display_value("POSTGRES_URI", "postgresql://orion:hunter2pw@db:5432/orion")
    assert "hunter2pw" not in shown and "orion:***@db:5432" in shown
    assert "pw123" not in display_value("GRAPHDB_PASS", "pw123")
    assert "AKIA" not in display_value("LIGHTDASH_S3_ACCESS_KEY", "AKIAxyz")
    assert display_value("LLM_CHAT_GENERAL_MAX_TOKENS", "4096") == "'4096'"
    assert display_value("ORION_STATE_KEY", "orion:state") == "'orion:state'"
    assert display_value("ORION_BUS_URL", "redis://100.92.216.81:6379/0") == "'redis://100.92.216.81:6379/0'"


def test_energy_keys_are_reached_by_the_default_sync() -> None:
    """orion-energy was absent from DEFAULT_SERVICES and no prefix matched ENERGY_: the default
    run skipped all its keys while reporting other services, which read as a pass."""
    assert "orion-energy" in sync_mod.DEFAULT_SERVICES
    keys = [k for k in sync_mod.parse_kv(ROOT / "services" / "orion-energy" / ".env_example")
            if k.startswith("ENERGY_")]
    assert len(keys) >= 14, keys
    for key in keys:
        if key in NEVER_SYNC_KEYS:
            continue
        assert should_sync_key(key, all_keys=False), key


def test_energy_usage_point_id_never_synced_even_with_force(tmp_path: Path) -> None:
    """The template ships ENERGY_USAGE_POINT_ID empty; the live value identifies the house's
    meter. --force must never flatten a pasted value back to that placeholder."""
    assert "ENERGY_USAGE_POINT_ID" in NEVER_SYNC_KEYS
    assert should_sync_key("ENERGY_USAGE_POINT_ID", all_keys=True) is False
    svc = tmp_path / "orion-energy"
    svc.mkdir()
    (svc / ".env_example").write_text("ENERGY_USAGE_POINT_ID=\nENERGY_STAKES_NEAR_RATIO=0.95\n", encoding="utf-8")
    (svc / ".env").write_text("ENERGY_USAGE_POINT_ID=up-123\nENERGY_STAKES_NEAR_RATIO=0.9\n", encoding="utf-8")

    result = sync_file(svc / ".env", svc / ".env_example", dry_run=False, all_keys=True, force=True)

    text = (svc / ".env").read_text(encoding="utf-8")
    assert "ENERGY_USAGE_POINT_ID=up-123\n" in text
    assert "ENERGY_STAKES_NEAR_RATIO=0.95" in text
    assert not any("ENERGY_USAGE_POINT_ID" in c for c in result.updated + result.diverged)

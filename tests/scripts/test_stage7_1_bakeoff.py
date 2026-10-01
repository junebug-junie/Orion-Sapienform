"""GPU pool stage 7.1 bake-off tooling (scripts/bench/stage7_1_*).

The orchestrator is exercised in DRY_RUN with stub docker / nvidia-smi / curl on PATH, so nothing
here touches a GPU, the pool, or the bus. Key property: every exit after the pool is paused
releases the hold and resumes the pool.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench" / "stage7_1_bakeoff.sh"
spec = importlib.util.spec_from_file_location("stage7_1_client", ROOT / "scripts" / "bench" / "stage7_1_client.py")
client = importlib.util.module_from_spec(spec)
sys.modules["stage7_1_client"] = client   # dataclasses resolve the module by name
spec.loader.exec_module(client)


# ----------------------------------------------------------------------------- fixtures

def clean_pool(**over) -> dict:
    pool = {
        "mode": "enforce", "generated_at": "2026-10-01T03:31:34Z", "actuation_paused": None,
        "cards": [{"card": "gpu2", "swap_state": "idle", "swapped_in": [],
                   "actuation": {"role": "agent-gpu2", "action": "unload", "phase": "ready_wait",
                                 "finished_at": "2026-09-30T23:48:41Z"}}],
        "roles": [{"role": "agent-gpu2", "status": "unloaded", "cards": ["gpu2"], "slots": 0},
                  {"role": "diffusion", "status": "static", "cards": ["gpu2"], "slots": 1},
                  {"role": "world", "status": "static", "cards": ["gpu2"], "slots": 2}],
        "leases": [{"lease_id": "old", "status": "dead_letter", "role": None, "kind": "hold", "holder": "x"}],
    }
    pool.update(over)
    return pool


def make_stubs(tmp: Path, pool: dict, gpu2_mib: int = 1248) -> dict:
    bin_ = tmp / "bin"
    bin_.mkdir()
    (tmp / "pool.json").write_text(json.dumps(pool))
    calls = tmp / "calls.log"
    stubs = {
        # GET /v1/pool -> fixture; the bake-off port's /health -> down (no worker running)
        "curl": f'#!/bin/sh\necho "curl $*" >> {calls}\ncase "$*" in *v1/pool*) cat {tmp}/pool.json ;; *) exit 7 ;; esac\n',
        "nvidia-smi": f'#!/bin/sh\necho "nvidia-smi $*" >> {calls}\ncase "$*" in *memory.used*) echo {gpu2_mib} ;; esac\n',
        "docker": f'#!/bin/sh\necho "docker $*" >> {calls}\nexit 0\n',
    }
    for name, body in stubs.items():
        p = bin_ / name
        p.write_text(body)
        p.chmod(0o755)
    # CI has no gitignored service .env: point at the template (dry run only reads its path).
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", DRY_RUN="1", RESULTS_ROOT=str(tmp / "results"),
               BONSAI_ENV_FILE=str(ROOT / "services" / "orion-llamacpp-bonsai-host" / ".env_example"))
    env.pop("BENCH_FAIL_AT", None)
    return env


def run(args, env, **kw):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60, **kw)


def lines_with(out: str, needle: str) -> list[int]:
    return [i for i, l in enumerate(out.splitlines()) if needle in l]


# ----------------------------------------------------------------------------- orchestrator (dry run)

def test_dry_run_prints_every_mutating_step_in_order(tmp_path):
    env = make_stubs(tmp_path, clean_pool())
    r = run(["run", "--yes"], env)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    pause = lines_with(out, "DRY_RUN: pool pause")
    hold = lines_with(out, "DRY_RUN: pool hold")
    up = lines_with(out, "docker compose -p orion-bench71")
    stop = lines_with(out, "DRY_RUN: docker stop -t 30 orion-bench71-bonsai-worker")
    release = lines_with(out, "DRY_RUN: pool release --lease-id dry-run-lease")
    resume = lines_with(out, "DRY_RUN: pool resume")
    assert pause and hold and up and stop and release and resume, out
    # hold BEFORE pause: the pool refuses operator holds while paused (pinned in-process by
    # services/orion-gpu-pool/tests/test_stage7_1_bakeoff_sequence.py)
    assert hold[0] < pause[0] < up[0] < stop[-1] < release[0] < resume[0]
    compose = out.splitlines()[up[0]]
    assert "BONSAI_CUDA_VISIBLE_DEVICES=2" in compose and "up -d --no-build bonsai-worker" in compose
    assert "BONSAI_PROFILE_NAME=ternary-bonsai2-27b-pq2-v100-32gb-circe-np4" in compose
    override = next((tmp_path / "results").glob("2*/override-bonsai.yml")).read_text()
    assert "LLAMACPP_N_PARALLEL_OVERRIDE=2" in override and "LLAMACPP_CTX_SIZE_OVERRIDE=262144" in override
    # dry run never ran a real docker mutation (the stub only saw read-only calls)
    docker_calls = [l for l in (tmp_path / "calls.log").read_text().splitlines() if l.startswith("docker ")]
    assert all(l.split()[1] in ("ps", "inspect", "logs") for l in docker_calls), docker_calls
    assert "--with-q4" not in out and "q4" not in "".join(l for l in out.splitlines() if "starting" in l)


@pytest.mark.parametrize("step", ["after_hold", "after_pause", "after_start_bonsai", "after_bench_bonsai"])
def test_failure_mid_run_still_releases_and_resumes(tmp_path, step):
    env = make_stubs(tmp_path, clean_pool())
    env["BENCH_FAIL_AT"] = step
    r = run(["run", "--yes"], env)
    out = r.stdout
    assert r.returncode != 0
    assert f"SIMULATED FAILURE at {step}" in out
    fail = lines_with(out, "SIMULATED FAILURE")[0]
    release = lines_with(out, "pool release --lease-id dry-run-lease")
    assert release and release[0] > fail, out
    resume = lines_with(out, "DRY_RUN: pool resume")
    if step == "after_hold":    # failed before pausing: nothing to resume
        assert not resume and "nothing to resume" in out, out
    else:
        assert resume and resume[0] > release[0], out
    assert not (tmp_path / "results" / ".state-dryrun" / "paused").exists()
    assert not (tmp_path / "results" / ".state").exists(), "dry run must never write the real markers"


def test_with_q4_runs_a_second_pass_at_2x65k(tmp_path):
    env = make_stubs(tmp_path, clean_pool())
    r = run(["run", "--yes", "--with-q4"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    q4 = [l for l in r.stdout.splitlines() if "starting q4" in l]
    assert q4 and "ctx=131072" in q4[0] and "orion-llamacpp-host:0.1.0" in q4[0]
    after_q4 = r.stdout.split("starting q4")[1]
    assert "q4 bench (depths 14000,32000,61000)" in after_q4 and "--depths 14000,32000,61000\n" in after_q4


def test_a_lost_pause_reply_is_still_resumed(tmp_path):
    """The pause may land while its reply is lost: the marker goes down before the pause is sent."""
    env = make_stubs(tmp_path, clean_pool())
    env["DRY_CTL_PAUSE"] = '{"ok": false, "reason": "no_answer", "detail": {}}'
    env["DRY_CTL_PAUSE_RC"] = "2"
    r = run(["run", "--yes"], env)
    assert r.returncode != 0 and "pause failed" in r.stdout
    assert "DRY_RUN: pool resume" in r.stdout and "pool release --lease-id dry-run-lease" in r.stdout


def test_someone_elses_pause_is_never_resumed(tmp_path):
    env = make_stubs(tmp_path, clean_pool())
    env["DRY_CTL_PAUSE"] = '{"ok": true, "reason": "already_paused", "detail": {"by": "juniper"}}'
    r = run(["run", "--yes"], env)
    assert r.returncode != 0 and "already_paused" in r.stdout
    assert "pool resume" not in r.stdout
    assert "pool release --lease-id dry-run-lease" in r.stdout      # our hold is still released


def test_a_refused_hold_changes_nothing_else(tmp_path):
    env = make_stubs(tmp_path, clean_pool())
    env["DRY_CTL_HOLD"] = '{"ok": false, "reason": "actuation_paused", "detail": {}}'
    env["DRY_CTL_HOLD_RC"] = "1"
    r = run(["run", "--yes"], env)
    assert r.returncode != 0 and "diffusion hold refused" in r.stdout
    assert "pool pause" not in r.stdout and "pool resume" not in r.stdout and "docker compose" not in r.stdout


def test_sigterm_mid_bench_cleans_up_promptly(tmp_path):
    env = make_stubs(tmp_path, clean_pool())
    env["DRY_RUN_STEP_SLEEP"] = "60"
    p = subprocess.Popen(["bash", str(SCRIPT), "run", "--yes"], env=env, stdout=subprocess.PIPE, text=True)
    out = []
    t0 = time.time()
    for line in p.stdout:
        out.append(line)
        if "bonsai bench" in line:
            break
    time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    rest, _ = p.communicate(timeout=30)
    text = "".join(out) + rest
    assert time.time() - t0 < 30, "TERM waited for the step to finish"
    assert "terminated" in text and "DRY_RUN: pool resume" in text and "pool release --lease-id" in text, text
    assert p.returncode == 143


def test_a_second_run_or_cleanup_is_refused_while_one_holds_the_lock(tmp_path):
    env = make_stubs(tmp_path, clean_pool())
    state = tmp_path / "results" / ".state-dryrun"
    state.mkdir(parents=True)
    holder = subprocess.Popen(["flock", str(state / "lock"), "sleep", "20"])
    try:
        time.sleep(0.3)
        for args in (["run", "--yes"], ["cleanup"]):
            r = run(args, env)
            assert r.returncode == 4 and "another bake-off run or cleanup is active" in r.stdout
            assert "pool" not in r.stdout.split("REFUSE")[0] or "pool pause" not in r.stdout
    finally:
        holder.kill()


def test_no_yes_changes_nothing(tmp_path):
    env = make_stubs(tmp_path, clean_pool())
    r = run(["run"], env)
    assert r.returncode == 0
    assert "Re-run with --yes" in r.stdout
    assert "pool pause" not in r.stdout and "cleanup" not in r.stdout


@pytest.mark.parametrize("pool,needle", [
    (clean_pool(roles=[{"role": "agent-gpu2", "status": "confirmed", "cards": ["gpu2"], "slots": 1}]), "agent-gpu2 is confirmed"),
    (clean_pool(actuation_paused={"by": "juniper", "since": "x"}), "already paused by 'juniper'"),
    (clean_pool(leases=[{"lease_id": "d1", "status": "granted", "role": "diffusion", "kind": "request",
                         "holder": "orion-thought"}]), "live request on diffusion"),
])
def test_preflight_refusals_never_pause(tmp_path, pool, needle):
    env = make_stubs(tmp_path, pool)
    r = run(["run", "--yes"], env)
    assert r.returncode == 3, r.stdout
    assert needle in r.stdout
    assert "pool pause" not in r.stdout and "pool resume" not in r.stdout


def test_preflight_refuses_a_loaded_card(tmp_path):
    env = make_stubs(tmp_path, clean_pool(), gpu2_mib=25700)
    r = run(["preflight"], env)
    assert r.returncode == 3 and "25700 MiB" in r.stdout


def test_cleanup_subcommand_dry_run_follows_the_marker(tmp_path):
    """In a dry run the pause is fake, so cleanup follows the marker. A real cleanup asks the pool who
    paused it (stage7_1_client.paused_by, tested below) and uses the marker only if the pool is unreadable."""
    env = make_stubs(tmp_path, clean_pool())
    r = run(["cleanup"], env)
    assert r.returncode == 0
    assert "nothing to resume" in r.stdout and "pool resume" not in r.stdout
    state = tmp_path / "results" / ".state-dryrun"
    state.mkdir(parents=True, exist_ok=True)
    (state / "paused").touch()
    (state / "hold_lease").write_text("L42")
    r = run(["cleanup"], env)
    assert "pool release --lease-id L42" in r.stdout and "DRY_RUN: pool resume" in r.stdout
    assert not (state / "paused").exists() and not (state / "hold_lease").exists()


# ----------------------------------------------------------------------------- pass rule (check 1)

def rows(ratios: dict[int, float], one: float = 30.0):
    return [{"depth": d, "n1_tps": one, "n2_wall_aggregate_tps": one * q, "valid": True} for d, q in ratios.items()]


def test_pass_rule_pass_fail_incomplete():
    assert client.pass_rule(rows({14000: 1.6, 32000: 1.5, 61000: 1.4, 100000: 1.3}))["verdict"] == "PASS"
    v = client.pass_rule(rows({14000: 1.6, 32000: 1.5, 61000: 1.29, 100000: 1.4}))
    assert v["verdict"] == "FAIL" and v["failing"] == [61000]
    v = client.pass_rule(rows({14000: 1.6, 32000: 1.5, 61000: 1.4}))
    assert v["verdict"] == "INCOMPLETE" and v["missing"] == [100000]
    # the 4x65K dip from PR #2434 (23.4 x 2 vs 51.7 alone) is a fail
    assert client.pass_rule([{"depth": 14000, "n1_tps": 51.7, "n2_wall_aggregate_tps": 46.8}], (14000,))["verdict"] == "FAIL"
    # a depth with no single-run number cannot pass
    assert client.pass_rule([{"depth": 14000, "n1_tps": None, "n2_wall_aggregate_tps": 80}], (14000,))["verdict"] == "INCOMPLETE"
    # the per-request sum reads ~2x even when the decodes never overlapped: it is not the pass number
    staggered = {"depth": 14000, "n1_tps": 40.0, "n2_total_tps": 80.0, "n2_wall_aggregate_tps": 41.0, "valid": True}
    assert client.pass_rule([staggered], (14000,))["verdict"] == "FAIL"
    # a row whose warm decodes missed the cache / did not overlap is INVALID -> INCOMPLETE, never PASS
    invalid = {"depth": 14000, "n1_tps": 40.0, "n2_wall_aggregate_tps": 70.0, "valid": False,
               "invalid_reason": "a warm decode missed the slot cache"}
    v = client.pass_rule([invalid], (14000,))
    assert v["verdict"] == "INCOMPLETE" and "missed" in v["per_depth"][0]["why"]


def test_window_overlap():
    assert client.window_overlap([(0, 10), (0, 10)]) == 1.0
    assert client.window_overlap([(0, 10), (10, 20)]) == 0.0
    assert client.window_overlap([(0, 10), (1, 10)]) == 0.9


def test_paused_by():
    assert client.paused_by(None, "stage7-1-bakeoff") == "unknown"
    assert client.paused_by(clean_pool(), "stage7-1-bakeoff") == "running"
    assert client.paused_by(clean_pool(actuation_paused={"by": "stage7-1-bakeoff"}), "stage7-1-bakeoff") == "mine"
    assert client.paused_by(clean_pool(actuation_paused={"by": "juniper"}), "stage7-1-bakeoff") == "other"


# ----------------------------------------------------------------------------- nonce + cache detectors (check 2)

def registry():
    rng = random.Random(7)
    a, b, a2 = client.new_nonce(rng), client.new_nonce(rng), client.new_nonce(rng)
    return a, b, a2, {a: "P1", b: "P2", a2: "P1"}


def test_nonce_detector_finds_a_planted_bleed_in_any_formatting():
    a, b, a2, reg = registry()
    assert client.foreign_nonces(f"the codeword is {a}", "P1", reg) == []
    assert client.foreign_nonces(f"{a} and {a2}", "P1", reg) == []           # both are P1's own
    planted = f"thinking... maybe {b.lower()[:4]}-{b.lower()[4:]} ?"           # P2's, re-typed
    assert client.foreign_nonces(planted, "P1", reg) == [{"nonce": b, "owner": "P2"}]
    spaced = " ".join(b)
    assert client.foreign_nonces(spaced, "P1", reg)[0]["owner"] == "P2"
    assert client.foreign_nonces("", "P1", reg) == []


def test_nonce_alphabet_has_no_ambiguous_characters():
    n = client.new_nonce(random.Random(1))
    assert n.startswith("NX") and len(n) == 12 and not set(n[2:]) & set("01IO")


def test_legit_reuse_bound():
    assert client.lcp([1, 2, 3], [1, 2, 4]) == 2
    hist = [([1, 2, 3, 4], 5), ([1, 9], 0)]
    assert client.legit_reuse_bound([1, 2, 3, 4, 7, 7, 7, 7, 7, 7, 7], hist) == 9   # full prefix + its 5 generated
    assert client.legit_reuse_bound([1, 2, 8], hist) == 2
    assert client.legit_reuse_bound([5, 5], hist, floor=4) == 2                     # floor capped at prompt len
    assert client.legit_reuse_bound([5, 5, 5, 5, 5], [], floor=4) == 4              # template header only


def rec(**k):
    base = {"phase": "pairs", "conv": "P1", "turn": 0, "kind": "recall", "error": None, "foreign": [],
            "cache_over_bound": False, "expects_own": True, "own_found": True, "legit_bound": 4,
            "cache_n": 4, "reasoning_chars": 0, "tool_call_emitted": False}
    base.update(k)
    return base


def test_canary_verdict():
    clean = [rec(turn=i) for i in range(9)] + [rec(turn=9, kind="think", think=True, reasoning_chars=120)]
    no_reasoning = [rec(turn=i) for i in range(9)] + [rec(turn=9, kind="think", think=True, reasoning_chars=0)]
    v = client.canary_verdict(no_reasoning, turns_target=10)
    assert v["verdict"] == "WEAK" and "reasoning" in v["reasons"][-1]
    assert client.canary_verdict(clean, turns_target=10)["verdict"] == "PASS"
    bleed = clean + [rec(kind="think", foreign=[{"nonce": "NXAAAA", "owner": "P2"}], reasoning_chars=40)]
    v = client.canary_verdict(bleed, turns_target=10)
    assert v["verdict"] == "FAIL" and v["bleed_responses"] == 1 and v["examples"][0]["kind"] == "think"
    over = clean + [rec(cache_n=2000, legit_bound=4, cache_over_bound=True)]
    assert client.canary_verdict(over, turns_target=10)["verdict"] == "FAIL"
    blind = [rec(turn=i, own_found=False) for i in range(9)] + [rec(kind="think", think=True, reasoning_chars=5)]
    v = client.canary_verdict(blind, turns_target=10)
    assert v["verdict"] == "WEAK" and "detector" in v["reasons"][0]
    assert client.canary_verdict(clean[:5], turns_target=10)["verdict"] == "WEAK"
    fresh_only = [rec(phase="fresh") for _ in range(10)]
    assert client.canary_verdict(fresh_only, turns_target=10)["conversation_turns"] == 0


def test_summary_and_field_note_table(tmp_path):
    d = tmp_path / "bonsai"
    d.mkdir()
    row = {"depth": 14000, "n1_tps": 40.0, "n2_total_tps": 62.0, "n2_wall_aggregate_tps": 60.0, "ratio": 1.5,
           "valid": True, "n2_min_overlap": 0.97}
    bench = {"server": {"slots": 2, "ctx_per_slot": 131072}, "planned_depths": [14000], "depths": [row],
             "complete": True}
    (d / "bench.json").write_text(json.dumps(bench))
    recs = [rec(), rec(kind="think", think=True, reasoning_chars=50)]
    (d / "canary.json").write_text(json.dumps({"records": recs, "turns_target": 2, "complete": True,
                                               "verdict": client.canary_verdict(recs, turns_target=2)}))
    s = client.summarize(tmp_path)
    assert s["passes"]["bonsai"]["bench"]["verdict"] == "PASS"
    table = client.field_note_table(s)
    assert "| 14,000 | 40.0 | 60.0 (62.0)" in table and "Canary verdict: **PASS**" in table


def test_summary_never_passes_an_interrupted_run(tmp_path):
    d = tmp_path / "bonsai"
    d.mkdir()
    row = {"depth": 14000, "n1_tps": 40.0, "n2_wall_aggregate_tps": 60.0, "valid": True}
    # killed after 14K: 100K was planned and never attempted
    (d / "bench.json").write_text(json.dumps({"planned_depths": [14000, 100000], "depths": [row], "complete": False}))
    (d / "canary.json").write_text(json.dumps({"records": [rec(turn=i) for i in range(5)], "turns_target": 400}))
    s = client.summarize(tmp_path)["passes"]["bonsai"]
    assert s["bench"]["verdict"] == "INCOMPLETE" and 100000 in s["bench"]["missing"]
    assert s["canary"]["verdict"] == "INCOMPLETE" and s["canary"]["turns_target"] == 400


# ----------------------------------------------------------------------------- pool control envelope

def test_pool_ctl_envelopes_validate_against_the_pool_contract():
    pytest.importorskip("pydantic")
    gp = pytest.importorskip("orion.schemas.gpu_pool")
    cspec = importlib.util.spec_from_file_location("stage7_1_pool_ctl", ROOT / "scripts" / "bench" / "stage7_1_pool_ctl.py")
    ctl = importlib.util.module_from_spec(cspec)
    cspec.loader.exec_module(ctl)
    for action, verb in (("pause", "pause_actuation"), ("resume", "resume_actuation"), ("hold", "hold")):
        reply, env = ctl.build(action)
        p = gp.GpuPoolControlV1.model_validate(env.payload)
        assert p.verb == verb and p.actor == "stage7-1-bakeoff" and env.reply_to == reply
        assert reply.startswith(gp.GPU_POOL_CONTROL_REPLY_PREFIX) and env.kind == gp.GPU_POOL_CONTROL_KIND
    assert gp.GpuPoolControlV1.model_validate(ctl.build("hold")[1].payload).work_class == "diffusion"
    assert gp.GpuPoolControlV1.model_validate(ctl.build("release", lease_id="L1")[1].payload).lease_id == "L1"
    with pytest.raises(ValueError):
        ctl.build("release")

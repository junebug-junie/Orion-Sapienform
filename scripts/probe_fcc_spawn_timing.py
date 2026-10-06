"""Measure claude -p spawn -> init / first API request, with no model call.

Runs inside orion-athena-harness-governor. ANTHROPIC_BASE_URL is pointed at a
local stub that records every request and answers 503 to /v1/messages, so no
model is ever reached. The process is killed as soon as the first
/v1/messages request lands (or 60 s).

Read-only diagnostic for spec 2026-10-06-unified-turn-latency-design.md L5.
Results: docs/superpowers/pr-reports/2026-10-06-fcc-spawn-timing-measurement.md

  docker cp scripts/probe_fcc_spawn_timing.py orion-athena-harness-governor:/tmp/fcc_spawn_probe.py
  docker exec orion-athena-harness-governor python3 /tmp/fcc_spawn_probe.py 7
  docker exec -e PROBE_PER_SERVER=1 orion-athena-harness-governor python3 /tmp/fcc_spawn_probe.py 6

Writes temp config dirs under /tmp/fcc-probe-cfg and results to
/tmp/fcc-probe-result.json inside the container; never modifies /root/.claude.
"""
import json, os, shutil, statistics, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/app")
from orion.harness import fcc_motor as m
from orion.fcc.claude_spawn import setting_sources_argv, extend_mcp_argv, claude_permission_argv

STUB_PORT = 18999
REQS = []  # (t, method, path)


class H(BaseHTTPRequestHandler):
    def _rec(self):
        REQS.append((time.monotonic(), self.command, self.path))
        n = int(self.headers.get("content-length") or 0)
        if n:
            self.rfile.read(n)
        if self.path.startswith("/v1/models"):
            body = b'{"data":[],"has_more":false}'
            self.send_response(200)
        else:
            body = b'{"type":"error","error":{"type":"overloaded_error","message":"probe"}}'
            self.send_response(503)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_HEAD = _rec

    def log_message(self, *a):
        pass


srv = ThreadingHTTPServer(("127.0.0.1", STUB_PORT), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
STUB = f"http://127.0.0.1:{STUB_PORT}"

WORKSPACE = os.environ.get("HARNESS_FCC_WORKSPACE", "/mnt/orion-fcc/repo")
env_fcc = m.load_fcc_env(m.expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env")))
MODEL = m.label_to_claude_model_id(m.DEFAULT_FCC_MODEL_LABEL, env_fcc)
PROMPT = "fcc spawn timing probe (no reply needed)"


def build_argv(with_mcp, with_breaker: bool):
    argv = [os.environ.get("HARNESS_FCC_CLAUDE_BIN", "claude"), "-p", PROMPT,
            "--output-format", "stream-json", "--verbose", "--model", MODEL]
    argv.extend(setting_sources_argv("HARNESS_FCC_SETTING_SOURCES"))
    if with_breaker:
        argv.extend(m.repeat_failure_breaker_argv(correlation_id="probe"))
    if with_mcp:
        p = m._maybe_render_mcp_config(correlation_id="fcc-spawn-probe")
        if p is not None and isinstance(with_mcp, list):
            from pathlib import Path
            data = json.loads(Path(p).read_text())
            data["mcpServers"] = {k: v for k, v in data["mcpServers"].items() if k in with_mcp}
            p = Path(f"/tmp/fcc-probe-cfg/mcp-{'_'.join(with_mcp) or 'none'}.json")
            p.write_text(json.dumps(data))
        if p is not None:
            extra = ["mcp__plugin_context-mode_context-mode"] if m._env_truthy("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED") else None
            extend_mcp_argv(argv, p, extra_allowed_tools=extra)
    perm = claude_permission_argv(auto_approve=True)
    i = argv.index("--model")
    for off, tok in enumerate(perm):
        argv.insert(i + off, tok)
    return argv


def build_env(config_dir=None):
    env = m._build_subprocess_env(fcc_server_url=STUB, auth_token="probe", fcc_env=env_fcc,
                                  turn_budget_sec=7200, turn_deadline_epoch=time.time() + 7200,
                                  turn_step_stall_sec=420, n_ctx=None, gpu_lease=None, chat_reply=True)
    env["ANTHROPIC_BASE_URL"] = STUB
    env["ANTHROPIC_AUTH_TOKEN"] = "probe"
    env.pop("ANTHROPIC_API_KEY", None)
    if config_dir:
        env["CLAUDE_CONFIG_DIR"] = config_dir
    return env


def make_cfg(name, settings):
    d = f"/tmp/fcc-probe-cfg/{name}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d + "/plugins")
    with open(d + "/settings.json", "w") as f:
        json.dump(settings, f)
    src = "/root/.claude"
    shutil.copytree(src + "/hooks", d + "/hooks")
    for fn in ("installed_plugins.json", "known_marketplaces.json"):
        if os.path.exists(f"{src}/plugins/{fn}"):
            shutil.copy(f"{src}/plugins/{fn}", f"{d}/plugins/{fn}")
    for sub in ("cache", "marketplaces", "data"):
        if os.path.exists(f"{src}/plugins/{sub}"):
            os.symlink(f"{src}/plugins/{sub}", f"{d}/plugins/{sub}")
    return d


def run_once(argv, env):
    REQS.clear()
    t0 = time.monotonic()
    p = subprocess.Popen(argv, cwd=WORKSPACE, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    marks = {}
    events = []
    done = threading.Event()

    def reader():
        for line in p.stdout:
            t = time.monotonic() - t0
            try:
                ev = json.loads(line)
            except Exception:
                continue
            key = f"{ev.get('type')}/{ev.get('subtype','')}"
            events.append((round(t, 3), key, ev.get("hook_name") or ""))
            if key == "system/init":
                marks.setdefault("init", t)
                marks["mcp"] = [(s.get("name"), s.get("status")) for s in ev.get("mcp_servers", [])]
                marks["session_id"] = ev.get("session_id")
            if "first_event" not in marks:
                marks["first_event"] = t
        done.set()

    threading.Thread(target=reader, daemon=True).start()
    deadline = t0 + 60
    while time.monotonic() < deadline:
        if any(r[1] == "POST" and r[2].startswith("/v1/messages") for r in REQS):
            break
        if done.is_set():
            break
        time.sleep(0.005)
    msg = [r for r in REQS if r[1] == "POST" and r[2].startswith("/v1/messages")]
    marks["first_messages_req"] = (msg[0][0] - t0) if msg else None
    marks["pre_init_reqs"] = [(round(r[0] - t0, 3), r[1], r[2]) for r in REQS
                              if "init" in marks and r[0] - t0 < marks["init"]]
    p.kill()
    p.wait()
    time.sleep(0.2)
    marks["events"] = events[:12]
    return marks


def pct(xs, q):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    real = json.load(open("/root/.claude/settings.json"))
    no_hook = {k: v for k, v in real.items() if k != "hooks"}
    no_plugin = dict(real, enabledPlugins={"context-mode@context-mode": False})
    variants = {
        "a_motor_exact_real_cfg": (None, True, True),
        "a2_copy_cfg_control": (make_cfg("full", real), True, True),
        "b_no_sessionstart_hook": (make_cfg("nohook", no_hook), True, True),
        "c_no_plugin": (make_cfg("noplugin", no_plugin), True, True),
        "d_minimal_cfg": (make_cfg("min", {}), True, True),
        "e_minimal_cfg_no_mcp_no_breaker": (make_cfg("bare", {}), False, False),
    }
    if os.environ.get("PROBE_PER_SERVER"):
        p = m._maybe_render_mcp_config(correlation_id="fcc-spawn-probe")
        names = list(json.loads(open(p).read())["mcpServers"])
        bare = make_cfg("bare", {})
        variants = {"e_bare_breaker_only": (bare, [], True)}
        for nm in names:
            variants[f"f_only_{nm}"] = (bare, [nm], True)
        variants["g_motor_minus_aitown"] = (None, [x for x in names if x != "orion-aitown"], True)
        variants["h_min_cfg_minus_aitown"] = (make_cfg("min", {}), [x for x in names if x != "orion-aitown"], True)
    out = {}
    order = list(variants)
    runs = {k: [] for k in order}
    for i in range(n):  # interleave variants to spread host-load noise
        for k in order:
            cfg, mcp, brk = variants[k]
            r = run_once(build_argv(mcp, brk), build_env(cfg))
            runs[k].append(r)
            print(k, i, round(r.get("init") or -1, 3), round(r["first_messages_req"] or -1, 3), flush=True)
    for k in order:
        inits = [r["init"] for r in runs[k] if r.get("init")]
        msgs = [r["first_messages_req"] for r in runs[k] if r["first_messages_req"]]
        out[k] = {
            "n": len(runs[k]),
            "init_median": statistics.median(inits) if inits else None,
            "init_p90": pct(inits, 0.9) if inits else None,
            "msg_median": statistics.median(msgs) if msgs else None,
            "msg_p90": pct(msgs, 0.9) if msgs else None,
            "init_all": [round(x, 3) for x in inits],
            "msg_all": [round(x, 3) for x in msgs],
            "sample": {kk: runs[k][-1].get(kk) for kk in ("mcp", "pre_init_reqs", "events", "session_id")},
            "session_ids": [r.get("session_id") for r in runs[k]],
        }
    json.dump(out, open("/tmp/fcc-probe-result.json", "w"), indent=1, default=str)
    print(json.dumps({k: {kk: v[kk] for kk in ("init_median", "init_p90", "msg_median", "msg_p90")} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()

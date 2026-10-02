"""Spawn claude -p exactly as orion/harness/fcc_motor.run_fcc_turn builds it,
except ANTHROPIC_BASE_URL -> recording proxy and no per-turn MCP config."""
import os, subprocess, sys
sys.path.insert(0, "/app")
from orion.harness import fcc_motor as m
from orion.fcc.claude_spawn import setting_sources_argv, claude_permission_argv
prompt = open("/tmp/fccprobe/prompt.txt").read()
argv = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--model", "llamacpp/agent"]
argv += setting_sources_argv("HARNESS_FCC_SETTING_SOURCES")
argv += m.repeat_failure_breaker_argv(correlation_id="fccprobe")
perm = claude_permission_argv(auto_approve=True)
i = argv.index("--model")
for k, t in enumerate(perm): argv.insert(i + k, t)
env = m._build_subprocess_env(fcc_server_url="http://127.0.0.1:18999", auth_token="probe", n_ctx=131072)
env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:18999"
print("ARGV", [a if len(a) < 200 else a[:80] + "..." for a in argv], file=sys.stderr)
r = subprocess.run(argv, cwd=os.environ.get("WS", "/mnt/orion-fcc/repo"), env=env, stdout=open("/tmp/fccprobe/stream.jsonl", "w"), timeout=1500)
print("exit", r.returncode, file=sys.stderr)
